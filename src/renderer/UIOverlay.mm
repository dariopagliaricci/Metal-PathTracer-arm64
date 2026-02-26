#import "renderer/UIOverlay.h"

#include "MetalShaderTypes.h"

#include "imgui.h"
#include "imgui_internal.h"
#include "backends/imgui_impl_metal.h"
#include "backends/imgui_impl_osx.h"
#include "ImGuizmo.h"

#import <AppKit/AppKit.h>


#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <TargetConditionals.h>

namespace {
std::string GenerateRenderFilename() {
    using namespace std::chrono;
    auto now = system_clock::now();
    std::time_t t = system_clock::to_time_t(now);
    std::tm tmStruct;
    localtime_r(&t, &tmStruct);
    char buffer[32];
    std::strftime(buffer, sizeof(buffer), "%Y%m%d-%H%M%S", &tmStruct);
    std::string filename = "renders/render-";
    filename += buffer;
    filename += ".exr";
    return filename;
}

constexpr float kPi = 3.14159265358979323846f;
constexpr float kRadToDeg = 180.0f / kPi;
constexpr float kDegToRad = kPi / 180.0f;

float WrapDegrees(float degrees) {
    degrees = std::fmod(degrees + 180.0f, 360.0f);
    if (degrees < 0.0f) {
        degrees += 360.0f;
    }
    return degrees - 180.0f;
}

namespace fs = std::filesystem;

struct CameraMatrices {
    simd::float4x4 view = matrix_identity_float4x4;
    simd::float4x4 projection = matrix_identity_float4x4;
};

CameraMatrices BuildCameraMatrices(const PathTracer::RenderSettings& settings,
                                   CGSize viewportSize) {
    CameraMatrices matrices{};
    const float distance = std::max(settings.cameraDistance, 0.1f);
    const float yaw = settings.cameraYaw;
    const float pitch = settings.cameraPitch;
    const float cosPitch = std::cos(pitch);
    const float sinPitch = std::sin(pitch);
    const float cosYaw = std::cos(yaw);
    const float sinYaw = std::sin(yaw);

    const simd::float3 lookAt = settings.cameraTarget;
    const simd::float3 offset = {
        distance * cosPitch * cosYaw,
        distance * sinPitch,
        distance * cosPitch * sinYaw
    };
    const simd::float3 eye = lookAt + offset;

    const simd::float3 forward = simd::normalize(lookAt - eye);
    const simd::float3 upReference = {0.0f, 1.0f, 0.0f};
    simd::float3 right = simd::normalize(simd::cross(forward, upReference));
    if (!std::isfinite(right.x) || simd::length(right) < 1.0e-4f) {
        right = {1.0f, 0.0f, 0.0f};
    }
    const simd::float3 up = simd::normalize(simd::cross(right, forward));
    const simd::float3 back = -forward;

    matrices.view.columns[0] = simd_make_float4(right.x, up.x, back.x, 0.0f);
    matrices.view.columns[1] = simd_make_float4(right.y, up.y, back.y, 0.0f);
    matrices.view.columns[2] = simd_make_float4(right.z, up.z, back.z, 0.0f);
    matrices.view.columns[3] = simd_make_float4(-simd::dot(right, eye),
                                                -simd::dot(up, eye),
                                                -simd::dot(back, eye),
                                                1.0f);

    const float aspect =
        (viewportSize.height <= 0.0f) ? 1.0f : static_cast<float>(viewportSize.width / viewportSize.height);
    const float vfov = std::clamp(settings.cameraVerticalFov, 1.0f, 179.0f) * kDegToRad;
    const float nearZ = 0.1f;
    const float farZ = 5000.0f;
    const float f = 1.0f / std::tan(vfov * 0.5f);

    matrices.projection.columns[0] = simd_make_float4(f / aspect, 0.0f, 0.0f, 0.0f);
    matrices.projection.columns[1] = simd_make_float4(0.0f, f, 0.0f, 0.0f);
    matrices.projection.columns[2] = simd_make_float4(0.0f,
                                                      0.0f,
                                                      (farZ + nearZ) / (nearZ - farZ),
                                                      -1.0f);
    matrices.projection.columns[3] = simd_make_float4(0.0f,
                                                      0.0f,
                                                      (2.0f * farZ * nearZ) / (nearZ - farZ),
                                                      0.0f);
    return matrices;
}

struct EnvironmentFileEntry {
    std::string displayName;
    std::string absolutePath;
    std::string canonicalPath;
};

std::string NormalizePath(const std::string& path) {
    if (path.empty()) {
        return {};
    }
    std::error_code ec;
    fs::path input(path);
    fs::path canonical = fs::weakly_canonical(input, ec);
    if (!ec) {
        return canonical.string();
    }
    fs::path normalized = input.lexically_normal();
    return normalized.string();
}

bool HasEnvironmentExtension(const fs::path& filePath) {
    std::string ext = filePath.extension().string();
    std::string lower;
    lower.resize(ext.size());
    std::transform(ext.begin(), ext.end(), lower.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return lower == ".hdr" || lower == ".exr";
}

std::optional<fs::path> ResolveAssetsDirectory() {
#if TARGET_OS_OSX
    @autoreleasepool {
        NSBundle* bundle = [NSBundle mainBundle];
        if (bundle) {
            NSString* resourcePath = [bundle resourcePath];
            if (resourcePath) {
                fs::path candidate([resourcePath UTF8String]);
                candidate /= "assets";
                std::error_code ec;
                fs::path canonical = fs::weakly_canonical(candidate, ec);
                if (!ec && fs::exists(canonical, ec) && fs::is_directory(canonical, ec)) {
                    return canonical;
                }
            }
        }
    }
#endif

    std::error_code ec;
    fs::path cwd = fs::current_path(ec);
    if (!ec) {
        fs::path candidate = cwd / "assets";
        fs::path canonical = fs::weakly_canonical(candidate, ec);
        if (!ec && fs::exists(canonical, ec) && fs::is_directory(canonical, ec)) {
            return canonical;
        }
    }
    return std::nullopt;
}

std::vector<EnvironmentFileEntry> EnumerateEnvironmentFiles() {
    std::vector<EnvironmentFileEntry> files;
    auto assetsDir = ResolveAssetsDirectory();
    if (!assetsDir) {
        return files;
    }

    std::error_code ec;
    fs::recursive_directory_iterator it(*assetsDir,
                                        fs::directory_options::skip_permission_denied,
                                        ec);
    fs::recursive_directory_iterator end;
    for (; it != end && !ec; it.increment(ec)) {
        if (ec) {
            break;
        }
        const auto& entry = *it;
        if (ec) {
            break;
        }
        if (!entry.is_regular_file(ec)) {
            continue;
        }
        if (!HasEnvironmentExtension(entry.path())) {
            continue;
        }
        EnvironmentFileEntry fileEntry;
        fileEntry.displayName = entry.path().filename().string();
        fileEntry.absolutePath = entry.path().string();
        std::error_code canonEc;
        fs::path canonical = fs::weakly_canonical(entry.path(), canonEc);
        if (!canonEc) {
            fileEntry.canonicalPath = canonical.string();
        } else {
            fileEntry.canonicalPath = NormalizePath(fileEntry.absolutePath);
        }
        files.push_back(std::move(fileEntry));
    }

    std::sort(files.begin(),
              files.end(),
              [](const EnvironmentFileEntry& a, const EnvironmentFileEntry& b) {
                  return a.displayName < b.displayName;
              });
    return files;
}

int FindEnvironmentSelection(const std::vector<EnvironmentFileEntry>& files,
                             const std::string& currentPath) {
    if (files.empty() || currentPath.empty()) {
        return -1;
    }
    std::string normalized = NormalizePath(currentPath);
    for (int i = 0; i < static_cast<int>(files.size()); ++i) {
        if (NormalizePath(files[i].canonicalPath) == normalized) {
            return i;
        }
    }
    return -1;
}

bool ApplyEnvironmentSelection(PathTracer::RenderSettings& settings, const std::string& newPath) {
    if (settings.environmentMapPath == newPath) {
        return false;
    }
    settings.environmentMapPath = newPath;
    settings.environmentMapDirty = true;
    if (newPath.empty()) {
        settings.backgroundMode = PathTracer::RenderSettings::BackgroundMode::Gradient;
    } else {
        settings.backgroundMode = PathTracer::RenderSettings::BackgroundMode::Environment;
    }
    return true;
}

void DrawEnvironmentPicker(PathTracer::RenderSettings& settings) {
    std::vector<EnvironmentFileEntry> envFiles = EnumerateEnvironmentFiles();
    const int selectedIndex = FindEnvironmentSelection(envFiles, settings.environmentMapPath);
    const char* previewValue =
        (selectedIndex >= 0 && selectedIndex < static_cast<int>(envFiles.size()))
            ? envFiles[static_cast<size_t>(selectedIndex)].displayName.c_str()
            : "<none>";

    if (ImGui::BeginCombo("Environment##env_picker", previewValue)) {
        const bool noneSelected = (selectedIndex < 0);
        if (ImGui::Selectable("<none>", noneSelected)) {
            if (!noneSelected) {
                ApplyEnvironmentSelection(settings, "");
            }
        }
        if (!envFiles.empty()) {
            for (size_t i = 0; i < envFiles.size(); ++i) {
                const bool isSelected = (static_cast<int>(i) == selectedIndex);
                if (ImGui::Selectable(envFiles[i].displayName.c_str(), isSelected)) {
                    if (!isSelected) {
                        const std::string& resolved =
                            envFiles[i].canonicalPath.empty()
                                ? envFiles[i].absolutePath
                                : envFiles[i].canonicalPath;
                        ApplyEnvironmentSelection(settings, resolved);
                    }
                }
                if (isSelected) {
                    ImGui::SetItemDefaultFocus();
                }
            }
        } else {
            ImGui::Selectable("No .hdr/.exr files found", false, ImGuiSelectableFlags_Disabled);
        }
        ImGui::EndCombo();
    }
}

}  // namespace

namespace PathTracer {

using PathTracerShaderTypes::MaterialType;

UIOverlay::~UIOverlay() {
    shutdown();
}

bool UIOverlay::consumeSaveExrRequest(std::string& outPath) {
    if (m_pendingExrPath.empty()) {
        return false;
    }
    outPath = m_pendingExrPath;
    m_pendingExrPath.clear();
    return true;
}

void UIOverlay::notifySaveExrResult(bool success, const std::string& message) {
    m_lastSaveExrSuccess = success;
    m_lastSaveExrMessage = message;
}

void UIOverlay::promptSaveExr() {
    if (m_savePanelActive) {
        return;
    }
    std::string autoPath = GenerateRenderFilename();
    m_pendingExrPath = autoPath;
    m_savePanelActive = false;
    m_lastSaveExrMessage = "Saving to " + autoPath;
    m_lastSaveExrSuccess = true;
}

void UIOverlay::drawSaveStatus() {
    if (m_lastSaveExrMessage.empty()) {
        ImGui::TextDisabled("Exports linear HDR EXR at render resolution.");
        return;
    }
    ImVec4 color = m_lastSaveExrSuccess ? ImVec4(0.55f, 0.85f, 0.55f, 1.0f)
                                        : ImVec4(0.95f, 0.45f, 0.45f, 1.0f);
    ImGui::TextColored(color, "%s", m_lastSaveExrMessage.c_str());
}

bool UIOverlay::initialize(NSViewHandle view, MTLDeviceHandle device, float contentScale) {
    if (m_initialized) {
        return true;
    }

    m_view = nullptr;
    if (!view || !device) {
        return false;
    }

    m_view = view;
    m_device = device;
    m_backingScale = std::max(contentScale, 1.0f);
    m_displayScale = m_hasForcedContentScale ? m_forcedContentScale : m_backingScale;
    if (view) {
        m_displaySizePoints = view.bounds.size;
    } else {
        m_displaySizePoints = CGSizeMake(1.0, 1.0);
    }
    m_fontScaleInitialized = false;
    m_pendingFontRebuild = false;

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigMacOSXBehaviors = false;  // Preserve Ctrl-based interactions (e.g., Ctrl+Click for text input)
    ImGui::StyleColorsDark();

    // Custom dark theme
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 6.0f;
    style.FrameRounding = 4.0f;
    style.GrabRounding = 4.0f;
    style.FramePadding = ImVec2(6.0f, 4.0f);
    style.WindowPadding = ImVec2(10.0f, 8.0f);
    style.ItemSpacing = ImVec2(8.0f, 6.0f);
    style.Colors[ImGuiCol_WindowBg] = ImVec4(0.06f, 0.06f, 0.09f, 0.95f);
    style.Colors[ImGuiCol_Header] = ImVec4(0.11f, 0.28f, 0.50f, 0.80f);
    style.Colors[ImGuiCol_HeaderHovered] = ImVec4(0.16f, 0.40f, 0.70f, 0.90f);
    style.Colors[ImGuiCol_HeaderActive] = ImVec4(0.20f, 0.48f, 0.80f, 1.00f);
    style.Colors[ImGuiCol_Button] = ImVec4(0.12f, 0.35f, 0.65f, 0.90f);
    style.Colors[ImGuiCol_ButtonHovered] = ImVec4(0.18f, 0.45f, 0.78f, 0.95f);
    style.Colors[ImGuiCol_ButtonActive] = ImVec4(0.22f, 0.55f, 0.90f, 1.00f);
    style.Colors[ImGuiCol_FrameBg] = ImVec4(0.10f, 0.18f, 0.32f, 0.85f);
    style.Colors[ImGuiCol_FrameBgHovered] = ImVec4(0.15f, 0.28f, 0.46f, 0.90f);
    style.Colors[ImGuiCol_FrameBgActive] = ImVec4(0.18f, 0.34f, 0.55f, 0.95f);
    style.Colors[ImGuiCol_TitleBg] = ImVec4(0.08f, 0.10f, 0.18f, 1.00f);
    style.Colors[ImGuiCol_TitleBgActive] = ImVec4(0.12f, 0.20f, 0.36f, 1.00f);
    style.Colors[ImGuiCol_SliderGrab] = ImVec4(0.26f, 0.62f, 0.90f, 0.90f);
    style.Colors[ImGuiCol_SliderGrabActive] = ImVec4(0.32f, 0.70f, 1.00f, 1.00f);

    // Let the platform backends handle Retina/HiDPI scaling via DisplayFramebufferScale.
    // ImGui sizes are authored in logical points, so we keep the style at its authored size.
    io.FontGlobalScale = 1.0f;

    if (!ImGui_ImplOSX_Init(view)) {
        ImGui::DestroyContext();
        m_view = nullptr;
        return false;
    }

    if (!ImGui_ImplMetal_Init(device)) {
        ImGui_ImplOSX_Shutdown();
        ImGui::DestroyContext();
        m_view = nullptr;
        return false;
    }

    m_initialized = true;
    updateDisplayMetrics(m_displaySizePoints, m_displayScale);
    return true;
}

void UIOverlay::updateDisplayMetrics(CGSize logicalSizePoints, float backingScale) {
    const CGFloat width = std::max<CGFloat>(logicalSizePoints.width, 1.0);
    const CGFloat height = std::max<CGFloat>(logicalSizePoints.height, 1.0);
    m_displaySizePoints = CGSizeMake(width, height);
    m_backingScale = std::max(backingScale, 1.0f);
    const float effectiveScale = m_hasForcedContentScale ? m_forcedContentScale : m_backingScale;
    m_displayScale = std::max(effectiveScale, 1.0f);

    if (!m_fontScaleInitialized) {
        m_fontScaleInitialized = true;
        m_lastFontScale = m_displayScale;
        m_pendingFontScale = m_displayScale;
        m_pendingFontRebuild = true;
        return;
    }

    const float denom = std::max(m_lastFontScale, 1e-3f);
    const float delta = std::fabs(m_displayScale - m_lastFontScale) / denom;
    if (delta >= 0.05f) {
        requestFontRebuild(m_displayScale);
    }
}

void UIOverlay::setForcedContentScale(float scale, bool enabled) {
    if (enabled) {
        m_hasForcedContentScale = true;
        m_forcedContentScale = std::max(scale, 1.0f);
    } else {
        m_hasForcedContentScale = false;
    }
    updateDisplayMetrics(m_displaySizePoints, m_backingScale);
}

void UIOverlay::setScenePanelProvider(ScenePanelProvider provider) {
    m_scenePanelProvider = std::move(provider);
    m_hasScenePanelProvider = true;
}

void UIOverlay::requestFontRebuild(float scale) {
    m_pendingFontScale = std::max(scale, 1.0f);
    m_pendingFontRebuild = true;
}

void UIOverlay::rebuildFonts(float scale) {
    if (!m_initialized || !m_device) {
        return;
    }

    ImGuiIO& io = ImGui::GetIO();
    io.Fonts->Clear();
    ImFontConfig config;
    const float normalizedScale = std::max(scale, 1.0f);
    config.SizePixels = std::max(10.0f, std::roundf(16.0f * normalizedScale));
    io.Fonts->AddFontDefault(&config);
    io.FontGlobalScale = 1.0f / normalizedScale;

    ImGui_ImplMetal_DestroyFontsTexture();
    if (!ImGui_ImplMetal_CreateFontsTexture(m_device)) {
        NSLog(@"[UIOverlay] Failed to rebuild font atlas (scale %.2f)", scale);
        return;
    }

    m_lastFontScale = normalizedScale;
    NSLog(@"[UIOverlay] Rebuilt font atlas for scale %.2f", normalizedScale);
}

void UIOverlay::rebuildFontsIfNeeded() {
    if (!m_pendingFontRebuild) {
        return;
    }
    const float targetScale = std::max(m_pendingFontScale, 1.0f);
    rebuildFonts(targetScale);
    m_pendingFontRebuild = false;
}

void UIOverlay::applyDisplayState() {
    if (!m_initialized) {
        return;
    }
    if (!m_hasForcedContentScale) {
        return;  // Trust ImGui backend display metrics unless explicitly overridden.
    }
    ImGuiIO& io = ImGui::GetIO();
    io.DisplaySize = ImVec2(static_cast<float>(m_displaySizePoints.width),
                            static_cast<float>(m_displaySizePoints.height));
    io.DisplayFramebufferScale = ImVec2(m_displayScale, m_displayScale);
}

void UIOverlay::beginFrame(MTLRenderPassDescriptorHandle renderPassDescriptor) {
    if (!m_initialized) {
        return;
    }
    
    rebuildFontsIfNeeded();
    ImGui_ImplMetal_NewFrame(renderPassDescriptor);
    ImGui_ImplOSX_NewFrame(m_view);
    applyDisplayState();
    ImGui::NewFrame();
}

void UIOverlay::buildUI(const PerformanceStats& stats,
                        RenderSettings& settings,
                        bool& outNeedsReset,
                        const std::vector<const char*>& sceneNames,
                        int& inOutSelectedScene,
                        PresentationSettings& presentation) {
    if (!m_initialized || !m_visible) {
        return;
    }

    outNeedsReset = false;

    const float fps = (stats.frameTimeMs > 0.0) ? static_cast<float>(1000.0 / stats.frameTimeMs) : 0.0f;

    if (m_mainPanelVisible) {
        ImGui::SetNextWindowPos(ImVec2(20.0f, 20.0f), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(350.0f, 0.0f), ImGuiCond_FirstUseEver);

        if (ImGui::Begin("Metal PathTracer")) {
            ImGui::Text("Frame Time: %.3f ms", stats.frameTimeMs);
            ImGui::Text("FPS: %.1f", fps);
            ImGui::Text("Sample Count: %u", stats.sampleCount);
            ImGui::Separator();

        ImGui::SetNextItemOpen(false, ImGuiCond_Once);
        if (ImGui::CollapsingHeader("Output / Export", ImGuiTreeNodeFlags_None)) {
            if (ImGui::Button("Save EXR...")) {
                promptSaveExr();
            }
            drawSaveStatus();
        }

        ImGui::SetNextItemOpen(false, ImGuiCond_Once);
        if (ImGui::CollapsingHeader("Performance")) {
            const bool renderScaleLocked =
                presentation.enabled &&
                presentation.resolutionLock != PresentationSettings::RenderResolutionLock::Off;
            if (renderScaleLocked) {
                ImGui::TextDisabled("Render Scale: locked by Presentation Mode");
            } else {
                float renderScale = settings.renderScale;
                if (ImGui::SliderFloat("Render Scale", &renderScale, 0.5f, 2.0f, "%.2fx")) {
                    settings.renderScale = std::clamp(renderScale, 0.5f, 2.0f);
                    outNeedsReset = true;
                }
            }
            ImGui::Text("Internal Resolution: %u x %u", stats.renderWidth, stats.renderHeight);
            ImGui::Text("Effective Scale: %.2fx", stats.renderScale);
            ImGui::Text("Samples / Frame: %u (active %u)",
                        settings.samplesPerFrame,
                        stats.activeSamplesPerFrame);
            ImGui::Text("Scene: %u spheres, %u triangles", stats.sphereCount, stats.triangleCount);
            ImGui::Text("BVH: %u nodes, %u prims", stats.bvhNodeCount, stats.bvhPrimitiveCount);
            ImGui::Text("BVH Avg Nodes / Ray: %.2f", stats.avgNodesVisited);
            ImGui::Text("BVH Avg Prim Tests / Ray: %.2f", stats.avgLeafPrimTests);
            ImGui::Text("BVH Shadow Early Exit: %.1f%%", stats.shadowRayEarlyExitPct);
            ImGui::Text("BVH Both Children Visited: %.1f%%", stats.bothChildrenVisitedPct);
            ImGui::Text("GPU Time: %.3f ms", stats.gpuTimeMs);
            ImGui::Text("CPU Encode: %.3f ms", stats.cpuEncodeMs);
            ImGui::Text("Drawable Wait: %.3f ms", stats.drawableWaitMs);
            ImGui::Text("Camera Motion: %s", stats.cameraMotionActive ? "Active" : "Idle");
            ImGui::Text("SPP / Min: %.1f", stats.samplesPerMinute);
            ImGui::Text("Progress: %s", "Infinite samples");
            const char* intersectionLabel =
                (stats.intersectionMode == static_cast<uint32_t>(PathTracerShaderTypes::IntersectionMode::HardwareRayTracing))
                    ? "Hardware Ray Tracing"
                    : "Software BVH";
            ImGui::Text("Intersection: %s", intersectionLabel);
            if (stats.hardwareRayCount > 0) {
                ImGui::Text("HW Rays: %llu (Hit %.2f%% / Miss %.2f%%)",
                            static_cast<unsigned long long>(stats.hardwareRayCount),
                            stats.hardwareRayPctHit,
                            stats.hardwareRayPctMiss);
                ImGui::Text("  None: %llu  Rejected: %llu  Unavailable: %llu",
                            static_cast<unsigned long long>(stats.hardwareResultNoneCount),
                            static_cast<unsigned long long>(stats.hardwareRejectedCount),
                            static_cast<unsigned long long>(stats.hardwareUnavailableCount));
                ImGui::Text("  Specular NEE occluded: %llu  Self-hit rejects: %llu",
                            static_cast<unsigned long long>(stats.specularNeeOcclusionHitCount),
                            static_cast<unsigned long long>(stats.hardwareSelfHitRejectedCount));
                if (stats.hardwareMissCount > 0) {
                    ImGui::Text("  Miss distances (log2 bins up to %u):", 32);
                    const float totalMisses = static_cast<float>(stats.hardwareMissCount);
                    for (int i = 0; i < 32; ++i) {
                        uint64_t binCount = stats.hardwareMissDistanceBins[i];
                        if (binCount == 0) {
                            continue;
                        }
                        float binStart = powf(2.0f, static_cast<float>(i - 8));
                        float binEnd = powf(2.0f, static_cast<float>(i - 7));
                        ImGui::Text("    [%.4f, %.4f): %.2f%%",
                                    binStart,
                                    binEnd,
                                    (static_cast<float>(binCount) / totalMisses) * 100.0f);
                    }
                    ImGui::Text("  Last miss distance: %.4f (m-ish units)", stats.hardwareMissLastDistance);
                    if (stats.hardwareMissLastInstanceId != 0xffffffffu) {
                        ImGui::Text("  Last miss source: inst=%u prim=%u",
                                    stats.hardwareMissLastInstanceId,
                                    stats.hardwareMissLastPrimitiveId);
                    } else {
                        ImGui::Text("  Last miss source: n/a");
                    }
                }
                ImGui::Text("  Exclusion retries: 0=%llu 1=%llu 2=%llu 3+=%llu",
                            static_cast<unsigned long long>(stats.hardwareExcludeRetryHistogram[0]),
                            static_cast<unsigned long long>(stats.hardwareExcludeRetryHistogram[1]),
                            static_cast<unsigned long long>(stats.hardwareExcludeRetryHistogram[2]),
                            static_cast<unsigned long long>(stats.hardwareExcludeRetryHistogram[3]));
                ImGui::Text("  SWRT fallback hits: miss=%llu firstHit=%llu",
                            static_cast<unsigned long long>(stats.hardwareFallbackHitCount),
                            static_cast<unsigned long long>(stats.hardwareFirstHitFallbackCount));
                ImGui::Text("  Last self-hit distance: %.4f", stats.hardwareSelfHitLastDistance);
                ImGui::Text("  Last: type=%u dist=%.4f inst=%u prim=%u",
                            stats.hardwareLastResultType,
                            stats.hardwareLastDistance,
                            stats.hardwareLastInstanceId,
                            stats.hardwareLastPrimitiveId);
            } else {
                ImGui::Text("HW Rays: n/a");
            }
        }

        ImGui::SetNextItemOpen(false, ImGuiCond_Once);
        if (ImGui::CollapsingHeader("Render Settings")) {
            int samplesPerFrame = static_cast<int>(settings.samplesPerFrame);
            if (ImGui::SliderInt("Samples / Frame", &samplesPerFrame, 1, 16)) {
                settings.samplesPerFrame = static_cast<uint32_t>(std::max(samplesPerFrame, 1));
                outNeedsReset = true;
            }

            bool rrEnabled = settings.enableRussianRoulette;
            if (ImGui::Checkbox("Russian Roulette", &rrEnabled)) {
                settings.enableRussianRoulette = rrEnabled;
                outNeedsReset = true;
            }
            if (stats.hardwareRaytracingAvailable) {
                bool softwareOverride = settings.enableSoftwareRayTracing;
                if (ImGui::Checkbox("Software Ray Tracing", &softwareOverride)) {
                    settings.enableSoftwareRayTracing = softwareOverride;
                    outNeedsReset = true;
                }
                ImGui::Text("Hardware RT %s", stats.hardwareRaytracingActive ? "active" : "available");
            } else {
                ImGui::TextDisabled("Hardware ray tracing unavailable on this device");
            }
        }

        ImGui::SetNextItemOpen(false, ImGuiCond_Once);
        if (ImGui::CollapsingHeader("Environment")) {
            DrawEnvironmentPicker(settings);

            float intensity = settings.environmentIntensity;
            if (ImGui::SliderFloat("Intensity", &intensity, 0.0f, 10.0f, "%.2f")) {
                settings.environmentIntensity = std::max(intensity, 0.0f);
                outNeedsReset = true;
            }

            float rotationDegrees = settings.environmentRotation * kRadToDeg;
            if (ImGui::SliderFloat("Rotation (deg)", &rotationDegrees, -180.0f, 180.0f, "%.1f")) {
                settings.environmentRotation = rotationDegrees * kDegToRad;
                outNeedsReset = true;
            }

            bool specNee = settings.enableSpecularNee;
            if (ImGui::Checkbox("Specular NEE (env + rect)", &specNee)) {
                settings.enableSpecularNee = specNee;
                outNeedsReset = true;
            }
            ImGui::TextDisabled("Connects delta bounces directly to environment and rect lights via MIS.");

            bool mneeEnabled = settings.enableMnee;
            if (ImGui::Checkbox("MNEE Caustics", &mneeEnabled)) {
                settings.enableMnee = mneeEnabled;
                outNeedsReset = true;
            }
            bool mneeSecondary = settings.enableMneeSecondary;
            ImGui::BeginDisabled(!settings.enableMnee);
            if (ImGui::Checkbox("MNEE Secondary Hop", &mneeSecondary)) {
                settings.enableMneeSecondary = mneeSecondary;
                outNeedsReset = true;
            }
            ImGui::EndDisabled();
            ImGui::TextDisabled("Single-bounce dielectric caustic connection for sharper refraction patterns.");
            ImGui::TextDisabled("Secondary hop adds an extra specular link for caustic refinement.");

            ImGui::TextWrapped("Rotate the environment map to steer its bright regions through portals or openings.");
        }

        ImGui::SetNextItemOpen(false, ImGuiCond_Once);
#if PT_DEBUG_TOOLS
        if (ImGui::CollapsingHeader("PBR Debug")) {
            auto setExclusiveDebugView = [&](bool& target, bool value) {
                if (value) {
                    settings.debugShowBaseColor = false;
                    settings.debugShowMetallic = false;
                    settings.debugShowRoughness = false;
                    settings.debugShowAO = false;
                }
                target = value;
            };

            bool showBase = settings.debugShowBaseColor;
            if (ImGui::Checkbox("Show BaseColor (linear)", &showBase)) {
                setExclusiveDebugView(settings.debugShowBaseColor, showBase);
                outNeedsReset = true;
            }
            bool showMetallic = settings.debugShowMetallic;
            if (ImGui::Checkbox("Show Metallic (from ORM)", &showMetallic)) {
                setExclusiveDebugView(settings.debugShowMetallic, showMetallic);
                outNeedsReset = true;
            }
            bool showRoughness = settings.debugShowRoughness;
            if (ImGui::Checkbox("Show Roughness (from ORM)", &showRoughness)) {
                setExclusiveDebugView(settings.debugShowRoughness, showRoughness);
                outNeedsReset = true;
            }
            bool showAO = settings.debugShowAO;
            if (ImGui::Checkbox("Show AO", &showAO)) {
                setExclusiveDebugView(settings.debugShowAO, showAO);
                outNeedsReset = true;
            }

            bool disableAO = settings.debugDisableAO;
            if (ImGui::Checkbox("Disable AO", &disableAO)) {
                settings.debugDisableAO = disableAO;
                outNeedsReset = true;
            }
            bool aoIndirectOnly = settings.debugAoIndirectOnly;
            if (ImGui::Checkbox("AO Indirect Only", &aoIndirectOnly)) {
                settings.debugAoIndirectOnly = aoIndirectOnly;
                outNeedsReset = true;
            }

            bool disableNormal = settings.debugDisableNormalMap;
            if (ImGui::Checkbox("Disable Normal Map", &disableNormal)) {
                settings.debugDisableNormalMap = disableNormal;
                outNeedsReset = true;
            }
            bool disableOrm = settings.debugDisableOrmTexture;
            if (ImGui::Checkbox("Disable ORM Texture", &disableOrm)) {
                settings.debugDisableOrmTexture = disableOrm;
                outNeedsReset = true;
            }
            bool flipGreen = settings.debugFlipNormalGreen;
            if (ImGui::Checkbox("Flip Normal Green (Y)", &flipGreen)) {
                settings.debugFlipNormalGreen = flipGreen;
                outNeedsReset = true;
            }
            bool specularOnly = settings.debugSpecularOnly;
            if (ImGui::Checkbox("Specular Only", &specularOnly)) {
                settings.debugSpecularOnly = specularOnly;
                outNeedsReset = true;
            }
            float normalStrength = settings.debugNormalStrengthScale;
            if (ImGui::SliderFloat("Normal Strength", &normalStrength, 0.0f, 2.0f, "%.2f")) {
                settings.debugNormalStrengthScale = std::max(normalStrength, 0.0f);
                outNeedsReset = true;
            }
            float normalLodBias = settings.debugNormalLodBias;
            if (ImGui::SliderFloat("Normal LOD Bias", &normalLodBias, 0.0f, 2.0f, "%.2f")) {
                settings.debugNormalLodBias = std::max(normalLodBias, 0.0f);
                outNeedsReset = true;
            }
            float ormLodBias = settings.debugOrmLodBias;
            if (ImGui::SliderFloat("ORM LOD Bias", &ormLodBias, 0.0f, 4.0f, "%.2f")) {
                settings.debugOrmLodBias = std::max(ormLodBias, 0.0f);
                outNeedsReset = true;
            }
            float envMipOverride = settings.debugEnvMipOverride;
            if (ImGui::SliderFloat("Force Env Mip", &envMipOverride, -1.0f, 16.0f, "%.2f")) {
                settings.debugEnvMipOverride = envMipOverride;
                outNeedsReset = true;
            }
            bool visorOverride = settings.debugEnableVisorOverride;
            if (ImGui::Checkbox("Enable Visor Debug Override", &visorOverride)) {
                settings.debugEnableVisorOverride = visorOverride;
                outNeedsReset = true;
            }
            uint32_t debugMaterialCount = 0u;
            if (m_hasScenePanelProvider && m_scenePanelProvider.materialCount) {
                debugMaterialCount = m_scenePanelProvider.materialCount();
            }
            int maxMaterialId = std::max(static_cast<int>(debugMaterialCount) - 1, -1);
            int visorMaterialId = settings.debugVisorOverrideMaterialId;
            if (ImGui::SliderInt("Visor Override Material ID", &visorMaterialId, -1, maxMaterialId, "%d")) {
                settings.debugVisorOverrideMaterialId = std::clamp(visorMaterialId, -1, maxMaterialId);
                outNeedsReset = true;
            }
            ImGui::TextDisabled("-1 = auto visor region mask, >=0 = exact material index");
            float visorRoughness = settings.debugVisorOverrideRoughness;
            if (ImGui::SliderFloat("Visor Override Roughness", &visorRoughness, 0.0f, 1.0f, "%.2f")) {
                settings.debugVisorOverrideRoughness = std::clamp(visorRoughness, 0.0f, 1.0f);
                outNeedsReset = true;
            }
            float visorF0 = settings.debugVisorOverrideF0;
            if (ImGui::SliderFloat("Visor Override F0", &visorF0, 0.0f, 0.12f, "%.3f")) {
                settings.debugVisorOverrideF0 = std::clamp(visorF0, 0.0f, 0.12f);
                outNeedsReset = true;
            }
            bool envNearest = settings.debugEnvNearest;
            if (ImGui::Checkbox("Env Nearest Filtering", &envNearest)) {
                settings.debugEnvNearest = envNearest;
                outNeedsReset = true;
            }
            ImGui::Text("Env Mip Count: %u", stats.envRadianceMipCount);
            if (ImGui::Button("Visor Debug Preset")) {
                settings.debugSpecularOnly = true;
                settings.debugDisableNormalMap = false;
                settings.debugDisableOrmTexture = false;
                settings.debugNormalStrengthScale = 0.0f;
                settings.debugEnableVisorOverride = true;
                settings.debugVisorOverrideMaterialId = -1;
                settings.debugVisorOverrideRoughness = 0.15f;
                settings.debugVisorOverrideF0 = 0.04f;
                settings.debugEnvNearest = false;
                settings.debugEnvMipOverride = 6.0f;
                outNeedsReset = true;
            }
        }
#endif

        ImGui::SetNextItemOpen(false, ImGuiCond_Once);
#if PT_DEBUG_TOOLS
        if (ImGui::CollapsingHeader("Debug Tools")) {
            bool pathDebug = settings.enablePathDebug;
            if (ImGui::Checkbox("Enable Path Debug", &pathDebug)) {
                settings.enablePathDebug = pathDebug;
                if (pathDebug && settings.debugMaxEntries == 0) {
                    settings.debugMaxEntries = 64;
                }
                outNeedsReset = true;
            }
            ImGui::BeginDisabled(!settings.enablePathDebug);
            int pixelX = static_cast<int>(settings.debugPixelX);
            int pixelY = static_cast<int>(settings.debugPixelY);
            if (ImGui::InputInt("Pixel X", &pixelX)) {
                settings.debugPixelX = static_cast<uint32_t>(std::max(pixelX, 0));
                outNeedsReset = true;
            }
            if (ImGui::InputInt("Pixel Y", &pixelY)) {
                settings.debugPixelY = static_cast<uint32_t>(std::max(pixelY, 0));
                outNeedsReset = true;
            }
            int maxEntries = static_cast<int>(settings.debugMaxEntries > 0
                                                  ? settings.debugMaxEntries
                                                  : 64u);
            const int maxAllowed =
                static_cast<int>(PathTracerShaderTypes::kPathtraceDebugMaxEntries);
            if (ImGui::SliderInt("Max Events", &maxEntries, 1, maxAllowed)) {
                settings.debugMaxEntries =
                    static_cast<uint32_t>(std::clamp(maxEntries, 1, maxAllowed));
                outNeedsReset = true;
            }
            ImGui::Text("Captured events: %u / %u",
                        stats.debugPathEntryCount,
                        stats.debugPathMaxEntries);
            ImGui::TextDisabled("Reset accumulation before capture; coordinates clamp to the current internal resolution.");
            ImGui::EndDisabled();
            ImGui::Separator();
            ImGui::Text("Parity Assert");
            bool parityEnabled = settings.parityAssertEnabled;
            if (ImGui::Checkbox("Enable Parity Assert", &parityEnabled)) {
                settings.parityAssertEnabled = parityEnabled;
                if (parityEnabled &&
                    settings.parityAssertMode == RenderSettings::ParityAssertMode::Off) {
                    settings.parityAssertMode = RenderSettings::ParityAssertMode::ProbePixelOnly;
                }
                outNeedsReset = true;
            }
            ImGui::BeginDisabled(!settings.parityAssertEnabled);
            int parityMode = static_cast<int>(settings.parityAssertMode);
            const char* parityModes[] = {"Off", "Probe Pixel", "First In-Medium"};
            if (ImGui::Combo("Mode", &parityMode, parityModes, IM_ARRAYSIZE(parityModes))) {
                parityMode = std::clamp(parityMode, 0, 2);
                settings.parityAssertMode = static_cast<RenderSettings::ParityAssertMode>(parityMode);
                outNeedsReset = true;
            }
            int parityX = static_cast<int>(settings.parityPixelX);
            int parityY = static_cast<int>(settings.parityPixelY);
            if (ImGui::InputInt("Parity Pixel X", &parityX)) {
                settings.parityPixelX = static_cast<uint32_t>(std::max(parityX, 0));
                outNeedsReset = true;
            }
            if (ImGui::InputInt("Parity Pixel Y", &parityY)) {
                settings.parityPixelY = static_cast<uint32_t>(std::max(parityY, 0));
                outNeedsReset = true;
            }
            bool parityOnce = settings.parityAssertOncePerFrame;
            if (ImGui::Checkbox("Once Per Frame", &parityOnce)) {
                settings.parityAssertOncePerFrame = parityOnce;
                outNeedsReset = true;
            }
            ImGui::Text("Divergences: %u / %u",
                        stats.parityEntryCount,
                        stats.parityMaxEntries);
            ImGui::Text("Parity checks: %u (in-medium: %u)",
                        stats.parityChecksPerformed,
                        stats.parityChecksInMedium);
            if (stats.parityEntryCount > 0) {
                ImGui::Text("Last: reason=0x%02x pixel=(%u,%u) depth=%u",
                            stats.parityLastReasonMask,
                            stats.parityLastPixelX,
                            stats.parityLastPixelY,
                            stats.parityLastDepth);
                ImGui::Text("HW: hit=%u t=%.4f mat=%u mesh=%u prim=%u front=%u",
                            stats.parityLastHwHit,
                            stats.parityLastHwT,
                            stats.parityLastHwMaterialIndex,
                            stats.parityLastHwMeshIndex,
                            stats.parityLastHwPrimitiveIndex,
                            stats.parityLastHwFrontFace);
                ImGui::Text("SW: hit=%u t=%.4f mat=%u mesh=%u prim=%u front=%u",
                            stats.parityLastSwHit,
                            stats.parityLastSwT,
                            stats.parityLastSwMaterialIndex,
                            stats.parityLastSwMeshIndex,
                            stats.parityLastSwPrimitiveIndex,
                            stats.parityLastSwFrontFace);
            } else {
                ImGui::TextDisabled("No parity divergences captured.");
            }
            ImGui::EndDisabled();
            ImGui::Separator();
            ImGui::Text("HWRT Debug");
            if (stats.hardwareRaytracingAvailable) {
                bool forcePure = settings.forcePureHWRTForGlass;
                if (ImGui::Checkbox("Force Pure HWRT (Glass)", &forcePure)) {
                    settings.forcePureHWRTForGlass = forcePure;
                    outNeedsReset = true;
                }
                int retries = static_cast<int>(settings.hardwareExcludeRetries);
                if (ImGui::SliderInt("HWRT Exclusion Retries", &retries, 0, 3)) {
                    settings.hardwareExcludeRetries = static_cast<uint32_t>(std::max(retries, 0));
                    outNeedsReset = true;
                }
                float normalBiasCm = settings.hardwareExitNormalBias * 100.0f;
                if (ImGui::SliderFloat("HWRT Exit Bias Normal (cm)", &normalBiasCm, 0.0f, 2.0f, "%.2f")) {
                    settings.hardwareExitNormalBias = std::max(normalBiasCm, 0.0f) * 0.01f;
                    outNeedsReset = true;
                }
                float directionalBiasCm = settings.hardwareExitDirectionalBias * 100.0f;
                if (ImGui::SliderFloat("HWRT Exit Bias Direction (cm)", &directionalBiasCm, 0.0f, 2.0f, "%.2f")) {
                    settings.hardwareExitDirectionalBias = std::max(directionalBiasCm, 0.0f) * 0.01f;
                    outNeedsReset = true;
                }
                bool missFallback = settings.enableHardwareMissFallback;
                if (ImGui::Checkbox("HWRT Fallback on Miss (SWRT)", &missFallback)) {
                    settings.enableHardwareMissFallback = missFallback;
                    outNeedsReset = true;
                }
                bool firstHitFallback = settings.enableHardwareFirstHitFromSoftware;
                if (ImGui::Checkbox("HWRT First Hit from SWRT", &firstHitFallback)) {
                    settings.enableHardwareFirstHitFromSoftware = firstHitFallback;
                    outNeedsReset = true;
                }
                bool forceSoftware = settings.enableHardwareForceSoftware;
                if (ImGui::Checkbox("HWRT Force SWRT (All Hits)", &forceSoftware)) {
                    settings.enableHardwareForceSoftware = forceSoftware;
                    outNeedsReset = true;
                }
            } else {
                ImGui::TextDisabled("HWRT debug controls require hardware ray tracing.");
            }
        }
#endif

        ImGui::SetNextItemOpen(false, ImGuiCond_Once);
        if (ImGui::CollapsingHeader("Fireflies Clamping")) {
            bool clampEnabled = settings.fireflyClampEnabled;
            if (ImGui::Checkbox("Enable##firefly", &clampEnabled)) {
                settings.fireflyClampEnabled = clampEnabled;
                outNeedsReset = true;
            }

            ImGui::BeginDisabled(!settings.fireflyClampEnabled);

            float clampFactor = settings.fireflyClampFactor;
            if (ImGui::SliderFloat("Clamp Factor", &clampFactor, 1.0f, 128.0f, "%.1f", ImGuiSliderFlags_Logarithmic)) {
                settings.fireflyClampFactor = std::clamp(clampFactor, 1.0f, 256.0f);
                outNeedsReset = true;
            }

            float clampFloor = settings.fireflyClampFloor;
            if (ImGui::SliderFloat("Clamp Floor", &clampFloor, 0.0f, 32.0f, "%.2f")) {
                settings.fireflyClampFloor = std::max(clampFloor, 0.0f);
                outNeedsReset = true;
            }

            float throughputClamp = settings.throughputClamp;
            if (ImGui::SliderFloat("Throughput Clamp", &throughputClamp, 1.0f, 256.0f, "%.1f", ImGuiSliderFlags_Logarithmic)) {
                settings.throughputClamp = std::clamp(throughputClamp, 1.0f, 512.0f);
                outNeedsReset = true;
            }

            float maxContribution = settings.fireflyClampMaxContribution;
            if (ImGui::SliderFloat("Max Contribution", &maxContribution, 0.0f, 1.0e6f, "%.1f", ImGuiSliderFlags_Logarithmic)) {
                settings.fireflyClampMaxContribution = std::max(maxContribution, 0.0f);
                outNeedsReset = true;
            }

            float specBase = settings.specularTailClampBase;
            if (ImGui::SliderFloat("Spec Tail Base", &specBase, 0.0f, 32.0f, "%.2f")) {
                settings.specularTailClampBase = std::max(specBase, 0.0f);
                outNeedsReset = true;
            }

            float specScale = settings.specularTailClampRoughnessScale;
            if (ImGui::SliderFloat("Spec Tail Roughness Scale", &specScale, 0.0f, 64.0f, "%.2f")) {
                settings.specularTailClampRoughnessScale = std::max(specScale, 0.0f);
                outNeedsReset = true;
            }

            float minSpecPdf = settings.minSpecularPdf;
            if (ImGui::SliderFloat("Min Specular PDF", &minSpecPdf, 0.0f, 1.0e-2f, "%.1e", ImGuiSliderFlags_Logarithmic)) {
                settings.minSpecularPdf = std::clamp(minSpecPdf, 0.0f, 1.0e-2f);
                outNeedsReset = true;
            }

            ImGui::TextDisabled("Higher clamp values reduce bias but may allow more fireflies.");

            ImGui::EndDisabled();
        }

        ImGui::SetNextItemOpen(false, ImGuiCond_Once);
        if (ImGui::CollapsingHeader("Tonemapping")) {
            const char* operators[] = {"Linear", "ACES", "Reinhard", "Hable"};
            int currentOperator = static_cast<int>(settings.tonemapMode) - 1;
            currentOperator = std::clamp(currentOperator, 0, 3);
            if (ImGui::Combo("Operator", &currentOperator, operators, IM_ARRAYSIZE(operators))) {
                settings.tonemapMode = static_cast<uint32_t>(currentOperator + 1);
            }

            if (settings.tonemapMode == 2) {
                int variant = static_cast<int>(settings.acesVariant);
                if (ImGui::RadioButton("Fitted", variant == 0)) {
                    settings.acesVariant = 0;
                }
                ImGui::SameLine();
                if (ImGui::RadioButton("Simple", variant == 1)) {
                    settings.acesVariant = 1;
                }
            } else if (settings.tonemapMode == 3) {
                if (ImGui::SliderFloat("White Point", &settings.reinhardWhitePoint, 0.1f, 5.0f, "%.2f")) {
                    settings.reinhardWhitePoint = std::clamp(settings.reinhardWhitePoint, 0.1f, 10.0f);
                }
            }

            bool bloomEnabled = settings.bloomEnabled;
            if (ImGui::Checkbox("Enable Bloom", &bloomEnabled)) {
                settings.bloomEnabled = bloomEnabled;
            }
            ImGui::BeginDisabled(!settings.bloomEnabled);
            if (ImGui::SliderFloat("Bloom Threshold", &settings.bloomThreshold, 0.1f, 8.0f, "%.2f")) {
                settings.bloomThreshold = std::max(settings.bloomThreshold, 0.0f);
            }
            if (ImGui::SliderFloat("Bloom Intensity", &settings.bloomIntensity, 0.0f, 2.0f, "%.2f")) {
                settings.bloomIntensity = std::max(settings.bloomIntensity, 0.0f);
            }
            if (ImGui::SliderFloat("Bloom Radius", &settings.bloomRadius, 0.0f, 8.0f, "%.2f")) {
                settings.bloomRadius = std::max(settings.bloomRadius, 0.0f);
            }
            ImGui::EndDisabled();
        }

        ImGui::SetNextItemOpen(false, ImGuiCond_Once);
        if (ImGui::CollapsingHeader("Exposure")) {
            ImGui::SliderFloat("Exposure (stops)", &settings.exposure, -5.0f, 5.0f, "%.2f");
        }

        ImGui::SetNextItemOpen(false, ImGuiCond_Once);
        if (ImGui::CollapsingHeader("Denoising (OIDN)")) {
            bool denoiseEnabled = settings.denoiseEnabled;
            if (ImGui::Checkbox("Enable Denoising", &denoiseEnabled)) {
                settings.denoiseEnabled = denoiseEnabled;
            }

            if (settings.denoiseEnabled) {
                const char* filterTypes[] = {"RT (Ray Tracing)", "RTLightmap"};
                int currentFilter = static_cast<int>(settings.denoiseFilterType);
                currentFilter = std::clamp(currentFilter, 0, 1);
                if (ImGui::Combo("Filter Type", &currentFilter, filterTypes, IM_ARRAYSIZE(filterTypes))) {
                    settings.denoiseFilterType = static_cast<uint32_t>(currentFilter);
                }

                // Denoising frequency slider
                int frequency = static_cast<int>(settings.denoiseFrequency);
                if (ImGui::SliderInt("Denoise Frequency", &frequency, 1, 60, "%d frames")) {
                    settings.denoiseFrequency = std::max(1u, static_cast<uint32_t>(frequency));
                }
                ImGui::SameLine();
                ImGui::TextDisabled("(every Nth frame)");

                bool useAlbedo = settings.denoiseUseAlbedo;
                if (ImGui::Checkbox("Use Albedo AOV##denoise", &useAlbedo)) {
                    settings.denoiseUseAlbedo = useAlbedo;
                }

                bool useNormal = settings.denoiseUseNormal;
                if (ImGui::Checkbox("Use Normal AOV##denoise", &useNormal)) {
                    settings.denoiseUseNormal = useNormal;
                }

                ImGui::TextWrapped("OIDN denoising provides 2-4x faster visual convergence at low sample counts. "
                                   "Albedo and normal auxiliary buffers improve edge preservation and detail retention. "
                                   "Progressive denoising (frequency > 1) reduces CPU overhead by denoising less frequently.");
            } else {
                ImGui::TextDisabled("Denoising is currently disabled.");
            }
        }

        ImGui::SetNextItemOpen(false, ImGuiCond_Once);
        if (ImGui::CollapsingHeader("Camera")) {
            float yawDegrees = WrapDegrees(settings.cameraYaw * kRadToDeg);
            if (ImGui::SliderFloat("Yaw (deg)", &yawDegrees, -180.0f, 180.0f, "%.1f")) {
                settings.cameraYaw = yawDegrees * kDegToRad;
            }

            float pitchDegrees = std::clamp(settings.cameraPitch * kRadToDeg, -89.9f, 89.9f);
            if (ImGui::SliderFloat("Pitch (deg)", &pitchDegrees, -89.9f, 89.9f, "%.1f")) {
                pitchDegrees = std::clamp(pitchDegrees, -89.9f, 89.9f);
                settings.cameraPitch = pitchDegrees * kDegToRad;
            }

            float vfov = settings.cameraVerticalFov;
            if (ImGui::SliderFloat("Vertical FOV", &vfov, 5.0f, 120.0f, "%.1f")) {
                settings.cameraVerticalFov = std::clamp(vfov, 1.0f, 179.0f);
                outNeedsReset = true;
            }

            float aperture = settings.cameraDefocusAngle;
            if (ImGui::SliderFloat("Aperture (deg)", &aperture, 0.0f, 10.0f, "%.3f")) {
                settings.cameraDefocusAngle = std::max(aperture, 0.0f);
                outNeedsReset = true;
            }

            float focus = settings.cameraFocusDistance;
            if (ImGui::SliderFloat("Focus Distance", &focus, 0.0f, 2000.0f, "%.1f")) {
                settings.cameraFocusDistance = std::max(focus, 0.0f);
                outNeedsReset = true;
            }
            ImGui::TextWrapped("Set focus distance to 0 to reuse the orbit camera radius.");
        }

        if (!sceneNames.empty()) {
            ImGui::SetNextItemOpen(true, ImGuiCond_Once);
            if (ImGui::CollapsingHeader("Scene", ImGuiTreeNodeFlags_DefaultOpen)) {
                int sceneIndex = std::clamp(inOutSelectedScene, 0, static_cast<int>(sceneNames.size()) - 1);
                if (ImGui::Combo("Active Scene", &sceneIndex, sceneNames.data(), static_cast<int>(sceneNames.size()))) {
                    inOutSelectedScene = sceneIndex;
                }
            }
        }

        ImGui::SetNextItemOpen(false, ImGuiCond_Once);
        if (ImGui::CollapsingHeader("Objects", ImGuiTreeNodeFlags_DefaultOpen)) {
            const bool hasMaterialFuncs =
                m_hasScenePanelProvider && m_scenePanelProvider.materialCount && m_scenePanelProvider.readMaterial;
            const bool hasObjectFuncs =
                m_hasScenePanelProvider && m_scenePanelProvider.objectCount && m_scenePanelProvider.readObject;
            if (!hasMaterialFuncs && !hasObjectFuncs) {
                ImGui::TextDisabled("Scene inspector unavailable.");
            } else {
                const uint32_t materialCount =
                    hasMaterialFuncs ? m_scenePanelProvider.materialCount() : 0u;
                const uint32_t objectCount =
                    hasObjectFuncs ? m_scenePanelProvider.objectCount() : 0u;

                m_materialStates.resize(materialCount);
                m_objectStates.resize(objectCount);

                if (hasMaterialFuncs) {
                    for (uint32_t i = 0; i < materialCount; ++i) {
                        MaterialDisplayInfo info;
                        if (!m_scenePanelProvider.readMaterial(i, info)) {
                            continue;
                        }
                        MaterialEditorState& state = m_materialStates[i];
                        const bool needsRefresh =
                            !state.initialized ||
                            state.name != info.name ||
                            std::memcmp(&state.sourceData, &info.data, sizeof(info.data)) != 0;
                        if (needsRefresh) {
                            state.initialized = true;
                            state.name = info.name;
                            state.sourceData = info.data;
                            state.workingData = info.data;
                        }
                    }
                } else {
                    m_materialStates.clear();
                }

                if (hasObjectFuncs) {
                    for (uint32_t i = 0; i < objectCount; ++i) {
                        ObjectDisplayInfo info;
                        if (!m_scenePanelProvider.readObject(i, info)) {
                            continue;
                        }
                        ObjectEditorState& state = m_objectStates[i];
                        const bool needsRefresh =
                            !state.initialized ||
                            state.meshIndex != info.meshIndex ||
                            state.name != info.name ||
                            std::memcmp(&state.sourceTransform, &info.transform, sizeof(info.transform)) != 0 ||
                            state.materialIndex != info.materialIndex;
                        if (needsRefresh) {
                            state.initialized = true;
                            state.name = info.name;
                            state.meshIndex = info.meshIndex;
                            state.materialIndex = info.materialIndex;
                            state.sourceTransform = info.transform;
                            state.workingTransform = info.transform;
                        }
                    }
                } else {
                    m_objectStates.clear();
                    m_gizmoWasActive = false;
                    m_gizmoActiveObject = -1;
                }

                if (materialCount == 0) {
                    m_selectedMaterial = -1;
                } else {
                    if (m_selectedMaterial < 0) {
                        m_selectedMaterial = 0;
                    }
                    if (m_selectedMaterial >= static_cast<int>(materialCount)) {
                        m_selectedMaterial = static_cast<int>(materialCount) - 1;
                    }
                }

                if (objectCount == 0) {
                    m_selectedObject = -1;
                    m_gizmoWasActive = false;
                    m_gizmoActiveObject = -1;
                } else {
                    if (m_selectedObject < 0) {
                        m_selectedObject = 0;
                    }
                    if (m_selectedObject >= static_cast<int>(objectCount)) {
                        m_selectedObject = static_cast<int>(objectCount) - 1;
                    }
                }

                MaterialEditorState* activeMaterial =
                    (m_selectedMaterial >= 0 && m_selectedMaterial < static_cast<int>(m_materialStates.size()))
                        ? &m_materialStates[static_cast<size_t>(m_selectedMaterial)]
                        : nullptr;
                ObjectEditorState* activeObject =
                    (m_selectedObject >= 0 && m_selectedObject < static_cast<int>(m_objectStates.size()))
                        ? &m_objectStates[static_cast<size_t>(m_selectedObject)]
                        : nullptr;

                auto refreshMaterialState = [&](uint32_t idx) {
                    if (!hasMaterialFuncs || !m_scenePanelProvider.readMaterial) {
                        return;
                    }
                    if (idx >= m_materialStates.size()) {
                        return;
                    }
                    MaterialDisplayInfo refreshed;
                    if (!m_scenePanelProvider.readMaterial(idx, refreshed)) {
                        return;
                    }
                    MaterialEditorState& state = m_materialStates[idx];
                    state.initialized = true;
                    state.name = refreshed.name;
                    state.sourceData = refreshed.data;
                    state.workingData = refreshed.data;
                };

                auto refreshObjectState = [&](uint32_t idx) {
                    if (!hasObjectFuncs || !m_scenePanelProvider.readObject) {
                        return;
                    }
                    if (idx >= m_objectStates.size()) {
                        return;
                    }
                    ObjectDisplayInfo refreshed;
                    if (!m_scenePanelProvider.readObject(idx, refreshed)) {
                        return;
                    }
                    ObjectEditorState& state = m_objectStates[idx];
                    state.initialized = true;
                    state.name = refreshed.name;
                    state.meshIndex = refreshed.meshIndex;
                    state.materialIndex = refreshed.materialIndex;
                    state.sourceTransform = refreshed.transform;
                    state.workingTransform = refreshed.transform;
                };

                ImGui::Separator();
                ImGui::TextUnformatted("Materials");
                if (materialCount == 0 || !hasMaterialFuncs) {
                    ImGui::TextDisabled("No editable materials.");
                } else {
                    const char* currentMaterialLabel =
                        activeMaterial ? activeMaterial->name.c_str() : "<unknown>";
                    if (ImGui::BeginCombo("Material", currentMaterialLabel)) {
                        for (uint32_t i = 0; i < materialCount; ++i) {
                            bool isSelected = (m_selectedMaterial == static_cast<int>(i));
                            const char* label = m_materialStates[i].name.empty()
                                                    ? "<unnamed>"
                                                    : m_materialStates[i].name.c_str();
                            if (ImGui::Selectable(label, isSelected)) {
                                m_selectedMaterial = static_cast<int>(i);
                                activeMaterial = &m_materialStates[i];
                            }
                            if (isSelected) {
                                ImGui::SetItemDefaultFocus();
                            }
                        }
                        ImGui::EndCombo();
                    }

                    if (!activeMaterial || !activeMaterial->initialized) {
                        ImGui::TextDisabled("Material data unavailable.");
                    } else {
                        auto& data = activeMaterial->workingData;
                        simd::float4 baseColorRoughness = data.baseColorRoughness;
                        float color[3] = {baseColorRoughness.x, baseColorRoughness.y, baseColorRoughness.z};
                        bool materialLiveChanged = false;
                        if (ImGui::ColorEdit3("Albedo",
                                              color,
                                              ImGuiColorEditFlags_HDR |
                                              ImGuiColorEditFlags_DisplayRGB |
                                              ImGuiColorEditFlags_InputRGB)) {
                            baseColorRoughness.x = std::clamp(color[0], 0.0f, 1.0f);
                            baseColorRoughness.y = std::clamp(color[1], 0.0f, 1.0f);
                            baseColorRoughness.z = std::clamp(color[2], 0.0f, 1.0f);
                            data.baseColorRoughness = baseColorRoughness;
                            materialLiveChanged = true;
                            const bool hadConductor =
                                (data.conductorEta.w > 0.0f) || (data.conductorK.w > 0.0f) ||
                                (data.conductorEta.x > 0.0f) || (data.conductorEta.y > 0.0f) ||
                                (data.conductorEta.z > 0.0f);
                            if (hadConductor) {
                                data.conductorEta = simd_make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                                data.conductorK = simd_make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                            }
                        }
                        float roughness = baseColorRoughness.w;
                        if (ImGui::SliderFloat("Roughness", &roughness, 0.02f, 1.0f, "%.3f")) {
                            data.baseColorRoughness.w = std::clamp(roughness, 0.02f, 1.0f);
                            materialLiveChanged = true;
                        }
                        float ior = data.typeEta.y;
                        if (ImGui::SliderFloat("Indices of Refraction", &ior, 1.01f, 2.5f, "%.3f")) {
                            data.typeEta.y = std::clamp(ior, 1.01f, 2.5f);
                            materialLiveChanged = true;
                        }

                        const MaterialType matType =
                            static_cast<MaterialType>(static_cast<int>(std::round(data.typeEta.x)));

                        bool metallicDisabled = (matType != MaterialType::CarPaint);
                        if (metallicDisabled) {
                            ImGui::BeginDisabled();
                        }
                        float metallic = data.carpaintBaseParams.x;
                        if (ImGui::SliderFloat("Metallic", &metallic, 0.0f, 1.0f, "%.3f")) {
                            data.carpaintBaseParams.x = std::clamp(metallic, 0.0f, 1.0f);
                            materialLiveChanged = true;
                        }
                        if (metallicDisabled) {
                            ImGui::EndDisabled();
                            ImGui::TextDisabled("Metallic editing is available for car paint materials.");
                        }

                        const bool supportsCoat =
                            matType == MaterialType::CarPaint ||
                            matType == MaterialType::Plastic ||
                            matType == MaterialType::Subsurface;
                        if (!supportsCoat) {
                            ImGui::BeginDisabled();
                        }
                        float coatRoughness = data.coatParams.x;
                        if (ImGui::SliderFloat("Coat Roughness", &coatRoughness, 0.0f, 1.0f, "%.3f")) {
                            data.coatParams.x = std::clamp(coatRoughness, 0.0f, 1.0f);
                            materialLiveChanged = true;
                        }
                        float coatIor = data.typeEta.z;
                        if (ImGui::SliderFloat("Coat IOR", &coatIor, 1.01f, 2.5f, "%.3f")) {
                            data.typeEta.z = std::clamp(coatIor, 1.01f, 2.5f);
                            materialLiveChanged = true;
                        }
                        float coatTint[3] = {data.coatTint.x, data.coatTint.y, data.coatTint.z};
                        if (ImGui::ColorEdit3("Coat Tint", coatTint)) {
                            data.coatTint = simd_make_float4(std::clamp(coatTint[0], 0.0f, 1.0f),
                                                             std::clamp(coatTint[1], 0.0f, 1.0f),
                                                             std::clamp(coatTint[2], 0.0f, 1.0f),
                                                             0.0f);
                            materialLiveChanged = true;
                        }
                        if (!supportsCoat) {
                            ImGui::EndDisabled();
                            ImGui::TextDisabled("Coat controls apply to plastic, subsurface, or car paint materials.");
                        }

                        if (matType == MaterialType::Subsurface) {
                            float meanFreePath = data.sssParams.x;
                            if (ImGui::SliderFloat("Mean Free Path", &meanFreePath, 0.01f, 10.0f, "%.3f")) {
                                data.sssParams.x = std::max(meanFreePath, 0.01f);
                                materialLiveChanged = true;
                            }
                        }

                        if (materialLiveChanged && m_scenePanelProvider.applyMaterial) {
                            MaterialDisplayInfo submission;
                            submission.name = activeMaterial->name;
                            submission.data = data;
                            m_scenePanelProvider.applyMaterial(static_cast<uint32_t>(m_selectedMaterial),
                                                               submission);
                            refreshMaterialState(static_cast<uint32_t>(m_selectedMaterial));
                        }

                        if (ImGui::Button("Apply##material")) {
                            if (m_scenePanelProvider.applyMaterial) {
                                MaterialDisplayInfo submission;
                                submission.name = activeMaterial->name;
                                submission.data = data;
                                m_scenePanelProvider.applyMaterial(static_cast<uint32_t>(m_selectedMaterial),
                                                                   submission);
                                refreshMaterialState(static_cast<uint32_t>(m_selectedMaterial));
                            }
                        }
                        ImGui::SameLine();
                        if (ImGui::Button("Reset##material")) {
                            if (m_scenePanelProvider.resetMaterial) {
                                m_scenePanelProvider.resetMaterial(static_cast<uint32_t>(m_selectedMaterial));
                                refreshMaterialState(static_cast<uint32_t>(m_selectedMaterial));
                            }
                        }
                    }
                }

                ImGui::Separator();
                ImGui::TextUnformatted("Transforms");
                if (objectCount == 0 || !hasObjectFuncs) {
                    ImGui::TextDisabled("No mesh instances available for editing.");
                } else {
                    const char* currentObjectLabel =
                        activeObject ? activeObject->name.c_str() : "<unknown>";
                    if (ImGui::BeginCombo("Object", currentObjectLabel)) {
                        for (uint32_t i = 0; i < objectCount; ++i) {
                            bool isSelected = (m_selectedObject == static_cast<int>(i));
                            const char* label = m_objectStates[i].name.empty()
                                                    ? "<unnamed>"
                                                    : m_objectStates[i].name.c_str();
                            if (ImGui::Selectable(label, isSelected)) {
                                m_selectedObject = static_cast<int>(i);
                                activeObject = &m_objectStates[i];
                            }
                            if (isSelected) {
                                ImGui::SetItemDefaultFocus();
                            }
                        }
                        ImGui::EndCombo();
                    }

                    if (!activeObject || !activeObject->initialized) {
                        ImGui::TextDisabled("Object data unavailable.");
                    } else {
                        ImGui::Text("Material Index: %u", activeObject->materialIndex);
                        if (activeObject->materialIndex < m_materialStates.size()) {
                            ImGui::SameLine();
                            if (ImGui::SmallButton("Edit Material##jump")) {
                                m_selectedMaterial = static_cast<int>(activeObject->materialIndex);
                            }
                        }

                        float matrix[16];
                        std::memcpy(matrix, &activeObject->workingTransform, sizeof(matrix));
                        float translation[3], rotation[3], scale[3];
                        ImGuizmo::DecomposeMatrixToComponents(matrix, translation, rotation, scale);

                        bool transformLiveChanged = false;
                        transformLiveChanged |= ImGui::InputFloat3("Tr", translation, "%.3f");
                        transformLiveChanged |= ImGui::InputFloat3("Rt", rotation, "%.3f");
                        transformLiveChanged |= ImGui::InputFloat3("Sc", scale, "%.3f");
                        if (transformLiveChanged) {
                            ImGuizmo::RecomposeMatrixFromComponents(translation, rotation, scale, matrix);
                            std::memcpy(&activeObject->workingTransform, matrix, sizeof(matrix));
                        }

                        if (ImGui::RadioButton("Translate", m_gizmoOperation == 0)) {
                            m_gizmoOperation = 0;
                        }
                        ImGui::SameLine();
                        if (ImGui::RadioButton("Rotate", m_gizmoOperation == 1)) {
                            m_gizmoOperation = 1;
                        }
                        ImGui::SameLine();
                        if (ImGui::RadioButton("Scale", m_gizmoOperation == 2)) {
                            m_gizmoOperation = 2;
                        }

                        if (ImGui::RadioButton("Local", m_gizmoMode == 0)) {
                            m_gizmoMode = 0;
                        }
                        ImGui::SameLine();
                        if (ImGui::RadioButton("World", m_gizmoMode == 1)) {
                            m_gizmoMode = 1;
                        }

                        if (ImGui::IsKeyPressed(ImGuiKey_Z)) {
                            m_gizmoOperation = 0;
                        } else if (ImGui::IsKeyPressed(ImGuiKey_E)) {
                            m_gizmoOperation = 1;
                        } else if (ImGui::IsKeyPressed(ImGuiKey_R)) {
                            m_gizmoOperation = 2;
                        }

                        if (ImGui::IsKeyPressed(ImGuiKey_S)) {
                            m_gizmoUseSnap = !m_gizmoUseSnap;
                        }

                        ImGui::Checkbox("Enable Snap", &m_gizmoUseSnap);
                        if (m_gizmoOperation == 0) {
                            ImGui::InputFloat3("Snap##translate", m_gizmoSnapTranslation, "%.3f");
                        } else if (m_gizmoOperation == 1) {
                            ImGui::InputFloat("Angle Snap", &m_gizmoSnapAngle, 0.1f, 1.0f, "%.3f");
                        } else {
                            ImGui::InputFloat("Scale Snap", &m_gizmoSnapScale, 0.01f, 0.1f, "%.3f");
                        }

                        if (ImGui::Button("Apply##object")) {
                            if (m_scenePanelProvider.applyObjectTransform) {
                                ObjectDisplayInfo submission;
                                submission.name = activeObject->name;
                                submission.meshIndex = activeObject->meshIndex;
                                submission.materialIndex = activeObject->materialIndex;
                                submission.transform = activeObject->workingTransform;
                                m_scenePanelProvider.applyObjectTransform(activeObject->meshIndex, submission);
                                refreshObjectState(activeObject->meshIndex);
                            }
                        }
                        ImGui::SameLine();
                        if (ImGui::Button("Reset##object")) {
                            if (m_scenePanelProvider.resetObjectTransform) {
                                m_scenePanelProvider.resetObjectTransform(activeObject->meshIndex);
                                refreshObjectState(activeObject->meshIndex);
                            }
                        }

                        if (m_scenePanelProvider.applyObjectTransform) {
                            ImGuizmo::BeginFrame();
                            ImGuiIO& io = ImGui::GetIO();
                            ImGuizmo::SetOrthographic(false);
                            ImGuizmo::SetDrawlist(ImGui::GetForegroundDrawList());
                            const ImGuiViewport* viewport = ImGui::GetMainViewport();
                            ImGuizmo::SetRect(viewport->Pos.x,
                                              viewport->Pos.y,
                                              viewport->Size.x,
                                              viewport->Size.y);
                            ImGuizmo::PushID(static_cast<int>(activeObject->meshIndex));

                            CameraMatrices cameraMatrices = BuildCameraMatrices(settings, m_displaySizePoints);
                            float viewMatrix[16];
                            float projectionMatrix[16];
                            std::memcpy(viewMatrix, &cameraMatrices.view, sizeof(viewMatrix));
                            std::memcpy(projectionMatrix, &cameraMatrices.projection, sizeof(projectionMatrix));

                            std::memcpy(matrix, &activeObject->workingTransform, sizeof(matrix));
                            ImGuizmo::OPERATION op =
                                (m_gizmoOperation == 0) ? ImGuizmo::TRANSLATE :
                                (m_gizmoOperation == 1) ? ImGuizmo::ROTATE : ImGuizmo::SCALE;
                            ImGuizmo::MODE mode = (m_gizmoMode == 0) ? ImGuizmo::LOCAL : ImGuizmo::WORLD;
                            float* snapPtr = nullptr;
                            float snapBuffer[3] = {m_gizmoSnapTranslation[0],
                                                   m_gizmoSnapTranslation[1],
                                                   m_gizmoSnapTranslation[2]};
                            float angleSnap = m_gizmoSnapAngle;
                            float scaleSnap = m_gizmoSnapScale;
                            if (m_gizmoUseSnap) {
                                if (op == ImGuizmo::TRANSLATE) {
                                    snapPtr = snapBuffer;
                                } else if (op == ImGuizmo::ROTATE) {
                                    snapPtr = &angleSnap;
                                } else {
                                    snapPtr = &scaleSnap;
                                }
                            }

                            ImGuizmo::Manipulate(viewMatrix,
                                                 projectionMatrix,
                                                 op,
                                                 mode,
                                                 matrix,
                                                 nullptr,
                                                 snapPtr);
                            bool gizmoUsing = ImGuizmo::IsUsing();
                            if (gizmoUsing) {
                                std::memcpy(&activeObject->workingTransform, matrix, sizeof(matrix));
                                transformLiveChanged = true;
                                m_gizmoActiveObject = m_selectedObject;
                            } else {
                                m_gizmoActiveObject = -1;
                            }
                            m_gizmoWasActive = gizmoUsing;
                            ImGuizmo::PopID();
                        }

                        if (transformLiveChanged && m_scenePanelProvider.applyObjectTransform) {
                            ObjectDisplayInfo submission;
                            submission.name = activeObject->name;
                            submission.meshIndex = activeObject->meshIndex;
                            submission.materialIndex = activeObject->materialIndex;
                            submission.transform = activeObject->workingTransform;
                            m_scenePanelProvider.applyObjectTransform(activeObject->meshIndex, submission);
                            refreshObjectState(activeObject->meshIndex);
                        }
                    }
                }
            }
        }

        ImGui::SetNextItemOpen(false, ImGuiCond_Once);
        if (ImGui::CollapsingHeader("UI")) {
            ImGui::SetNextItemOpen(true, ImGuiCond_Once);
            if (ImGui::CollapsingHeader("Presentation")) {
                bool presentationEnabled = presentation.enabled;
                if (ImGui::Checkbox("Enable Presentation Mode", &presentationEnabled)) {
                    presentation.enabled = presentationEnabled;
                }

                int targetScreen = static_cast<int>(presentation.targetScreen);
                const char* screenOptions[] = {"Auto", "Primary", "External (if present)"};
                if (ImGui::Combo("Target Screen", &targetScreen, screenOptions, IM_ARRAYSIZE(screenOptions))) {
                    presentation.targetScreen =
                        static_cast<PresentationSettings::TargetScreen>(targetScreen);
                }

                int windowMode = static_cast<int>(presentation.windowMode);
                const char* windowOptions[] = {"Borderless Fullscreen", "Maximized Window"};
                if (ImGui::Combo("Window Mode", &windowMode, windowOptions, IM_ARRAYSIZE(windowOptions))) {
                    presentation.windowMode =
                        static_cast<PresentationSettings::WindowMode>(windowMode);
                }

                bool hidePanels = presentation.hideUIPanels;
                if (ImGui::Checkbox("Hide UI Panels", &hidePanels)) {
                    presentation.hideUIPanels = hidePanels;
                }

                bool minimalOverlay = presentation.minimalOverlay;
                if (ImGui::Checkbox("Minimal Overlay", &minimalOverlay)) {
                    presentation.minimalOverlay = minimalOverlay;
                }

                int resolutionLock = static_cast<int>(presentation.resolutionLock);
                const char* lockOptions[] = {"Off", "1280x720", "1920x1080"};
                if (ImGui::Combo("Lock Render Resolution",
                                 &resolutionLock,
                                 lockOptions,
                                 IM_ARRAYSIZE(lockOptions))) {
                    presentation.resolutionLock =
                        static_cast<PresentationSettings::RenderResolutionLock>(resolutionLock);
                }

                int contentScale = static_cast<int>(presentation.contentScale);
                const char* scaleOptions[] = {"Auto", "1.0", "2.0"};
                if (ImGui::Combo("Force Content Scale",
                                 &contentScale,
                                 scaleOptions,
                                 IM_ARRAYSIZE(scaleOptions))) {
                    presentation.contentScale =
                        static_cast<PresentationSettings::ContentScaleMode>(contentScale);
                }

                bool resetOnToggle = presentation.resetAccumulationOnToggle;
                if (ImGui::Checkbox("Reset accumulation on toggle", &resetOnToggle)) {
                    presentation.resetAccumulationOnToggle = resetOnToggle;
                }
            }
        }

        ImGui::SetNextItemOpen(false, ImGuiCond_Once);
        if (ImGui::CollapsingHeader("Dev Tools")) {
            ImGui::Text("Backing Scale: %.2fx", m_backingScale);
            ImGui::Text("Effective Scale: %.2fx", m_displayScale);
            ImGui::Text("Font Atlas Scale: %.2fx", m_lastFontScale);
            if (ImGui::Button("Rebuild Fonts##dev")) {
                requestFontRebuild(m_displayScale);
            }
            ImGui::TextDisabled("Fonts rebuild automatically when backing scale shifts by >=5%%.");
        }

        ImGui::SetNextItemOpen(false, ImGuiCond_Once);
        if (ImGui::CollapsingHeader("Debug", ImGuiTreeNodeFlags_None)) {
            ImGui::Checkbox("Show Dear ImGui Demo", &m_showDemo);
        }
        }
        ImGui::End();
    }

    if (m_minimalOverlayVisible) {
        ImGuiIO& io = ImGui::GetIO();
        const ImGuiWindowFlags flags =
            ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
            ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing |
            ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoInputs;
        ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x - 20.0f, 20.0f),
                                ImGuiCond_Always,
                                ImVec2(1.0f, 0.0f));
        ImGui::SetNextWindowBgAlpha(0.35f);
        if (ImGui::Begin("Presentation Overlay", nullptr, flags)) {
            const char* backendLabel = stats.hardwareRaytracingActive ? "HWRT" : "SWRT";
            ImGui::Text("Backend: %s", backendLabel);
            ImGui::Text("SPP: %u (spf %u)", stats.sampleCount, settings.samplesPerFrame);
            ImGui::Text("Frame: %.2f ms (%.1f fps)", stats.frameTimeMs, fps);
        }
        ImGui::End();
    }

    if (m_showDemo && m_mainPanelVisible) {
        ImGui::ShowDemoWindow(&m_showDemo);
    }
}

void UIOverlay::render(MTLCommandBufferHandle commandBuffer,
                       MTLRenderCommandEncoderHandle renderEncoder) {
    if (!m_initialized || !m_visible) {
        return;
    }
    
    ImGui::Render();
    ImDrawData* drawData = ImGui::GetDrawData();
    if (!drawData) {
        return;
    }
    
    [renderEncoder pushDebugGroup:@"ImGui Overlay"];
    ImGui_ImplMetal_RenderDrawData(drawData,
                                   (id<MTLCommandBuffer>)commandBuffer,
                                   (id<MTLRenderCommandEncoder>)renderEncoder);
    [renderEncoder popDebugGroup];
}

void UIOverlay::shutdown() {
    if (!m_initialized) {
        return;
    }
    
    ImGui_ImplMetal_Shutdown();
    ImGui_ImplOSX_Shutdown();
    ImGui::DestroyContext();
    m_view = nullptr;
    m_device = nullptr;
    m_fontScaleInitialized = false;
    m_pendingFontRebuild = false;
    m_mainPanelVisible = true;
    m_minimalOverlayVisible = false;
    m_backingScale = 1.0f;
    m_displayScale = 1.0f;
    m_hasForcedContentScale = false;
    m_forcedContentScale = 1.0f;
    m_lastFontScale = 1.0f;
    m_displaySizePoints = CGSizeMake(1.0, 1.0);
    m_initialized = false;
}

bool UIOverlay::wantsInput() const {
    if (!m_initialized || !m_visible) {
        return false;
    }
    if (!m_mainPanelVisible) {
        return false;
    }

    ImGuiIO& io = ImGui::GetIO();
    return io.WantCaptureMouse || io.WantCaptureKeyboard;
}

}  // namespace PathTracer
