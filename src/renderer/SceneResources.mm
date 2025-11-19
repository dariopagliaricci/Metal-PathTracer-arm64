#import "renderer/SceneResources.h"
#import "renderer/MetalContext.h"
#import "renderer/SceneAccel.h"
#import "renderer/EnvImportanceSampler.h"

#import <MetalKit/MetalKit.h>
#import <CoreFoundation/CoreFoundation.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <limits>
#include <random>
#include <vector>

using PathTracerShaderTypes::EnvironmentAliasEntry;
using PathTracerShaderTypes::MaterialData;
using PathTracerShaderTypes::MaterialType;
using PathTracerShaderTypes::RectData;
using PathTracerShaderTypes::SphereData;
using PathTracerShaderTypes::kMaxMaterials;
using PathTracerShaderTypes::kMaxRectangles;
using PathTracerShaderTypes::kMaxSpheres;

namespace PathTracer {

namespace {

inline float SrgbToLinear(float value) {
    if (value <= 0.04045f) {
        return value / 12.92f;
    }
    return powf((value + 0.055f) / 1.055f, 2.4f);
}

NSUInteger BytesPerPixel(MTLPixelFormat format) {
    switch (format) {
        case MTLPixelFormatRGBA32Float:
            return sizeof(float) * 4u;
        case MTLPixelFormatRGBA16Float:
            return sizeof(uint16_t) * 4u;
        case MTLPixelFormatRGBA8Unorm:
        case MTLPixelFormatRGBA8Unorm_sRGB:
        case MTLPixelFormatBGRA8Unorm:
        case MTLPixelFormatBGRA8Unorm_sRGB:
            return sizeof(uint8_t) * 4u;
        default:
            return 0u;
    }
}

bool CopyTextureToFloat(id<MTLTexture> texture, std::vector<float>& outFloats) {
    if (!texture) {
        return false;
    }

    const NSUInteger width = texture.width;
    const NSUInteger height = texture.height;
    if (width == 0 || height == 0) {
        return false;
    }

    const MTLPixelFormat format = texture.pixelFormat;
    const NSUInteger bytesPerPixel = BytesPerPixel(format);
    if (bytesPerPixel == 0) {
        NSLog(@"Environment map pixel format %lu not supported for importance sampling", static_cast<unsigned long>(format));
        return false;
    }

    const NSUInteger pixelCount = width * height;
    const NSUInteger bytesPerRow = width * bytesPerPixel;
    std::vector<uint8_t> rawData(static_cast<size_t>(bytesPerRow) * height);

    MTLRegion region = MTLRegionMake2D(0, 0, width, height);
    [texture getBytes:rawData.data()
          bytesPerRow:bytesPerRow
           fromRegion:region
          mipmapLevel:0];

    outFloats.assign(pixelCount * 4u, 0.0f);

    switch (format) {
        case MTLPixelFormatRGBA32Float: {
            const float* src = reinterpret_cast<const float*>(rawData.data());
            std::copy(src, src + outFloats.size(), outFloats.begin());
            break;
        }
        case MTLPixelFormatRGBA16Float: {
            const __fp16* src = reinterpret_cast<const __fp16*>(rawData.data());
            for (size_t i = 0; i < outFloats.size(); ++i) {
                outFloats[i] = static_cast<float>(src[i]);
            }
            break;
        }
        case MTLPixelFormatRGBA8Unorm:
        case MTLPixelFormatBGRA8Unorm: {
            const uint8_t* src = rawData.data();
            for (NSUInteger i = 0; i < pixelCount; ++i) {
                uint8_t r = src[i * 4u + 0u];
                uint8_t g = src[i * 4u + 1u];
                uint8_t b = src[i * 4u + 2u];
                uint8_t a = src[i * 4u + 3u];
                if (format == MTLPixelFormatBGRA8Unorm) {
                    std::swap(r, b);
                }
                outFloats[i * 4u + 0u] = static_cast<float>(r) / 255.0f;
                outFloats[i * 4u + 1u] = static_cast<float>(g) / 255.0f;
                outFloats[i * 4u + 2u] = static_cast<float>(b) / 255.0f;
                outFloats[i * 4u + 3u] = static_cast<float>(a) / 255.0f;
            }
            break;
        }
        case MTLPixelFormatRGBA8Unorm_sRGB:
        case MTLPixelFormatBGRA8Unorm_sRGB: {
            const uint8_t* src = rawData.data();
            for (NSUInteger i = 0; i < pixelCount; ++i) {
                uint8_t r = src[i * 4u + 0u];
                uint8_t g = src[i * 4u + 1u];
                uint8_t b = src[i * 4u + 2u];
                uint8_t a = src[i * 4u + 3u];
                if (format == MTLPixelFormatBGRA8Unorm_sRGB) {
                    std::swap(r, b);
                }
                outFloats[i * 4u + 0u] = SrgbToLinear(static_cast<float>(r) / 255.0f);
                outFloats[i * 4u + 1u] = SrgbToLinear(static_cast<float>(g) / 255.0f);
                outFloats[i * 4u + 2u] = SrgbToLinear(static_cast<float>(b) / 255.0f);
                outFloats[i * 4u + 3u] = static_cast<float>(a) / 255.0f;
            }
            break;
        }
        default:
            return false;
    }

    return true;
}

std::string CanonicalizePath(const std::string& path) {
    if (path.empty()) {
        return {};
    }
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::path input(path);
    fs::path canonical = fs::weakly_canonical(input, ec);
    if (!ec) {
        return canonical.string();
    }
    fs::path normalized = input.lexically_normal();
    return normalized.string();
}

std::string DefaultMaterialName(uint32_t index) {
    return "Material " + std::to_string(index);
}

std::string DefaultMeshName(uint32_t index) {
    return "Mesh " + std::to_string(index);
}

struct ThresholdChecksums {
    double head = 0.0;
    double total = 0.0;
};

ThresholdChecksums ComputeThresholdChecksums(const std::vector<float>& values) {
    constexpr size_t kHeadCount = 1024;
    ThresholdChecksums sums{};
    const size_t headLimit = std::min(values.size(), kHeadCount);
    for (size_t i = 0; i < values.size(); ++i) {
        const double value = static_cast<double>(values[i]);
        sums.total += value;
        if (i < headLimit) {
            sums.head += value;
        }
    }
    return sums;
}

constexpr float kSchlickAverageFactor = 1.0f / 21.0f;

inline float ComputeCoatAverage(float coatIor) {
    float eta = std::max(coatIor, 1.0f);
    float numerator = eta - 1.0f;
    float denominator = std::max(eta + 1.0f, 1.0e-6f);
    float ratio = numerator / denominator;
    float f0 = ratio * ratio;
    float average = f0 + (1.0f - f0) * kSchlickAverageFactor;
    return std::clamp(average, 0.0f, 0.999f);
}

inline float ComputeCoatSampleWeight(MaterialType type,
                                     float coatRoughness,
                                     float coatThickness,
                                     float coatAverage) {
    bool hasLayer = (coatThickness > 1.0e-4f) || (coatRoughness > 1.0e-4f) ||
                    type == MaterialType::Plastic || type == MaterialType::CarPaint;
    if (!hasLayer) {
        return 0.0f;
    }

    float weight = coatAverage * 2.5f + coatRoughness * 0.5f;
    if (type == MaterialType::CarPaint) {
        weight = std::max(weight, 0.35f);
    } else if (type == MaterialType::Plastic) {
        weight = std::max(weight, 0.25f);
    }
    return std::clamp(weight, 0.0f, 0.95f);
}

inline simd::float3 ClampColor01(const simd::float3& value) {
    return simd_make_float3(std::clamp(value.x, 0.0f, 1.0f),
                            std::clamp(value.y, 0.0f, 1.0f),
                            std::clamp(value.z, 0.0f, 1.0f));
}

inline simd::float3 ClampPositive(const simd::float3& value) {
    return simd_make_float3(std::max(value.x, 0.0f),
                            std::max(value.y, 0.0f),
                            std::max(value.z, 0.0f));
}

}  // namespace

SceneResources::SceneResources() = default;

SceneResources::~SceneResources() = default;

void SceneResources::initialize(const MetalContext& context) {
    m_device = context.device();
    m_commandQueue = context.commandQueue();
    m_supportsRaytracing = context.supportsRaytracing();
    m_sphereCount = 0;
    m_rectangleCount = 0;
    m_materialCount = 0;
    m_triangleCount = 0;
    m_primitiveCount = 0;
    m_dirty = true;
    m_meshes.clear();
    m_environmentTexture = nil;
    m_meshInfoBuffer = nil;
    m_triangleBuffer = nil;
    m_forceSoftwareOverride = false;
    clearEnvironmentDistribution();

    SceneAccelConfig accelConfig{};
    accelConfig.hardwareRaytracingSupported = m_supportsRaytracing;
    accelConfig.commandQueue = m_commandQueue;
    m_sceneAccel = CreateSceneAccel(accelConfig);
}

uint32_t SceneResources::addMaterial(const simd::float3& baseColor,
                                     float roughness,
                                     MaterialType type,
                                     float indexOfRefraction,
                                     const simd::float3& emission,
                                     bool emissionUsesEnvironment,
                                     const simd::float3& conductorEta,
                                     const simd::float3& conductorK,
                                     bool hasConductorParameters,
                                     float coatRoughness,
                                     float coatThickness,
                                     const simd::float3& coatTint,
                                     const simd::float3& coatAbsorption,
                                     float coatIor,
                                     const simd::float3& dielectricSigmaA,
                                     const simd::float3& sssSigmaA,
                                     const simd::float3& sssSigmaS,
                                     float sssMeanFreePath,
                                     float sssAnisotropy,
                                     uint32_t sssMethod,
                                     bool sssCoatEnabled,
                                     bool sssSigmaOverride,
                                     float carpaintBaseMetallic,
                                     float carpaintBaseRoughness,
                                     float carpaintFlakeSampleWeight,
                                     float carpaintFlakeRoughness,
                                     float carpaintFlakeAnisotropy,
                                     float carpaintFlakeNormalStrength,
                                     float carpaintFlakeScale,
                                     float carpaintFlakeReflectanceScale,
                                     simd::float3 carpaintBaseEta,
                                     simd::float3 carpaintBaseK,
                                     bool carpaintHasBaseConductor,
                                     simd::float3 carpaintBaseTint,
                                     std::string name) {
    if (m_materialCount >= kMaxMaterials) {
        return static_cast<uint32_t>(kMaxMaterials - 1);
    }

    uint32_t index = m_materialCount++;
    MaterialData material{};

    simd::float3 clampedBaseColor = ClampColor01(baseColor);
    float clampedRoughness = std::clamp(roughness, 0.0f, 1.0f);
    material.baseColorRoughness = simd_make_float4(clampedBaseColor, clampedRoughness);

    float clampedIor = std::max(indexOfRefraction, 0.0f);
    float clampedCoatIor = std::max(coatIor, 0.0f);
    material.typeEta = simd_make_float4(static_cast<float>(type),
                                        clampedIor,
                                        clampedCoatIor,
                                        0.0f);

    material.emission = simd_make_float4(emission, emissionUsesEnvironment ? 1.0f : 0.0f);

    simd::float3 etaClamped = ClampPositive(conductorEta);
    simd::float3 kClamped = ClampPositive(conductorK);
    float conductorFlag = hasConductorParameters ? 1.0f : 0.0f;
    material.conductorEta = simd_make_float4(etaClamped, conductorFlag);
    material.conductorK = simd_make_float4(kClamped, conductorFlag);

    float clampedCoatRoughness = std::clamp(coatRoughness, 0.0f, 1.0f);
    float clampedCoatThickness = std::max(coatThickness, 0.0f);
    float coatAverage = ComputeCoatAverage(clampedCoatIor);
    float coatSampleWeight = ComputeCoatSampleWeight(type,
                                                    clampedCoatRoughness,
                                                    clampedCoatThickness,
                                                    coatAverage);
    coatSampleWeight = std::clamp(coatSampleWeight, 0.0f, 0.95f);
    material.coatParams = simd_make_float4(clampedCoatRoughness,
                                           clampedCoatThickness,
                                           coatSampleWeight,
                                           coatAverage);

    simd::float3 coatTintClamped = ClampColor01(coatTint);
    simd::float3 coatAbsorptionClamped = ClampPositive(coatAbsorption);
    material.coatTint = simd_make_float4(coatTintClamped, 0.0f);
    material.coatAbsorption = simd_make_float4(coatAbsorptionClamped, 0.0f);
    simd::float3 dielectricSigmaAClamped = ClampPositive(dielectricSigmaA);
    material.dielectricSigmaA = simd_make_float4(dielectricSigmaAClamped, 0.0f);

    simd::float3 sigmaAClamped = ClampPositive(sssSigmaA);
    simd::float3 sigmaSClamped = ClampPositive(sssSigmaS);
    float sssFlag = sssSigmaOverride ? 1.0f : 0.0f;
    material.sssSigmaA = simd_make_float4(sigmaAClamped, sssFlag);
    float clampedAnisotropy = std::clamp(sssAnisotropy, -0.99f, 0.99f);
    material.sssSigmaS = simd_make_float4(sigmaSClamped, clampedAnisotropy);
    material.sssParams = simd_make_float4(std::max(sssMeanFreePath, 0.0f),
                                          static_cast<float>(sssMethod),
                                          sssCoatEnabled ? 1.0f : 0.0f,
                                          0.0f);

    float baseMetallicClamped = std::clamp(carpaintBaseMetallic, 0.0f, 1.0f);
    float baseRoughnessClamped = std::clamp(carpaintBaseRoughness, 0.0f, 1.0f);
    float flakeReflectanceClamped = std::clamp(carpaintFlakeReflectanceScale, 0.0f, 1.0f);
    float flakeSampleWeightClamped = std::clamp(carpaintFlakeSampleWeight, 0.0f, 0.95f);
    // Keep flake sampling weight aligned with the energy contributed by the lobe
    flakeSampleWeightClamped = std::clamp(flakeSampleWeightClamped * std::max(flakeReflectanceClamped, 0.01f), 0.0f, 0.95f);
    float flakeRoughnessClamped = std::clamp(carpaintFlakeRoughness, 0.0f, 1.0f);
    float flakeAnisotropyClamped = std::clamp(carpaintFlakeAnisotropy, -0.99f, 0.99f);
    float flakeNormalStrengthClamped = std::clamp(carpaintFlakeNormalStrength, 0.0f, 1.0f);
    float flakeScaleClamped = std::max(carpaintFlakeScale, 1.0e-4f);
    material.carpaintBaseParams = simd_make_float4(baseMetallicClamped,
                                                   baseRoughnessClamped,
                                                   flakeScaleClamped,
                                                   flakeReflectanceClamped);
    material.carpaintFlakeParams = simd_make_float4(flakeSampleWeightClamped,
                                                    flakeRoughnessClamped,
                                                    flakeAnisotropyClamped,
                                                    flakeNormalStrengthClamped);

    simd::float3 baseEtaClamped = ClampPositive(carpaintBaseEta);
    simd::float3 baseKClamped = ClampPositive(carpaintBaseK);
    float baseConductorFlag = carpaintHasBaseConductor ? 1.0f : 0.0f;
    if (!carpaintHasBaseConductor) {
        baseEtaClamped = simd_make_float3(0.0f, 0.0f, 0.0f);
        baseKClamped = simd_make_float3(0.0f, 0.0f, 0.0f);
    }
    material.carpaintBaseEta = simd_make_float4(baseEtaClamped, baseConductorFlag);
    material.carpaintBaseK = simd_make_float4(baseKClamped, baseConductorFlag);
    material.carpaintBaseTint = simd_make_float4(ClampColor01(carpaintBaseTint), 0.0f);

    m_materials[index] = material;
    m_materialDefaults[index] = material;
    m_materialNames[index] = name.empty() ? DefaultMaterialName(index) : std::move(name);
    m_dirty = true;
    return index;
}

uint32_t SceneResources::addMaterial(const simd::float3& albedo,
                                     float fuzz,
                                     MaterialType type,
                                     float indexOfRefraction,
                                     const simd::float3& emission,
                                     bool emissionUsesEnvironment,
                                     std::string name) {
    float clampedRoughness = std::clamp(fuzz, 0.0f, 1.0f);
    simd::float3 zero = simd_make_float3(0.0f, 0.0f, 0.0f);
    simd::float3 coatTint = simd_make_float3(1.0f, 1.0f, 1.0f);

    return addMaterial(albedo,
                       clampedRoughness,
                       type,
                       indexOfRefraction,
                       emission,
                       emissionUsesEnvironment,
                       zero,
                       zero,
                       false,
                       0.0f,
                       0.0f,
                       coatTint,
                       zero,
                       1.5f,
                       zero,
                       zero,
                       zero,
                       0.0f,
                       0.0f,
                       0u,
                       false,
                       false,
                       0.0f,
                       clampedRoughness,
                       0.0f,
                       0.0f,
                       0.0f,
                       0.0f,
                       1.0f,
                       1.0f,
                       zero,
                       zero,
                       false,
                       ClampColor01(simd_make_float3(1.0f, 1.0f, 1.0f)),
                       std::move(name));
}

const std::string& SceneResources::materialName(uint32_t index) const {
    static const std::string kEmptyName;
    if (index >= m_materialCount) {
        return kEmptyName;
    }
    return m_materialNames[index];
}

bool SceneResources::updateMaterial(uint32_t index, const MaterialData& material) {
    if (index >= m_materialCount) {
        return false;
    }
    m_materials[index] = material;
    uploadMaterialToGpu(index);
    return true;
}

bool SceneResources::resetMaterial(uint32_t index) {
    if (index >= m_materialCount) {
        return false;
    }
    m_materials[index] = m_materialDefaults[index];
    uploadMaterialToGpu(index);
    return true;
}

void SceneResources::setSoftwareRayTracingOverride(bool force) {
    setForceSoftwareBvh(force);
}

void SceneResources::addSphere(const simd::float3& center, float radius, uint32_t materialIndex) {
    if (m_sphereCount >= kMaxSpheres) {
        return;
    }

    if (materialIndex >= m_materialCount) {
        materialIndex = (m_materialCount == 0) ? 0u : (m_materialCount - 1);
    }

    SphereData sphere{};
    sphere.centerRadius = simd_make_float4(center, radius);
    sphere.materialIndex = simd_make_uint4(materialIndex, 0u, 0u, 0u);
    m_spheres[m_sphereCount++] = sphere;
    m_dirty = true;
}

bool SceneResources::setEnvironmentMap(const std::string& path) {
    if (!m_device) {
        return false;
    }

    if (path.empty()) {
        clearEnvironmentMap();
        return false;
    }

    EnvGpuHandles handles;
    bool success = reloadEnvironmentIfNeeded(path, &handles);
    if (!success) {
        return false;
    }

    return m_environmentTexture != nil;
}

bool SceneResources::reloadEnvironmentIfNeeded(const std::string& path, EnvGpuHandles* outHandles) {
    if (!m_device) {
        return false;
    }

    if (path.empty()) {
        const bool hadEnvironment = (m_environmentTexture != nil) || (m_environmentAliasCount > 0);
        if (hadEnvironment) {
            clearEnvironmentMap();
        }
        if (outHandles) {
            *outHandles = EnvGpuHandles{};
        }
        return true;
    }

    const std::string canonicalPath = CanonicalizePath(path);
    const bool needsReload =
        (m_environmentTexture == nil) || (canonicalPath != m_environmentPath);
    if (!needsReload) {
        if (outHandles) {
            outHandles->texture = m_environmentTexture;
            outHandles->conditionalAlias = m_environmentConditionalAliasBuffer;
            outHandles->marginalAlias = m_environmentMarginalAliasBuffer;
            outHandles->pdf = m_environmentPdfBuffer;
            outHandles->aliasCount = m_environmentAliasCount;
            outHandles->width = m_environmentWidth;
            outHandles->height = m_environmentHeight;
            outHandles->thresholdHeadSum = 0.0;
            outHandles->thresholdTotalSum = 0.0;
        }
        return true;
    }

    CFAbsoluteTime rebuildStart = CFAbsoluteTimeGetCurrent();
    NSString* nsPath = [NSString stringWithUTF8String:path.c_str()];
    NSURL* url = [NSURL fileURLWithPath:nsPath];
    MTKTextureLoader* loader = [[MTKTextureLoader alloc] initWithDevice:m_device];
    NSDictionary* options = @{MTKTextureLoaderOptionSRGB : @NO,
                              MTKTextureLoaderOptionTextureUsage : @(MTLTextureUsageShaderRead),
                              MTKTextureLoaderOptionTextureStorageMode : @(MTLStorageModeShared),
                              MTKTextureLoaderOptionAllocateMipmaps : @NO,
                              MTKTextureLoaderOptionGenerateMipmaps : @NO};
    NSError* error = nil;
    id<MTLTexture> texture = [loader newTextureWithContentsOfURL:url options:options error:&error];
    if (!texture) {
        if (error) {
            NSLog(@"Failed to load environment map %@: %@", nsPath, error);
        } else {
            NSLog(@"Failed to load environment map %@", nsPath);
        }
        return false;
    }

    const NSUInteger width = texture.width;
    const NSUInteger height = texture.height;
    if (width == 0 || height == 0) {
        NSLog(@"Environment map %@ has invalid dimensions (%lu x %lu)",
              nsPath,
              static_cast<unsigned long>(width),
              static_cast<unsigned long>(height));
        return false;
    }

    std::vector<float> rgbaFloats;
    if (!CopyTextureToFloat(texture, rgbaFloats)) {
        NSLog(@"Failed to extract texels for environment map %@", nsPath);
        return false;
    }

    EnvGpuHandles tempHandles;
    if (!buildEnvironmentDistribution(rgbaFloats.data(),
                                      static_cast<uint32_t>(width),
                                      static_cast<uint32_t>(height),
                                      tempHandles)) {
        return false;
    }

    texture.label = @"Environment Map";
    tempHandles.texture = texture;

    const double elapsedMs = (CFAbsoluteTimeGetCurrent() - rebuildStart) * 1000.0;
    std::string fileName = std::filesystem::path(path).filename().string();
    NSLog(@"[EnvRebuild] file=\"%s\" size=%lux%lu aliasN=%u thrHead=%.5f thrSum=%.5f elapsedMs=%.2f",
          fileName.c_str(),
          static_cast<unsigned long>(width),
          static_cast<unsigned long>(height),
          tempHandles.aliasCount,
          tempHandles.thresholdHeadSum,
          tempHandles.thresholdTotalSum,
          elapsedMs);

    m_environmentTexture = tempHandles.texture;
    m_environmentConditionalAliasBuffer = tempHandles.conditionalAlias;
    m_environmentMarginalAliasBuffer = tempHandles.marginalAlias;
    m_environmentPdfBuffer = tempHandles.pdf;
    m_environmentAliasCount = tempHandles.aliasCount;
    m_environmentWidth = tempHandles.width;
    m_environmentHeight = tempHandles.height;
    m_environmentPath = canonicalPath;

    if (outHandles) {
        *outHandles = tempHandles;
    }
    return true;
}

void SceneResources::clearEnvironmentMap() {
    m_environmentTexture = nil;
    clearEnvironmentDistribution();
    m_environmentPath.clear();
}

void SceneResources::clearEnvironmentDistribution() {
    m_environmentConditionalAliasBuffer = nil;
    m_environmentMarginalAliasBuffer = nil;
    m_environmentPdfBuffer = nil;
    m_environmentAliasCount = 0;
    m_environmentWidth = 0;
    m_environmentHeight = 0;
}

bool SceneResources::buildEnvironmentDistribution(const float* rgba32,
                                                  uint32_t width,
                                                  uint32_t height,
                                                  EnvGpuHandles& outHandles) {
    outHandles = EnvGpuHandles{};

    if (!rgba32 || width == 0 || height == 0 || !m_device) {
        return false;
    }

    EnvImportanceDistribution distribution;
    std::string errorMessage;
    if (!BuildEnvImportanceDistribution(rgba32,
                                        width,
                                        height,
                                        &distribution,
                                        &errorMessage)) {
        if (!errorMessage.empty()) {
            NSLog(@"Environment importance sampler build failed: %s", errorMessage.c_str());
        }
        return false;
    }

    const size_t texelCount = static_cast<size_t>(distribution.aliasCount);
    if (texelCount == 0 || distribution.marginalAlias.empty() || distribution.texelPdf.empty()) {
        return false;
    }

    const NSUInteger conditionalBytes =
        static_cast<NSUInteger>(texelCount * sizeof(EnvironmentAliasEntry));
    const NSUInteger marginalBytes = static_cast<NSUInteger>(
        distribution.marginalAlias.size() * sizeof(EnvironmentAliasEntry));
    const NSUInteger pdfBytes =
        static_cast<NSUInteger>(distribution.texelPdf.size() * sizeof(float));
    if (conditionalBytes == 0 || marginalBytes == 0 || pdfBytes == 0) {
        return false;
    }

    std::vector<EnvironmentAliasEntry> conditionalEntries(texelCount);
    for (size_t i = 0; i < texelCount; ++i) {
        conditionalEntries[i].threshold =
            std::clamp(distribution.conditionalThreshold[i], 0.0f, 1.0f);
        conditionalEntries[i].alias = distribution.conditionalAlias[i];
        conditionalEntries[i].padding0 = 0;
        conditionalEntries[i].padding1 = 0;
    }

    std::vector<EnvironmentAliasEntry> marginalEntries(distribution.marginalAlias.size());
    for (size_t i = 0; i < distribution.marginalAlias.size(); ++i) {
        marginalEntries[i].threshold = std::clamp(distribution.marginalThreshold[i], 0.0f, 1.0f);
        marginalEntries[i].alias = distribution.marginalAlias[i];
        marginalEntries[i].padding0 = 0;
        marginalEntries[i].padding1 = 0;
    }

    MTLBufferHandle conditionalBuffer =
        [m_device newBufferWithLength:conditionalBytes options:MTLResourceStorageModeShared];
    MTLBufferHandle marginalBuffer =
        [m_device newBufferWithLength:marginalBytes options:MTLResourceStorageModeShared];
    MTLBufferHandle pdfBuffer =
        [m_device newBufferWithLength:pdfBytes options:MTLResourceStorageModeShared];

    if (!conditionalBuffer || !marginalBuffer || !pdfBuffer) {
        return false;
    }

    auto* conditionalPtr =
        reinterpret_cast<EnvironmentAliasEntry*>([conditionalBuffer contents]);
    memcpy(conditionalPtr, conditionalEntries.data(), conditionalBytes);

    auto* marginalPtr = reinterpret_cast<EnvironmentAliasEntry*>([marginalBuffer contents]);
    memcpy(marginalPtr, marginalEntries.data(), marginalBytes);

    memcpy([pdfBuffer contents], distribution.texelPdf.data(), pdfBytes);

    const ThresholdChecksums sums = ComputeThresholdChecksums(distribution.conditionalThreshold);
    outHandles.conditionalAlias = conditionalBuffer;
    outHandles.marginalAlias = marginalBuffer;
    outHandles.pdf = pdfBuffer;
    outHandles.aliasCount = distribution.aliasCount;
    outHandles.width = distribution.width;
    outHandles.height = distribution.height;
    outHandles.thresholdHeadSum = sums.head;
    outHandles.thresholdTotalSum = sums.total;
    return true;
}

void SceneResources::setForceSoftwareBvh(bool force) {
    if (m_forceSoftwareOverride == force) {
        return;
    }

    m_forceSoftwareOverride = force;

    if (m_sceneAccel) {
        m_sceneAccel->clear();
        m_sceneAccel.reset();
    }

    SceneAccelConfig accelConfig{};
    accelConfig.hardwareRaytracingSupported = m_supportsRaytracing && !m_forceSoftwareOverride;
    accelConfig.commandQueue = m_commandQueue;
    m_sceneAccel = CreateSceneAccel(accelConfig);

    m_intersectionProvider = IntersectionProvider{};
    m_meshInfoBuffer = nil;
    m_triangleBuffer = nil;
    m_dirty = true;
}

void SceneResources::addRectangle(const simd::float3& boundsMin,
                                  const simd::float3& boundsMax,
                                  uint32_t normalAxis,
                                  bool normalPositive,
                                  bool twoSided,
                                  uint32_t materialIndex) {
    if (m_rectangleCount >= kMaxRectangles) {
        return;
    }
    if (materialIndex >= m_materialCount) {
        materialIndex = (m_materialCount == 0) ? 0u : (m_materialCount - 1);
    }
    if (normalAxis > 2u) {
        normalAxis = 2u;
    }

    simd::float3 min = simd_make_float3(std::min(boundsMin.x, boundsMax.x),
                                        std::min(boundsMin.y, boundsMax.y),
                                        std::min(boundsMin.z, boundsMax.z));
    simd::float3 max = simd_make_float3(std::max(boundsMin.x, boundsMax.x),
                                        std::max(boundsMin.y, boundsMax.y),
                                        std::max(boundsMin.z, boundsMax.z));

    simd::float3 corner{};
    simd::float3 edgeU{};
    simd::float3 edgeV{};

    switch (normalAxis) {
        case 0: { // X constant
            float y0 = min.y;
            float y1 = max.y;
            float z0 = min.z;
            float z1 = max.z;
            edgeU = simd_make_float3(0.0f, y1 - y0, 0.0f);
            if (normalPositive) {
                corner = simd_make_float3(max.x, y0, z0);
                edgeV = simd_make_float3(0.0f, 0.0f, z1 - z0);
            } else {
                corner = simd_make_float3(min.x, y0, z1);
                edgeV = simd_make_float3(0.0f, 0.0f, z0 - z1);
            }
            break;
        }
        case 1: { // Y constant
            float x0 = min.x;
            float x1 = max.x;
            float z0 = min.z;
            float z1 = max.z;
            edgeU = simd_make_float3(x1 - x0, 0.0f, 0.0f);
            if (normalPositive) {
                corner = simd_make_float3(x0, max.y, z0);
                edgeV = simd_make_float3(0.0f, 0.0f, z1 - z0);
            } else {
                corner = simd_make_float3(x0, min.y, z1);
                edgeV = simd_make_float3(0.0f, 0.0f, z0 - z1);
            }
            break;
        }
        default: { // Z constant
            float x0 = min.x;
            float x1 = max.x;
            float y0 = min.y;
            float y1 = max.y;
            edgeU = simd_make_float3(x1 - x0, 0.0f, 0.0f);
            if (normalPositive) {
                corner = simd_make_float3(x0, y0, max.z);
                edgeV = simd_make_float3(0.0f, y1 - y0, 0.0f);
            } else {
                corner = simd_make_float3(x1, y0, min.z);
                edgeV = simd_make_float3(0.0f, y1 - y0, 0.0f);
                edgeU = simd_make_float3(x0 - x1, 0.0f, 0.0f);
            }
            break;
        }
    }

    simd::float3 desiredNormal = simd_make_float3(0.0f, 0.0f, 0.0f);
    switch (normalAxis) {
        case 0:
            desiredNormal = simd_make_float3(normalPositive ? 1.0f : -1.0f, 0.0f, 0.0f);
            break;
        case 1:
            desiredNormal = simd_make_float3(0.0f, normalPositive ? 1.0f : -1.0f, 0.0f);
            break;
        default:
            desiredNormal = simd_make_float3(0.0f, 0.0f, normalPositive ? 1.0f : -1.0f);
            break;
    }

    storeRectangleOriented(corner, edgeU, edgeV, twoSided, materialIndex, desiredNormal);
}

void SceneResources::addBox(const simd::float3& minCorner,
                            const simd::float3& maxCorner,
                            uint32_t materialIndex,
                            bool includeBottomFace,
                            bool twoSided) {
    addBoxTransformed(minCorner, maxCorner, materialIndex, matrix_identity_float4x4, includeBottomFace, twoSided);
}

void SceneResources::addBoxTransformed(const simd::float3& minCorner,
                                       const simd::float3& maxCorner,
                                       uint32_t materialIndex,
                                       const simd::float4x4& transform,
                                       bool includeBottomFace,
                                       bool twoSided) {
    if (materialIndex >= m_materialCount) {
        materialIndex = (m_materialCount == 0) ? 0u : (m_materialCount - 1);
    }

    simd::float3 min = simd_make_float3(std::min(minCorner.x, maxCorner.x),
                                        std::min(minCorner.y, maxCorner.y),
                                        std::min(minCorner.z, maxCorner.z));
    simd::float3 max = simd_make_float3(std::max(minCorner.x, maxCorner.x),
                                        std::max(minCorner.y, maxCorner.y),
                                        std::max(minCorner.z, maxCorner.z));

    struct Face {
        simd::float3 corner;
        simd::float3 edgeU;
        simd::float3 edgeV;
        simd::float3 normal;
        bool include;
    } faces[6] = {
        {simd_make_float3(max.x, min.y, min.z), simd_make_float3(0.0f, max.y - min.y, 0.0f), simd_make_float3(0.0f, 0.0f, max.z - min.z), simd_make_float3(1.0f, 0.0f, 0.0f), true},   // +X
        {simd_make_float3(min.x, min.y, max.z), simd_make_float3(0.0f, max.y - min.y, 0.0f), simd_make_float3(0.0f, 0.0f, min.z - max.z), simd_make_float3(-1.0f, 0.0f, 0.0f), true},   // -X
        {simd_make_float3(min.x, max.y, min.z), simd_make_float3(max.x - min.x, 0.0f, 0.0f), simd_make_float3(0.0f, 0.0f, max.z - min.z), simd_make_float3(0.0f, 1.0f, 0.0f), true},   // +Y
        {simd_make_float3(min.x, min.y, max.z), simd_make_float3(max.x - min.x, 0.0f, 0.0f), simd_make_float3(0.0f, 0.0f, min.z - max.z), simd_make_float3(0.0f, -1.0f, 0.0f), includeBottomFace}, // -Y
        {simd_make_float3(min.x, min.y, max.z), simd_make_float3(max.x - min.x, 0.0f, 0.0f), simd_make_float3(0.0f, max.y - min.y, 0.0f), simd_make_float3(0.0f, 0.0f, 1.0f), true},   // +Z
        {simd_make_float3(max.x, min.y, min.z), simd_make_float3(min.x - max.x, 0.0f, 0.0f), simd_make_float3(0.0f, max.y - min.y, 0.0f), simd_make_float3(0.0f, 0.0f, -1.0f), true},   // -Z
    };

    auto transformPoint = [&](const simd::float3& p) -> simd::float3 {
        simd::float4 result = simd_mul(transform, simd_make_float4(p, 1.0f));
        return simd_make_float3(result.x, result.y, result.z);
    };

    auto transformVector = [&](const simd::float3& v) -> simd::float3 {
        simd::float4 result = simd_mul(transform, simd_make_float4(v, 0.0f));
        return simd_make_float3(result.x, result.y, result.z);
    };

    for (const Face& face : faces) {
        if (!face.include) {
            continue;
        }
        simd::float3 corner = transformPoint(face.corner);
        simd::float3 edgeU = transformVector(face.edgeU);
        simd::float3 edgeV = transformVector(face.edgeV);
        simd::float3 desiredNormal = transformVector(face.normal);
        storeRectangleOriented(corner, edgeU, edgeV, twoSided, materialIndex, desiredNormal);
    }
}

uint32_t SceneResources::addMesh(const MeshVertex* vertices,
                                 uint32_t vertexCount,
                                 const uint32_t* indices,
                                 uint32_t indexCount,
                                 const simd::float4x4& localToWorld,
                                 uint32_t materialIndex,
                                 std::string name) {
    if (!vertices || vertexCount == 0 || !indices || indexCount == 0) {
        NSLog(@"SceneResources::addMesh received empty mesh data");
        return std::numeric_limits<uint32_t>::max();
    }
    if (indexCount % 3 != 0) {
        NSLog(@"SceneResources::addMesh requires triangle indices (count must be divisible by 3)");
        return std::numeric_limits<uint32_t>::max();
    }

    uint32_t meshIndex = static_cast<uint32_t>(m_meshes.size());
    Mesh mesh{};
    mesh.vertices.assign(vertices, vertices + vertexCount);
    mesh.indices.assign(indices, indices + indexCount);
    mesh.localToWorld = localToWorld;
    mesh.defaultLocalToWorld = localToWorld;
    mesh.materialIndex = materialIndex;
    mesh.name = name.empty() ? DefaultMeshName(meshIndex) : std::move(name);
    m_meshes.push_back(std::move(mesh));
    m_dirty = true;
    return static_cast<uint32_t>(m_meshes.size() - 1);
}

void SceneResources::clear() {
    m_sphereCount = 0;
    m_rectangleCount = 0;
    m_materialCount = 0;
    m_triangleCount = 0;
    m_primitiveCount = 0;
    m_materialNames.fill(std::string{});
    m_materialDefaults.fill(MaterialData{});
    m_sphereBuffer = nil;
    m_materialBuffer = nil;
    m_rectangleBuffer = nil;
    m_meshInfoBuffer = nil;
    m_triangleBuffer = nil;
    m_meshVertexBuffer = nil;
    m_meshIndexBuffer = nil;
    m_intersectionProvider = IntersectionProvider{};
    for (auto& mesh : m_meshes) {
        mesh.vertexBuffer = nil;
        mesh.indexBuffer = nil;
    }
    m_meshes.clear();
    if (m_sceneAccel) {
        m_sceneAccel->clear();
    }
    m_dirty = true;
    clearEnvironmentMap();
}

bool SceneResources::setMeshTransform(uint32_t meshIndex, const simd::float4x4& localToWorld) {
    if (meshIndex >= m_meshes.size()) {
        return false;
    }
    m_meshes[meshIndex].localToWorld = localToWorld;
    m_dirty = true;
    return true;
}

bool SceneResources::resetMeshTransform(uint32_t meshIndex) {
    if (meshIndex >= m_meshes.size()) {
        return false;
    }
    m_meshes[meshIndex].localToWorld = m_meshes[meshIndex].defaultLocalToWorld;
    m_dirty = true;
    return true;
}

const simd::float4x4& SceneResources::meshTransform(uint32_t meshIndex) const {
    static const simd::float4x4 kIdentity = matrix_identity_float4x4;
    if (meshIndex >= m_meshes.size()) {
        return kIdentity;
    }
    return m_meshes[meshIndex].localToWorld;
}

const std::string& SceneResources::meshName(uint32_t meshIndex) const {
    static const std::string kEmptyName;
    if (meshIndex >= m_meshes.size()) {
        return kEmptyName;
    }
    return m_meshes[meshIndex].name;
}

void SceneResources::uploadMaterialToGpu(uint32_t index) {
    if (!m_device || index >= m_materialCount) {
        return;
    }
    const NSUInteger materialBytes =
        static_cast<NSUInteger>(m_materialCount) * sizeof(MaterialData);
    if (materialBytes == 0) {
        return;
    }
    if (!m_materialBuffer || m_materialBuffer.length < materialBytes) {
        m_materialBuffer = [m_device newBufferWithLength:materialBytes
                                                 options:MTLResourceStorageModeShared];
    }
    if (!m_materialBuffer) {
        return;
    }
    uint8_t* dst = reinterpret_cast<uint8_t*>([m_materialBuffer contents]);
    if (!dst) {
        return;
    }
    memcpy(dst + static_cast<NSUInteger>(index) * sizeof(MaterialData),
           &m_materials[index],
           sizeof(MaterialData));
}

void SceneResources::uploadBuffers() {
    if (!m_device) {
        return;
    }

    const NSUInteger sphereBytes = static_cast<NSUInteger>(m_sphereCount) * sizeof(SphereData);
    if (sphereBytes > 0) {
        if (!m_sphereBuffer || m_sphereBuffer.length < sphereBytes) {
            m_sphereBuffer = [m_device newBufferWithLength:sphereBytes
                                                   options:MTLResourceStorageModeShared];
        }
        if (m_sphereBuffer) {
            memcpy([m_sphereBuffer contents], m_spheres.data(), sphereBytes);
        }
    } else {
        m_sphereBuffer = nil;
    }

    const NSUInteger materialBytes = static_cast<NSUInteger>(m_materialCount) * sizeof(MaterialData);
    if (materialBytes > 0) {
        if (!m_materialBuffer || m_materialBuffer.length < materialBytes) {
            m_materialBuffer = [m_device newBufferWithLength:materialBytes
                                                     options:MTLResourceStorageModeShared];
        }
        if (m_materialBuffer) {
            memcpy([m_materialBuffer contents], m_materials.data(), materialBytes);
        }
    } else {
        m_materialBuffer = nil;
    }

    uploadRectangles();
    uploadMeshes();
}

void SceneResources::rebuildAccelerationStructures() {
    // Wait for any previous rebuild to complete first
    if (m_rebuildInProgress && m_rebuildCommandBuffer) {
        [m_rebuildCommandBuffer waitUntilCompleted];
        m_rebuildCommandBuffer = nullptr;
        m_rebuildInProgress = false;
    }
    
    if (!m_dirty) {
        return;
    }
    
    uploadBuffers();

    SceneAccelBuildInput buildInput{};
    buildInput.device = m_device;
    buildInput.commandQueue = m_commandQueue;
    buildInput.spheres = m_spheres.data();
    buildInput.sphereCount = m_sphereCount;
    std::vector<SceneAccelMeshInput> meshInputs;
    meshInputs.reserve(m_meshes.size());
    std::vector<PathTracerShaderTypes::MeshInfo> meshInfos;
    meshInfos.reserve(m_meshes.size());
    std::vector<PathTracerShaderTypes::TriangleData> triangleData;
    size_t totalVertexCount = 0;
    size_t totalTriangleCount = 0;
    for (const auto& mesh : m_meshes) {
        totalVertexCount += mesh.vertices.size();
        totalTriangleCount += mesh.indices.size() / 3u;
    }
    std::vector<PathTracerShaderTypes::SceneVertex> sceneVertices;
    sceneVertices.reserve(totalVertexCount);
    std::vector<simd::uint3> sceneIndices;
    sceneIndices.reserve(totalTriangleCount);
    uint32_t triangleOffset = 0;

    for (const auto& mesh : m_meshes) {
        SceneAccelMeshInput meshInput{};
        meshInput.vertexBuffer = mesh.vertexBuffer;
        meshInput.indexBuffer = mesh.indexBuffer;
        meshInput.vertexStride = sizeof(MeshVertex);
        meshInput.vertexCount = static_cast<uint32_t>(mesh.vertices.size());
        meshInput.indexCount = static_cast<uint32_t>(mesh.indices.size());
        meshInput.localToWorldTransform = mesh.localToWorld;
        meshInput.materialIndex = mesh.materialIndex;
        meshInputs.push_back(meshInput);

        PathTracerShaderTypes::MeshInfo info{};
        info.materialIndex = mesh.materialIndex;
        info.triangleOffset = triangleOffset;
        uint32_t triangleCount = meshInput.indexCount / 3u;
        info.triangleCount = triangleCount;
        info.vertexOffset = static_cast<uint32_t>(sceneVertices.size());
        info.vertexCount = static_cast<uint32_t>(mesh.vertices.size());
        info.indexOffset = static_cast<uint32_t>(sceneIndices.size());
        info.indexCount = triangleCount;
        info.localToWorld = mesh.localToWorld;
        info.worldToLocal = simd_inverse(mesh.localToWorld);
        meshInfos.push_back(info);

        sceneVertices.reserve(sceneVertices.size() + mesh.vertices.size());
        for (const auto& vertex : mesh.vertices) {
            PathTracerShaderTypes::SceneVertex packed{};
            packed.position = simd_make_float4(vertex.position, 1.0f);
            packed.normal = simd_make_float4(vertex.normal, 0.0f);
            packed.tangent = simd_make_float4(1.0f, 0.0f, 0.0f, 1.0f);
            packed.uv = simd_make_float4(vertex.uv, 0.0f, 0.0f);
            sceneVertices.push_back(packed);
        }

        sceneIndices.reserve(sceneIndices.size() + triangleCount);

        for (uint32_t tri = 0; tri < triangleCount; ++tri) {
            uint32_t base = tri * 3u;
            PathTracerShaderTypes::TriangleData triData{};
            if (base + 2u >= mesh.indices.size()) {
                triangleData.push_back(triData);
                simd::uint3 packed = simd_make_uint3(info.vertexOffset, info.vertexOffset, info.vertexOffset);
                sceneIndices.push_back(packed);
                continue;
            }
            uint32_t i0 = mesh.indices[base + 0u];
            uint32_t i1 = mesh.indices[base + 1u];
            uint32_t i2 = mesh.indices[base + 2u];
            if (i0 >= mesh.vertices.size() ||
                i1 >= mesh.vertices.size() ||
                i2 >= mesh.vertices.size()) {
                triangleData.push_back(triData);
                simd::uint3 packed = simd_make_uint3(info.vertexOffset, info.vertexOffset, info.vertexOffset);
                sceneIndices.push_back(packed);
                continue;
            }

            const MeshVertex& v0 = mesh.vertices[i0];
            const MeshVertex& v1 = mesh.vertices[i1];
            const MeshVertex& v2 = mesh.vertices[i2];

            triData.v0 = simd_make_float4(v0.position, 0.0f);
            triData.v1 = simd_make_float4(v1.position, 0.0f);
            triData.v2 = simd_make_float4(v2.position, 0.0f);
            uint32_t matIndex = mesh.materialIndex;
            if (m_materialCount > 0) {
                matIndex = std::min<uint32_t>(matIndex, m_materialCount - 1);
            } else {
                matIndex = 0u;
            }
            triData.metadata = simd_make_uint4(matIndex,
                                               static_cast<uint32_t>(meshInputs.size() - 1),
                                               0u,
                                               0u);
            triangleData.push_back(triData);

            simd::uint3 packedIdx = simd_make_uint3(info.vertexOffset + i0,
                                                    info.vertexOffset + i1,
                                                    info.vertexOffset + i2);
            sceneIndices.push_back(packedIdx);
        }
        triangleOffset += triangleCount;
    }
    buildInput.meshes = meshInputs.empty() ? nullptr : meshInputs.data();
    buildInput.meshCount = static_cast<uint32_t>(meshInputs.size());

    if (!meshInfos.empty() && m_device) {
        const NSUInteger infoBytes =
            static_cast<NSUInteger>(meshInfos.size() * sizeof(PathTracerShaderTypes::MeshInfo));
        if (!m_meshInfoBuffer || m_meshInfoBuffer.length < infoBytes) {
            m_meshInfoBuffer =
                [m_device newBufferWithLength:infoBytes options:MTLResourceStorageModeShared];
        }
        if (m_meshInfoBuffer) {
            memcpy([m_meshInfoBuffer contents], meshInfos.data(), infoBytes);
        }
    } else {
        m_meshInfoBuffer = nil;
    }

    if (!triangleData.empty() && m_device) {
        const NSUInteger triangleBytes =
            static_cast<NSUInteger>(triangleData.size() * sizeof(PathTracerShaderTypes::TriangleData));
        if (!m_triangleBuffer || m_triangleBuffer.length < triangleBytes) {
            m_triangleBuffer =
                [m_device newBufferWithLength:triangleBytes options:MTLResourceStorageModeShared];
        }
        if (m_triangleBuffer) {
            memcpy([m_triangleBuffer contents], triangleData.data(), triangleBytes);
        }
    } else {
        m_triangleBuffer = nil;
    }

    if (!sceneVertices.empty() && m_device) {
        const NSUInteger vertexBytes =
            static_cast<NSUInteger>(sceneVertices.size() * sizeof(PathTracerShaderTypes::SceneVertex));
        if (!m_meshVertexBuffer || m_meshVertexBuffer.length < vertexBytes) {
            m_meshVertexBuffer =
                [m_device newBufferWithLength:vertexBytes options:MTLResourceStorageModeShared];
        }
        if (m_meshVertexBuffer) {
            memcpy([m_meshVertexBuffer contents], sceneVertices.data(), vertexBytes);
        }
    } else {
        m_meshVertexBuffer = nil;
    }

    if (!sceneIndices.empty() && m_device) {
        const NSUInteger indexBytes =
            static_cast<NSUInteger>(sceneIndices.size() * sizeof(simd::uint3));
        if (!m_meshIndexBuffer || m_meshIndexBuffer.length < indexBytes) {
            m_meshIndexBuffer =
                [m_device newBufferWithLength:indexBytes options:MTLResourceStorageModeShared];
        }
        if (m_meshIndexBuffer) {
            memcpy([m_meshIndexBuffer contents], sceneIndices.data(), indexBytes);
        }
    } else {
        m_meshIndexBuffer = nil;
    }


    if (m_sceneAccel) {
        m_sceneAccel->rebuild(buildInput, m_intersectionProvider);
        m_primitiveCount = m_sceneAccel->primitiveCount() + m_rectangleCount;
        if (m_sceneAccel->mode() == PathTracerShaderTypes::IntersectionMode::HardwareRayTracing) {
            m_primitiveCount += m_sphereCount;
        }
    } else {
        m_intersectionProvider = IntersectionProvider{};
        m_primitiveCount = m_rectangleCount;
    }

    m_triangleCount = triangleOffset;
    
    m_dirty = false;

    NSString* modeString = (m_intersectionProvider.mode ==
                            PathTracerShaderTypes::IntersectionMode::HardwareRayTracing)
                               ? @"HardwareRT"
                               : @"SoftwareBVH";
    NSLog(@"SceneResources::rebuildAccelerationStructures - mode=%@ meshes=%zu triangles=%u spheres=%u primitives=%u",
          modeString,
          m_meshes.size(),
          m_triangleCount,
          m_sphereCount,
          m_primitiveCount);
}

void SceneResources::uploadMeshes() {
    if (!m_device) {
        return;
    }

    for (auto& mesh : m_meshes) {
        const NSUInteger vertexBytes =
            static_cast<NSUInteger>(mesh.vertices.size()) * sizeof(MeshVertex);
        if (vertexBytes > 0) {
            if (!mesh.vertexBuffer || mesh.vertexBuffer.length < vertexBytes) {
                mesh.vertexBuffer = [m_device newBufferWithLength:vertexBytes
                                                          options:MTLResourceStorageModeShared];
            }
            if (mesh.vertexBuffer && !mesh.vertices.empty()) {
                memcpy([mesh.vertexBuffer contents], mesh.vertices.data(), vertexBytes);
            }
        } else {
            mesh.vertexBuffer = nil;
        }

        const NSUInteger indexBytes =
            static_cast<NSUInteger>(mesh.indices.size()) * sizeof(uint32_t);
        if (indexBytes > 0) {
            if (!mesh.indexBuffer || mesh.indexBuffer.length < indexBytes) {
                mesh.indexBuffer = [m_device newBufferWithLength:indexBytes
                                                         options:MTLResourceStorageModeShared];
            }
            if (mesh.indexBuffer && !mesh.indices.empty()) {
                memcpy([mesh.indexBuffer contents], mesh.indices.data(), indexBytes);
            }
        } else {
            mesh.indexBuffer = nil;
        }
    }
}

void SceneResources::uploadRectangles() {
    if (!m_device) {
        return;
    }

    const NSUInteger rectBytes =
        static_cast<NSUInteger>(m_rectangleCount) * sizeof(RectData);
    if (rectBytes > 0) {
        if (!m_rectangleBuffer || m_rectangleBuffer.length < rectBytes) {
            m_rectangleBuffer =
                [m_device newBufferWithLength:rectBytes options:MTLResourceStorageModeShared];
        }
        if (m_rectangleBuffer) {
            memcpy([m_rectangleBuffer contents], m_rectangles.data(), rectBytes);
        }
    } else {
        m_rectangleBuffer = nil;
    }
}

void SceneResources::storeRectangleOriented(const simd::float3& corner,
                                            const simd::float3& edgeU,
                                            const simd::float3& edgeV,
                                            bool twoSided,
                                            uint32_t materialIndex,
                                            const simd::float3& desiredNormal) {
    if (m_rectangleCount >= kMaxRectangles) {
        return;
    }

    float uLenSq = simd::dot(edgeU, edgeU);
    float vLenSq = simd::dot(edgeV, edgeV);
    if (uLenSq <= std::numeric_limits<float>::min() ||
        vLenSq <= std::numeric_limits<float>::min()) {
        return;
    }

    simd::float3 normal = simd::cross(edgeU, edgeV);
    float normalLenSq = simd::dot(normal, normal);
    if (normalLenSq <= std::numeric_limits<float>::min()) {
        return;
    }
    simd::float3 unitNormal = normal / std::sqrt(normalLenSq);

    float desiredLenSq = simd::dot(desiredNormal, desiredNormal);
    simd::float3 targetNormal = unitNormal;
    if (desiredLenSq > std::numeric_limits<float>::min()) {
        targetNormal = desiredNormal / std::sqrt(desiredLenSq);
    }
    if (simd::dot(unitNormal, targetNormal) < 0.0f) {
        unitNormal = -unitNormal;
    }

    if (!std::isfinite(unitNormal.x) || !std::isfinite(unitNormal.y) || !std::isfinite(unitNormal.z)) {
        return;
    }

    float planeConstant = simd::dot(unitNormal, corner);

    RectData rect{};
    rect.corner = simd_make_float4(corner, 0.0f);
    rect.edgeU = simd_make_float4(edgeU, (uLenSq > 0.0f) ? (1.0f / uLenSq) : 0.0f);
    rect.edgeV = simd_make_float4(edgeV, (vLenSq > 0.0f) ? (1.0f / vLenSq) : 0.0f);
    rect.normalAndPlane = simd_make_float4(unitNormal, planeConstant);
    rect.materialTwoSided = simd_make_uint4(materialIndex, twoSided ? 1u : 0u, 0u, 0u);

    m_rectangles[m_rectangleCount++] = rect;
    m_dirty = true;
}

}  // namespace PathTracer
