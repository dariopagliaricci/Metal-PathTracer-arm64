#import "renderer/SceneResources.h"
#import "renderer/MetalContext.h"
#import "renderer/SceneAccel.h"
#import "renderer/EnvImportanceSampler.h"
#import "renderer/ImageWriter.h"

#import <MetalKit/MetalKit.h>
#import <CoreFoundation/CoreFoundation.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <limits>
#include <random>
#include <vector>

using PathTracerShaderTypes::EnvironmentAliasEntry;
using PathTracerShaderTypes::MaterialData;
using PathTracerShaderTypes::MaterialTextureInfo;
using PathTracerShaderTypes::MaterialType;
using PathTracerShaderTypes::RectData;
using PathTracerShaderTypes::SphereData;
using PathTracerShaderTypes::kMaxMaterials;
using PathTracerShaderTypes::kMaxMaterialSamplers;
using PathTracerShaderTypes::kMaxMaterialTextures;
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

constexpr uint32_t kInvalidTextureIndex = 0xFFFFFFFFu;
constexpr int32_t kGltfFilterNearest = 9728;
constexpr int32_t kGltfFilterLinear = 9729;
constexpr int32_t kGltfFilterNearestMipmapNearest = 9984;
constexpr int32_t kGltfFilterLinearMipmapNearest = 9985;
constexpr int32_t kGltfFilterNearestMipmapLinear = 9986;
constexpr int32_t kGltfFilterLinearMipmapLinear = 9987;
constexpr int32_t kGltfWrapClampToEdge = 33071;
constexpr int32_t kGltfWrapMirroredRepeat = 33648;
constexpr int32_t kGltfWrapRepeat = 10497;
constexpr NSUInteger kDefaultMaterialAnisotropy = 8u;

inline simd::float4 MakeTextureTransformRow0Identity() {
    return simd_make_float4(1.0f, 0.0f, 0.0f, 0.0f);
}

inline simd::float4 MakeTextureTransformRow1Identity() {
    return simd_make_float4(0.0f, 1.0f, 0.0f, 0.0f);
}

inline bool TextureTransformRowsAreUsable(const simd::float4& row0, const simd::float4& row1) {
    if (!std::isfinite(row0.x) || !std::isfinite(row0.y) || !std::isfinite(row0.z) ||
        !std::isfinite(row1.x) || !std::isfinite(row1.y) || !std::isfinite(row1.z)) {
        return false;
    }
    const float linearSum = std::fabs(row0.x) + std::fabs(row0.y) + std::fabs(row1.x) + std::fabs(row1.y);
    return linearSum > 1.0e-8f;
}

inline void SetMaterialTextureTransformIdentity(MaterialData& material) {
    material.textureUvSet0 = simd_make_uint4(0u, 0u, 0u, 0u);
    material.textureUvSet1 = simd_make_uint4(0u, 0u, 0u, 0u);
    material.textureTransform0 = MakeTextureTransformRow0Identity();
    material.textureTransform1 = MakeTextureTransformRow1Identity();
    material.textureTransform2 = MakeTextureTransformRow0Identity();
    material.textureTransform3 = MakeTextureTransformRow1Identity();
    material.textureTransform4 = MakeTextureTransformRow0Identity();
    material.textureTransform5 = MakeTextureTransformRow1Identity();
    material.textureTransform6 = MakeTextureTransformRow0Identity();
    material.textureTransform7 = MakeTextureTransformRow1Identity();
    material.textureTransform8 = MakeTextureTransformRow0Identity();
    material.textureTransform9 = MakeTextureTransformRow1Identity();
    material.textureTransform10 = MakeTextureTransformRow0Identity();
    material.textureTransform11 = MakeTextureTransformRow1Identity();
}

inline void SanitizeMaterialTextureMapping(MaterialData& material) {
    material.textureUvSet0.x = std::min<uint32_t>(material.textureUvSet0.x, 1u);
    material.textureUvSet0.y = std::min<uint32_t>(material.textureUvSet0.y, 1u);
    material.textureUvSet0.z = std::min<uint32_t>(material.textureUvSet0.z, 1u);
    material.textureUvSet0.w = std::min<uint32_t>(material.textureUvSet0.w, 1u);
    material.textureUvSet1.x = std::min<uint32_t>(material.textureUvSet1.x, 1u);
    material.textureUvSet1.y = std::min<uint32_t>(material.textureUvSet1.y, 1u);

    if (!TextureTransformRowsAreUsable(material.textureTransform0, material.textureTransform1)) {
        material.textureTransform0 = MakeTextureTransformRow0Identity();
        material.textureTransform1 = MakeTextureTransformRow1Identity();
    }
    if (!TextureTransformRowsAreUsable(material.textureTransform2, material.textureTransform3)) {
        material.textureTransform2 = MakeTextureTransformRow0Identity();
        material.textureTransform3 = MakeTextureTransformRow1Identity();
    }
    if (!TextureTransformRowsAreUsable(material.textureTransform4, material.textureTransform5)) {
        material.textureTransform4 = MakeTextureTransformRow0Identity();
        material.textureTransform5 = MakeTextureTransformRow1Identity();
    }
    if (!TextureTransformRowsAreUsable(material.textureTransform6, material.textureTransform7)) {
        material.textureTransform6 = MakeTextureTransformRow0Identity();
        material.textureTransform7 = MakeTextureTransformRow1Identity();
    }
    if (!TextureTransformRowsAreUsable(material.textureTransform8, material.textureTransform9)) {
        material.textureTransform8 = MakeTextureTransformRow0Identity();
        material.textureTransform9 = MakeTextureTransformRow1Identity();
    }
    if (!TextureTransformRowsAreUsable(material.textureTransform10, material.textureTransform11)) {
        material.textureTransform10 = MakeTextureTransformRow0Identity();
        material.textureTransform11 = MakeTextureTransformRow1Identity();
    }
}

MaterialTextureSamplerDesc EffectiveSamplerDesc(const MaterialTextureSamplerDesc* desc) {
    MaterialTextureSamplerDesc effective{};
    if (desc) {
        effective = *desc;
    }
    if (effective.magFilter < 0) {
        effective.magFilter = kGltfFilterLinear;
    }
    if (effective.minFilter < 0) {
        effective.minFilter = kGltfFilterLinearMipmapLinear;
    }
    if (effective.wrapS != kGltfWrapClampToEdge &&
        effective.wrapS != kGltfWrapMirroredRepeat &&
        effective.wrapS != kGltfWrapRepeat) {
        effective.wrapS = kGltfWrapRepeat;
    }
    if (effective.wrapT != kGltfWrapClampToEdge &&
        effective.wrapT != kGltfWrapMirroredRepeat &&
        effective.wrapT != kGltfWrapRepeat) {
        effective.wrapT = kGltfWrapRepeat;
    }
    return effective;
}

uint64_t SamplerDescKey(const MaterialTextureSamplerDesc& desc) {
    uint64_t key = 0u;
    key |= static_cast<uint64_t>(static_cast<uint16_t>(desc.magFilter));
    key |= static_cast<uint64_t>(static_cast<uint16_t>(desc.minFilter)) << 16u;
    key |= static_cast<uint64_t>(static_cast<uint16_t>(desc.wrapS)) << 32u;
    key |= static_cast<uint64_t>(static_cast<uint16_t>(desc.wrapT)) << 48u;
    return key;
}

std::string SamplerKeySuffix(const MaterialTextureSamplerDesc* desc) {
    MaterialTextureSamplerDesc effective = EffectiveSamplerDesc(desc);
    return "|samp:" + std::to_string(effective.magFilter) + ":" +
           std::to_string(effective.minFilter) + ":" +
           std::to_string(effective.wrapS) + ":" +
           std::to_string(effective.wrapT);
}

std::string TextureSemanticKeySuffix(MaterialTextureSemantic semantic) {
    return "|sem:" + std::to_string(static_cast<uint32_t>(semantic));
}

MTLSamplerAddressMode AddressModeFromGltfWrap(int32_t wrap) {
    switch (wrap) {
        case kGltfWrapClampToEdge:
            return MTLSamplerAddressModeClampToEdge;
        case kGltfWrapMirroredRepeat:
            return MTLSamplerAddressModeMirrorRepeat;
        case kGltfWrapRepeat:
        default:
            return MTLSamplerAddressModeRepeat;
    }
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

bool IsSrgbPixelFormat(MTLPixelFormat format) {
    return format == MTLPixelFormatRGBA8Unorm_sRGB ||
           format == MTLPixelFormatBGRA8Unorm_sRGB;
}

MTLPixelFormat LinearEquivalentPixelFormat(MTLPixelFormat format) {
    switch (format) {
        case MTLPixelFormatRGBA8Unorm_sRGB:
            return MTLPixelFormatRGBA8Unorm;
        case MTLPixelFormatBGRA8Unorm_sRGB:
            return MTLPixelFormatRGBA8Unorm;
        case MTLPixelFormatBGRA8Unorm:
            return MTLPixelFormatRGBA8Unorm;
        default:
            return format;
    }
}

NSUInteger MipDimension(NSUInteger base, NSUInteger level) {
    return std::max<NSUInteger>(1u, base >> level);
}

bool CopyTextureLevelToFloat(id<MTLTexture> texture,
                             NSUInteger level,
                             bool decodeSrgb,
                             std::vector<float>& outFloats) {
    if (!texture) {
        return false;
    }

    const NSUInteger width = MipDimension(texture.width, level);
    const NSUInteger height = MipDimension(texture.height, level);
    if (width == 0 || height == 0) {
        return false;
    }

    const MTLPixelFormat format = texture.pixelFormat;
    const NSUInteger bytesPerPixel = BytesPerPixel(format);
    if (bytesPerPixel == 0) {
        return false;
    }

    const NSUInteger pixelCount = width * height;
    const NSUInteger bytesPerRow = width * bytesPerPixel;
    std::vector<uint8_t> rawData(static_cast<size_t>(bytesPerRow) * height);

    MTLRegion region = MTLRegionMake2D(0, 0, width, height);
    [texture getBytes:rawData.data()
          bytesPerRow:bytesPerRow
           fromRegion:region
          mipmapLevel:level];

    outFloats.assign(pixelCount * 4u, 0.0f);

    switch (format) {
        case MTLPixelFormatRGBA32Float: {
            const float* src = reinterpret_cast<const float*>(rawData.data());
            std::copy(src, src + outFloats.size(), outFloats.begin());
            return true;
        }
        case MTLPixelFormatRGBA16Float: {
            const __fp16* src = reinterpret_cast<const __fp16*>(rawData.data());
            for (size_t i = 0; i < outFloats.size(); ++i) {
                outFloats[i] = static_cast<float>(src[i]);
            }
            return true;
        }
        case MTLPixelFormatRGBA8Unorm:
        case MTLPixelFormatBGRA8Unorm:
        case MTLPixelFormatRGBA8Unorm_sRGB:
        case MTLPixelFormatBGRA8Unorm_sRGB: {
            const bool isBgra = (format == MTLPixelFormatBGRA8Unorm ||
                                 format == MTLPixelFormatBGRA8Unorm_sRGB);
            const bool isSrgbFormat = (format == MTLPixelFormatRGBA8Unorm_sRGB ||
                                       format == MTLPixelFormatBGRA8Unorm_sRGB);
            const uint8_t* src = rawData.data();
            for (NSUInteger i = 0; i < pixelCount; ++i) {
                uint8_t r8 = src[i * 4u + 0u];
                uint8_t g8 = src[i * 4u + 1u];
                uint8_t b8 = src[i * 4u + 2u];
                uint8_t a8 = src[i * 4u + 3u];
                if (isBgra) {
                    std::swap(r8, b8);
                }

                float r = static_cast<float>(r8) / 255.0f;
                float g = static_cast<float>(g8) / 255.0f;
                float b = static_cast<float>(b8) / 255.0f;
                if (decodeSrgb && isSrgbFormat) {
                    r = SrgbToLinear(r);
                    g = SrgbToLinear(g);
                    b = SrgbToLinear(b);
                }

                outFloats[i * 4u + 0u] = r;
                outFloats[i * 4u + 1u] = g;
                outFloats[i * 4u + 2u] = b;
                outFloats[i * 4u + 3u] = static_cast<float>(a8) / 255.0f;
            }
            return true;
        }
        default:
            return false;
    }
}

bool WriteTextureLevelFromFloat(id<MTLTexture> texture,
                                NSUInteger level,
                                const std::vector<float>& inFloats) {
    if (!texture) {
        return false;
    }

    const NSUInteger width = MipDimension(texture.width, level);
    const NSUInteger height = MipDimension(texture.height, level);
    if (width == 0 || height == 0) {
        return false;
    }

    const NSUInteger pixelCount = width * height;
    if (inFloats.size() != static_cast<size_t>(pixelCount) * 4u) {
        return false;
    }

    const MTLPixelFormat format = texture.pixelFormat;
    const NSUInteger bytesPerPixel = BytesPerPixel(format);
    if (bytesPerPixel == 0) {
        return false;
    }

    const NSUInteger bytesPerRow = width * bytesPerPixel;
    std::vector<uint8_t> rawData(static_cast<size_t>(bytesPerRow) * height, 0u);

    switch (format) {
        case MTLPixelFormatRGBA32Float: {
            float* dst = reinterpret_cast<float*>(rawData.data());
            std::copy(inFloats.begin(), inFloats.end(), dst);
            break;
        }
        case MTLPixelFormatRGBA16Float: {
            __fp16* dst = reinterpret_cast<__fp16*>(rawData.data());
            for (size_t i = 0; i < inFloats.size(); ++i) {
                float value = std::isfinite(inFloats[i]) ? inFloats[i] : 0.0f;
                dst[i] = static_cast<__fp16>(value);
            }
            break;
        }
        case MTLPixelFormatRGBA8Unorm:
        case MTLPixelFormatBGRA8Unorm:
        case MTLPixelFormatRGBA8Unorm_sRGB:
        case MTLPixelFormatBGRA8Unorm_sRGB: {
            const bool isBgra = (format == MTLPixelFormatBGRA8Unorm ||
                                 format == MTLPixelFormatBGRA8Unorm_sRGB);
            uint8_t* dst = rawData.data();
            for (NSUInteger i = 0; i < pixelCount; ++i) {
                float r = std::isfinite(inFloats[i * 4u + 0u]) ? inFloats[i * 4u + 0u] : 0.0f;
                float g = std::isfinite(inFloats[i * 4u + 1u]) ? inFloats[i * 4u + 1u] : 0.0f;
                float b = std::isfinite(inFloats[i * 4u + 2u]) ? inFloats[i * 4u + 2u] : 0.0f;
                float a = std::isfinite(inFloats[i * 4u + 3u]) ? inFloats[i * 4u + 3u] : 1.0f;
                uint8_t r8 = static_cast<uint8_t>(std::lround(std::clamp(r, 0.0f, 1.0f) * 255.0f));
                uint8_t g8 = static_cast<uint8_t>(std::lround(std::clamp(g, 0.0f, 1.0f) * 255.0f));
                uint8_t b8 = static_cast<uint8_t>(std::lround(std::clamp(b, 0.0f, 1.0f) * 255.0f));
                uint8_t a8 = static_cast<uint8_t>(std::lround(std::clamp(a, 0.0f, 1.0f) * 255.0f));
                if (isBgra) {
                    std::swap(r8, b8);
                }
                dst[i * 4u + 0u] = r8;
                dst[i * 4u + 1u] = g8;
                dst[i * 4u + 2u] = b8;
                dst[i * 4u + 3u] = a8;
            }
            break;
        }
        default:
            return false;
    }

    MTLRegion region = MTLRegionMake2D(0, 0, width, height);
    [texture replaceRegion:region
               mipmapLevel:level
                 withBytes:rawData.data()
               bytesPerRow:bytesPerRow];
    return true;
}

void DownsampleOrmMipLevel(const std::vector<float>& src,
                           NSUInteger srcWidth,
                           NSUInteger srcHeight,
                           std::vector<float>& dst,
                           NSUInteger dstWidth,
                           NSUInteger dstHeight,
                           int32_t wrapS,
                           int32_t wrapT) {
    dst.assign(static_cast<size_t>(dstWidth * dstHeight) * 4u, 0.0f);
    for (NSUInteger y = 0; y < dstHeight; ++y) {
        for (NSUInteger x = 0; x < dstWidth; ++x) {
            float sumAo = 0.0f;
            float sumAlpha = 0.0f;
            float sumMetal = 0.0f;
            float sumA = 0.0f;
            for (NSUInteger ky = 0; ky < 2; ++ky) {
                for (NSUInteger kx = 0; kx < 2; ++kx) {
                    auto wrap_coord = [](NSInteger coord, NSUInteger extent, int32_t mode) -> NSUInteger {
                        if (extent == 0u) {
                            return 0u;
                        }
                        const NSInteger maxIndex = static_cast<NSInteger>(extent) - 1;
                        switch (mode) {
                            case kGltfWrapRepeat: {
                                const NSInteger period = static_cast<NSInteger>(extent);
                                NSInteger wrapped = coord % period;
                                if (wrapped < 0) {
                                    wrapped += period;
                                }
                                return static_cast<NSUInteger>(wrapped);
                            }
                            case kGltfWrapMirroredRepeat: {
                                const NSInteger period = static_cast<NSInteger>(extent) * 2;
                                NSInteger wrapped = coord % period;
                                if (wrapped < 0) {
                                    wrapped += period;
                                }
                                if (wrapped >= static_cast<NSInteger>(extent)) {
                                    wrapped = period - wrapped - 1;
                                }
                                return static_cast<NSUInteger>(wrapped);
                            }
                            case kGltfWrapClampToEdge:
                            default:
                                if (coord <= 0) {
                                    return 0u;
                                }
                                if (coord >= maxIndex) {
                                    return static_cast<NSUInteger>(maxIndex);
                                }
                                return static_cast<NSUInteger>(coord);
                        }
                    };
                    NSInteger sxCoord = static_cast<NSInteger>(x * 2u + kx);
                    NSInteger syCoord = static_cast<NSInteger>(y * 2u + ky);
                    NSUInteger sx = wrap_coord(sxCoord, srcWidth, wrapS);
                    NSUInteger sy = wrap_coord(syCoord, srcHeight, wrapT);
                    size_t srcIndex = static_cast<size_t>((sy * srcWidth + sx) * 4u);
                    float ao = std::isfinite(src[srcIndex + 0u]) ? src[srcIndex + 0u] : 1.0f;
                    float rough = std::isfinite(src[srcIndex + 1u]) ? src[srcIndex + 1u] : 1.0f;
                    float metal = std::isfinite(src[srcIndex + 2u]) ? src[srcIndex + 2u] : 0.0f;
                    float alpha = std::isfinite(src[srcIndex + 3u]) ? src[srcIndex + 3u] : 1.0f;
                    rough = std::clamp(rough, 0.0f, 1.0f);
                    sumAo += ao;
                    sumAlpha += rough * rough;
                    sumMetal += metal;
                    sumA += alpha;
                }
            }

            const float invSampleCount = 0.25f;
            size_t dstIndex = static_cast<size_t>((y * dstWidth + x) * 4u);
            dst[dstIndex + 0u] = sumAo * invSampleCount;
            dst[dstIndex + 1u] = std::sqrt(std::max(sumAlpha * invSampleCount, 0.0f));
            dst[dstIndex + 2u] = sumMetal * invSampleCount;
            dst[dstIndex + 3u] = sumA * invSampleCount;
        }
    }
}

void PrefilterOrmBaseLevel(const std::vector<float>& src,
                           NSUInteger width,
                           NSUInteger height,
                           std::vector<float>& dst,
                           int32_t wrapS,
                           int32_t wrapT) {
    dst.assign(src.size(), 0.0f);
    if (width == 0 || height == 0) {
        return;
    }

    for (NSUInteger y = 0; y < height; ++y) {
        for (NSUInteger x = 0; x < width; ++x) {
            float sumAo = 0.0f;
            float sumAlpha = 0.0f;
            float sumMetal = 0.0f;
            float sumA = 0.0f;
            float totalW = 0.0f;
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    auto wrap_coord = [](NSInteger coord, NSUInteger extent, int32_t mode) -> NSUInteger {
                        if (extent == 0u) {
                            return 0u;
                        }
                        const NSInteger maxIndex = static_cast<NSInteger>(extent) - 1;
                        switch (mode) {
                            case kGltfWrapRepeat: {
                                const NSInteger period = static_cast<NSInteger>(extent);
                                NSInteger wrapped = coord % period;
                                if (wrapped < 0) {
                                    wrapped += period;
                                }
                                return static_cast<NSUInteger>(wrapped);
                            }
                            case kGltfWrapMirroredRepeat: {
                                const NSInteger period = static_cast<NSInteger>(extent) * 2;
                                NSInteger wrapped = coord % period;
                                if (wrapped < 0) {
                                    wrapped += period;
                                }
                                if (wrapped >= static_cast<NSInteger>(extent)) {
                                    wrapped = period - wrapped - 1;
                                }
                                return static_cast<NSUInteger>(wrapped);
                            }
                            case kGltfWrapClampToEdge:
                            default:
                                if (coord <= 0) {
                                    return 0u;
                                }
                                if (coord >= maxIndex) {
                                    return static_cast<NSUInteger>(maxIndex);
                                }
                                return static_cast<NSUInteger>(coord);
                        }
                    };
                    const NSUInteger sx = wrap_coord(static_cast<NSInteger>(x) + kx, width, wrapS);
                    const NSUInteger sy = wrap_coord(static_cast<NSInteger>(y) + ky, height, wrapT);
                    const float weight = (kx == 0 && ky == 0) ? 4.0f : ((kx == 0 || ky == 0) ? 2.0f : 1.0f);
                    const size_t srcIndex = static_cast<size_t>((sy * width + sx) * 4u);
                    const float ao = std::isfinite(src[srcIndex + 0u]) ? src[srcIndex + 0u] : 1.0f;
                    const float rough = std::isfinite(src[srcIndex + 1u]) ? src[srcIndex + 1u] : 1.0f;
                    const float metal = std::isfinite(src[srcIndex + 2u]) ? src[srcIndex + 2u] : 0.0f;
                    const float alpha = std::isfinite(src[srcIndex + 3u]) ? src[srcIndex + 3u] : 1.0f;
                    const float rClamped = std::clamp(rough, 0.0f, 1.0f);
                    sumAo += weight * ao;
                    sumAlpha += weight * (rClamped * rClamped);
                    sumMetal += weight * metal;
                    sumA += weight * alpha;
                    totalW += weight;
                }
            }

            const float invW = (totalW > 0.0f) ? (1.0f / totalW) : 0.0f;
            const size_t dstIndex = static_cast<size_t>((y * width + x) * 4u);
            dst[dstIndex + 0u] = sumAo * invW;
            dst[dstIndex + 1u] = std::sqrt(std::max(sumAlpha * invW, 0.0f));
            dst[dstIndex + 2u] = sumMetal * invW;
            dst[dstIndex + 3u] = sumA * invW;
        }
    }
}

id<MTLTexture> BuildTextureWithBlitMips(id<MTLDevice> device,
                                        id<MTLCommandQueue> commandQueue,
                                        id<MTLTexture> source) {
    if (!device || !commandQueue || !source) {
        return nil;
    }
    MTLTextureDescriptor* desc =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:source.pixelFormat
                                                           width:source.width
                                                          height:source.height
                                                       mipmapped:YES];
    desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
    desc.storageMode = MTLStorageModeShared;
    id<MTLTexture> mipTexture = [device newTextureWithDescriptor:desc];
    if (!mipTexture) {
        return nil;
    }
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
    if (!blit) {
        return nil;
    }
    MTLOrigin origin = MTLOriginMake(0, 0, 0);
    MTLSize size = MTLSizeMake(source.width, source.height, 1);
    [blit copyFromTexture:source
             sourceSlice:0
             sourceLevel:0
            sourceOrigin:origin
              sourceSize:size
               toTexture:mipTexture
        destinationSlice:0
        destinationLevel:0
       destinationOrigin:origin];
    [blit generateMipmapsForTexture:mipTexture];
    [blit endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    return mipTexture;
}

id<MTLTexture> CloneTextureWithFormat(id<MTLDevice> device,
                                      id<MTLTexture> source,
                                      MTLPixelFormat pixelFormat) {
    if (!device || !source) {
        return nil;
    }
    MTLTextureDescriptor* desc =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:pixelFormat
                                                           width:source.width
                                                          height:source.height
                                                       mipmapped:(source.mipmapLevelCount > 1u)];
    desc.usage = MTLTextureUsageShaderRead;
    desc.storageMode = MTLStorageModeShared;
    id<MTLTexture> outTexture = [device newTextureWithDescriptor:desc];
    if (!outTexture) {
        return nil;
    }

    for (NSUInteger level = 0u; level < source.mipmapLevelCount; ++level) {
        std::vector<float> levelData;
        if (!CopyTextureLevelToFloat(source, level, /*decodeSrgb=*/false, levelData)) {
            return nil;
        }
        if (!WriteTextureLevelFromFloat(outTexture, level, levelData)) {
            return nil;
        }
    }
    return outTexture;
}

id<MTLTexture> BuildTextureWithOrmMips(id<MTLDevice> device,
                                       id<MTLTexture> source,
                                       const MaterialTextureSamplerDesc* samplerDesc) {
    if (!device || !source || source.width == 0 || source.height == 0) {
        return nil;
    }

    if (BytesPerPixel(source.pixelFormat) == 0) {
        return nil;
    }

    MTLTextureDescriptor* desc =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:source.pixelFormat
                                                           width:source.width
                                                          height:source.height
                                                       mipmapped:YES];
    desc.usage = MTLTextureUsageShaderRead;
    desc.storageMode = MTLStorageModeShared;
    id<MTLTexture> mipTexture = [device newTextureWithDescriptor:desc];
    if (!mipTexture) {
        return nil;
    }

    MaterialTextureSamplerDesc effectiveSampler = EffectiveSamplerDesc(samplerDesc);
    std::vector<float> prevLevel;
    if (!CopyTextureLevelToFloat(source, 0u, /*decodeSrgb=*/false, prevLevel)) {
        return nil;
    }
    if (!WriteTextureLevelFromFloat(mipTexture, 0u, prevLevel)) {
        return nil;
    }

    NSUInteger prevWidth = source.width;
    NSUInteger prevHeight = source.height;
    for (NSUInteger level = 1u; level < mipTexture.mipmapLevelCount; ++level) {
        NSUInteger dstWidth = MipDimension(source.width, level);
        NSUInteger dstHeight = MipDimension(source.height, level);
        std::vector<float> nextLevel;
        DownsampleOrmMipLevel(prevLevel,
                              prevWidth,
                              prevHeight,
                              nextLevel,
                              dstWidth,
                              dstHeight,
                              effectiveSampler.wrapS,
                              effectiveSampler.wrapT);
        if (!WriteTextureLevelFromFloat(mipTexture, level, nextLevel)) {
            return nil;
        }
        prevLevel.swap(nextLevel);
        prevWidth = dstWidth;
        prevHeight = dstHeight;
    }
#if !defined(NDEBUG)
    static bool sDumpedOrmMipChain = false;
    if (!sDumpedOrmMipChain) {
        sDumpedOrmMipChain = true;
        std::error_code ec;
        namespace fs = std::filesystem;
        fs::path dumpDir = fs::path("renders") / "debug" / "orm_mips";
        fs::create_directories(dumpDir, ec);
        for (NSUInteger level = 0u; level < mipTexture.mipmapLevelCount; ++level) {
            std::vector<float> levelData;
            if (!CopyTextureLevelToFloat(mipTexture, level, /*decodeSrgb=*/false, levelData)) {
                continue;
            }
            fs::path outPath = dumpDir / ("orm_mip_" + std::to_string(level) + ".exr");
            bool wrote = ImageWriter::WriteEXR(outPath.string().c_str(),
                                               levelData.data(),
                                               static_cast<int>(MipDimension(mipTexture.width, level)),
                                               static_cast<int>(MipDimension(mipTexture.height, level)),
                                               "Linear");
            if (!wrote) {
                NSLog(@"[glTF] Failed to dump ORM mip %lu", static_cast<unsigned long>(level));
            }
        }
        NSLog(@"[glTF] Dumped ORM mip chain to %s", dumpDir.string().c_str());
    }
#endif
    return mipTexture;
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
    m_materialTextures.clear();
    m_materialSamplers.clear();
    m_materialTextureSamplerIndices.clear();
    m_materialTextureLabels.clear();
    m_materialTextureIndex.clear();
    m_materialSamplerIndices.clear();
    m_materialTextureInfoBuffer = nil;
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
                                     bool thinDielectric,
                                     std::string name) {
    if (m_materialCount >= kMaxMaterials) {
        return static_cast<uint32_t>(kMaxMaterials - 1);
    }

    uint32_t index = m_materialCount++;
    MaterialData material{};
    SetMaterialTextureTransformIdentity(material);

    simd::float3 clampedBaseColor = ClampColor01(baseColor);
    float clampedRoughness = std::clamp(roughness, 0.0f, 1.0f);
    material.baseColorRoughness = simd_make_float4(clampedBaseColor, clampedRoughness);

    float clampedIor = std::max(indexOfRefraction, 0.0f);
    float clampedCoatIor = std::max(coatIor, 0.0f);
    material.typeEta = simd_make_float4(static_cast<float>(type),
                                        clampedIor,
                                        clampedCoatIor,
                                        thinDielectric ? 1.0f : 0.0f);

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
    material.textureIndices0 =
        simd_make_uint4(kInvalidTextureIndex, kInvalidTextureIndex, kInvalidTextureIndex, kInvalidTextureIndex);
    material.textureIndices1 =
        simd_make_uint4(kInvalidTextureIndex, kInvalidTextureIndex, kInvalidTextureIndex, kInvalidTextureIndex);
    material.pbrParams = simd_make_float4(0.0f, clampedRoughness, 1.0f, 1.0f);
    SanitizeMaterialTextureMapping(material);

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
                       /*thinDielectric=*/false,
                       std::move(name));
}

uint32_t SceneResources::addMaterialData(const MaterialData& material,
                                         std::string name) {
    if (m_materialCount >= kMaxMaterials) {
        return static_cast<uint32_t>(kMaxMaterials - 1);
    }

    uint32_t index = m_materialCount++;
    MaterialData sanitized = material;
    SanitizeMaterialTextureMapping(sanitized);
    m_materials[index] = sanitized;
    m_materialDefaults[index] = sanitized;
    m_materialNames[index] = name.empty() ? DefaultMaterialName(index) : std::move(name);
    m_dirty = true;
    return index;
}

uint32_t SceneResources::materialSamplerIndexForDesc(
    const MaterialTextureSamplerDesc* samplerDesc) {
    if (!m_device) {
        return 0u;
    }

    MaterialTextureSamplerDesc effective = EffectiveSamplerDesc(samplerDesc);
    uint64_t key = SamplerDescKey(effective);
    auto cached = m_materialSamplerIndices.find(key);
    if (cached != m_materialSamplerIndices.end()) {
        return cached->second;
    }

    MTLSamplerMinMagFilter minFilter = MTLSamplerMinMagFilterLinear;
    MTLSamplerMinMagFilter magFilter = MTLSamplerMinMagFilterLinear;
    MTLSamplerMipFilter mipFilter = MTLSamplerMipFilterLinear;
    switch (effective.minFilter) {
        case kGltfFilterNearest:
            minFilter = MTLSamplerMinMagFilterNearest;
            mipFilter = MTLSamplerMipFilterNotMipmapped;
            break;
        case kGltfFilterLinear:
            minFilter = MTLSamplerMinMagFilterLinear;
            mipFilter = MTLSamplerMipFilterNotMipmapped;
            break;
        case kGltfFilterNearestMipmapNearest:
            minFilter = MTLSamplerMinMagFilterNearest;
            mipFilter = MTLSamplerMipFilterNearest;
            break;
        case kGltfFilterLinearMipmapNearest:
            minFilter = MTLSamplerMinMagFilterLinear;
            mipFilter = MTLSamplerMipFilterNearest;
            break;
        case kGltfFilterNearestMipmapLinear:
            minFilter = MTLSamplerMinMagFilterNearest;
            mipFilter = MTLSamplerMipFilterLinear;
            break;
        case kGltfFilterLinearMipmapLinear:
        default:
            minFilter = MTLSamplerMinMagFilterLinear;
            mipFilter = MTLSamplerMipFilterLinear;
            break;
    }

    switch (effective.magFilter) {
        case kGltfFilterNearest:
            magFilter = MTLSamplerMinMagFilterNearest;
            break;
        case kGltfFilterLinear:
        default:
            magFilter = MTLSamplerMinMagFilterLinear;
            break;
    }

    MTLSamplerDescriptor* descriptor = [[MTLSamplerDescriptor alloc] init];
    descriptor.minFilter = minFilter;
    descriptor.magFilter = magFilter;
    descriptor.mipFilter = mipFilter;
    descriptor.sAddressMode = AddressModeFromGltfWrap(effective.wrapS);
    descriptor.tAddressMode = AddressModeFromGltfWrap(effective.wrapT);
    descriptor.normalizedCoordinates = YES;
    descriptor.lodMinClamp = 0.0f;
    descriptor.lodMaxClamp = FLT_MAX;
    if (minFilter == MTLSamplerMinMagFilterLinear &&
        magFilter == MTLSamplerMinMagFilterLinear &&
        mipFilter != MTLSamplerMipFilterNotMipmapped) {
        descriptor.maxAnisotropy = kDefaultMaterialAnisotropy;
    } else {
        descriptor.maxAnisotropy = 1;
    }

    MTLSamplerStateHandle sampler = [m_device newSamplerStateWithDescriptor:descriptor];
    if (!sampler || m_materialSamplers.size() >= kMaxMaterialSamplers) {
        MaterialTextureSamplerDesc fallback = EffectiveSamplerDesc(nullptr);
        uint64_t fallbackKey = SamplerDescKey(fallback);
        auto fallbackIt = m_materialSamplerIndices.find(fallbackKey);
        if (fallbackIt != m_materialSamplerIndices.end()) {
            return fallbackIt->second;
        }
        return 0u;
    }
    uint32_t samplerIndex = static_cast<uint32_t>(m_materialSamplers.size());
    m_materialSamplers.push_back(sampler);
    m_materialSamplerIndices.emplace(key, samplerIndex);
    return samplerIndex;
}

void SceneResources::rebuildMaterialTextureInfoBuffer() {
    if (!m_device || m_materialTextures.empty()) {
        m_materialTextureInfoBuffer = nil;
        return;
    }
    const NSUInteger infoCount = static_cast<NSUInteger>(m_materialTextures.size());
    const NSUInteger infoBytes = infoCount * sizeof(MaterialTextureInfo);
    if (!m_materialTextureInfoBuffer || m_materialTextureInfoBuffer.length < infoBytes) {
        m_materialTextureInfoBuffer = [m_device newBufferWithLength:infoBytes
                                                            options:MTLResourceStorageModeShared];
    }
    if (!m_materialTextureInfoBuffer) {
        return;
    }

    auto* infos = reinterpret_cast<MaterialTextureInfo*>([m_materialTextureInfoBuffer contents]);
    if (!infos) {
        return;
    }
    for (NSUInteger i = 0; i < infoCount; ++i) {
        MaterialTextureInfo info{};
        id<MTLTexture> texture = m_materialTextures[i];
        if (texture) {
            info.width = static_cast<uint32_t>(texture.width);
            info.height = static_cast<uint32_t>(texture.height);
            info.mipCount = static_cast<uint32_t>(texture.mipmapLevelCount);
            if (i < m_materialTextureSamplerIndices.size()) {
                info.flags = m_materialTextureSamplerIndices[i];
            }
        }
        infos[i] = info;
    }
}

uint32_t SceneResources::registerMaterialTexture(MTLTextureHandle texture,
                                                 const std::string& key,
                                                 const std::string& label,
                                                 const MaterialTextureSamplerDesc* samplerDesc,
                                                 MaterialTextureSemantic semantic) {
    if (!texture) {
        return kInvalidTextureIndex;
    }
    if (m_materialTextures.size() >= kMaxMaterialTextures) {
        return kInvalidTextureIndex;
    }
    auto found = m_materialTextureIndex.find(key);
    if (found != m_materialTextureIndex.end()) {
        return found->second;
    }
    if (m_device) {
        if (semantic == MaterialTextureSemantic::Orm) {
            id<MTLTexture> ormMipTexture = BuildTextureWithOrmMips(m_device, texture, samplerDesc);
            if (ormMipTexture) {
                texture = ormMipTexture;
            } else if (texture.mipmapLevelCount <= 1 && m_commandQueue) {
                id<MTLTexture> mipTexture = BuildTextureWithBlitMips(m_device, m_commandQueue, texture);
                if (mipTexture) {
                    texture = mipTexture;
                }
            }
        } else if (texture.mipmapLevelCount <= 1 && m_commandQueue) {
            id<MTLTexture> mipTexture = BuildTextureWithBlitMips(m_device, m_commandQueue, texture);
            if (mipTexture) {
                texture = mipTexture;
            }
        }
    }
    if (texture) {
        texture.label = [NSString stringWithFormat:@"Material Texture (%s)", label.c_str()];
    }
    if (m_materialSamplers.empty()) {
        (void)materialSamplerIndexForDesc(nullptr);
    }
    uint32_t samplerIndex = materialSamplerIndexForDesc(samplerDesc);
    uint32_t index = static_cast<uint32_t>(m_materialTextures.size());
    m_materialTextures.push_back(texture);
    m_materialTextureSamplerIndices.push_back(samplerIndex);
    m_materialTextureLabels.push_back(label);
    m_materialTextureIndex.emplace(key, index);
    rebuildMaterialTextureInfoBuffer();
    return index;
}

uint32_t SceneResources::addMaterialTextureFromFile(const std::string& path,
                                                    bool srgb,
                                                    std::string* errorMessage,
                                                    const MaterialTextureSamplerDesc* samplerDesc,
                                                    MaterialTextureSemantic semantic) {
    if (!m_device) {
        if (errorMessage) {
            *errorMessage = "Material texture load failed: Metal device not ready";
        }
        return kInvalidTextureIndex;
    }
    if (path.empty()) {
        if (errorMessage) {
            *errorMessage = "Material texture load failed: empty path";
        }
        return kInvalidTextureIndex;
    }

    const std::string canonicalPath = CanonicalizePath(path);
    const std::string key =
        canonicalPath + (srgb ? "|srgb" : "|linear") + SamplerKeySuffix(samplerDesc) +
        TextureSemanticKeySuffix(semantic);
    auto cached = m_materialTextureIndex.find(key);
    if (cached != m_materialTextureIndex.end()) {
        return cached->second;
    }

    NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:canonicalPath.c_str()]];
    if (!url) {
        if (errorMessage) {
            *errorMessage = "Material texture load failed: invalid path";
        }
        return kInvalidTextureIndex;
    }

    MTKTextureLoader* loader = [[MTKTextureLoader alloc] initWithDevice:m_device];
    NSDictionary* options = @{
        MTKTextureLoaderOptionSRGB : @(srgb),
        MTKTextureLoaderOptionTextureUsage : @(MTLTextureUsageShaderRead),
        MTKTextureLoaderOptionTextureStorageMode : @(MTLStorageModeShared),
        MTKTextureLoaderOptionGenerateMipmaps : @YES,
        MTKTextureLoaderOptionAllocateMipmaps : @YES
    };

    NSError* error = nil;
    id<MTLTexture> texture = [loader newTextureWithContentsOfURL:url options:options error:&error];
    if (!texture) {
        if (errorMessage) {
            NSString* errStr = error ? error.localizedDescription : @"unknown error";
            *errorMessage = std::string("Material texture load failed: ") + errStr.UTF8String;
        }
        return kInvalidTextureIndex;
    }
    if (!srgb) {
        MTLPixelFormat linearFormat = LinearEquivalentPixelFormat(texture.pixelFormat);
        if (linearFormat != texture.pixelFormat) {
            id<MTLTexture> linearTexture = CloneTextureWithFormat(m_device, texture, linearFormat);
            if (linearTexture) {
                texture = linearTexture;
            } else if (errorMessage) {
                *errorMessage = "Material texture load failed: unable to canonicalize non-color texture format";
            }
        }
    }
    texture.label = [NSString stringWithFormat:@"Material Texture (%s)", canonicalPath.c_str()];
    return registerMaterialTexture(texture, key, canonicalPath, samplerDesc, semantic);
}

uint32_t SceneResources::addMaterialTextureFromData(const uint8_t* data,
                                                    size_t size,
                                                    const std::string& label,
                                                    bool srgb,
                                                    std::string* errorMessage,
                                                    const MaterialTextureSamplerDesc* samplerDesc,
                                                    MaterialTextureSemantic semantic) {
    if (!m_device) {
        if (errorMessage) {
            *errorMessage = "Material texture load failed: Metal device not ready";
        }
        return kInvalidTextureIndex;
    }
    if (!data || size == 0) {
        if (errorMessage) {
            *errorMessage = "Material texture load failed: empty data";
        }
        return kInvalidTextureIndex;
    }

    const std::string key =
        label + (srgb ? "|srgb" : "|linear") + SamplerKeySuffix(samplerDesc) +
        TextureSemanticKeySuffix(semantic);
    auto cached = m_materialTextureIndex.find(key);
    if (cached != m_materialTextureIndex.end()) {
        return cached->second;
    }

    NSData* nsData = [NSData dataWithBytes:data length:size];
    if (!nsData) {
        if (errorMessage) {
            *errorMessage = "Material texture load failed: NSData allocation failed";
        }
        return kInvalidTextureIndex;
    }

    MTKTextureLoader* loader = [[MTKTextureLoader alloc] initWithDevice:m_device];
    NSDictionary* options = @{
        MTKTextureLoaderOptionSRGB : @(srgb),
        MTKTextureLoaderOptionTextureUsage : @(MTLTextureUsageShaderRead),
        MTKTextureLoaderOptionTextureStorageMode : @(MTLStorageModeShared),
        MTKTextureLoaderOptionGenerateMipmaps : @YES,
        MTKTextureLoaderOptionAllocateMipmaps : @YES
    };

    NSError* error = nil;
    id<MTLTexture> texture = [loader newTextureWithData:nsData options:options error:&error];
    if (!texture) {
        if (errorMessage) {
            NSString* errStr = error ? error.localizedDescription : @"unknown error";
            *errorMessage = std::string("Material texture load failed: ") + errStr.UTF8String;
        }
        return kInvalidTextureIndex;
    }
    if (!srgb) {
        MTLPixelFormat linearFormat = LinearEquivalentPixelFormat(texture.pixelFormat);
        if (linearFormat != texture.pixelFormat) {
            id<MTLTexture> linearTexture = CloneTextureWithFormat(m_device, texture, linearFormat);
            if (linearTexture) {
                texture = linearTexture;
            } else if (errorMessage) {
                *errorMessage = "Material texture load failed: unable to canonicalize non-color texture format";
            }
        }
    }
    texture.label = [NSString stringWithFormat:@"Material Texture (%s)", label.c_str()];
    return registerMaterialTexture(texture, key, label, samplerDesc, semantic);
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
                              MTKTextureLoaderOptionTextureUsage :
                                  @(MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget),
                              MTKTextureLoaderOptionTextureStorageMode : @(MTLStorageModeShared),
                              MTKTextureLoaderOptionAllocateMipmaps : @YES,
                              MTKTextureLoaderOptionGenerateMipmaps : @YES};
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

    if (texture.mipmapLevelCount <= 1 && m_commandQueue) {
        MTLTextureDescriptor* desc =
            [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:texture.pixelFormat
                                                               width:width
                                                              height:height
                                                           mipmapped:YES];
        desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
        desc.storageMode = MTLStorageModeShared;
        id<MTLTexture> mipTexture = [m_device newTextureWithDescriptor:desc];
        if (mipTexture) {
            id<MTLCommandBuffer> commandBuffer = [m_commandQueue commandBuffer];
            id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
            if (blit) {
                MTLOrigin origin = MTLOriginMake(0, 0, 0);
                MTLSize size = MTLSizeMake(width, height, 1);
                [blit copyFromTexture:texture
                         sourceSlice:0
                         sourceLevel:0
                        sourceOrigin:origin
                          sourceSize:size
                           toTexture:mipTexture
                    destinationSlice:0
                    destinationLevel:0
                   destinationOrigin:origin];
                [blit generateMipmapsForTexture:mipTexture];
                [blit endEncoding];
                [commandBuffer commit];
                [commandBuffer waitUntilCompleted];
                texture = mipTexture;
            }
        }
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
    NSLog(@"[EnvRebuild] file=\"%s\" size=%lux%lu mips=%lu aliasN=%u thrHead=%.5f thrSum=%.5f elapsedMs=%.2f",
          fileName.c_str(),
          static_cast<unsigned long>(width),
          static_cast<unsigned long>(height),
          static_cast<unsigned long>(texture.mipmapLevelCount),
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
    m_materialTextures.clear();
    m_materialSamplers.clear();
    m_materialTextureSamplerIndices.clear();
    m_materialTextureLabels.clear();
    m_materialTextureIndex.clear();
    m_materialSamplerIndices.clear();
    m_materialTextureInfoBuffer = nil;
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
            packed.tangent = vertex.tangent;
            packed.uv = simd_make_float4(vertex.uv.x, vertex.uv.y, vertex.uv1.x, vertex.uv1.y);
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
