using namespace metal;

#if __METAL_VERSION__ >= 310
#include <metal_raytracing>
using namespace metal::raytracing;
#endif

#ifndef PT_DEBUG_TOOLS
#define PT_DEBUG_TOOLS 0
#endif

#ifndef PT_MNEE_SWRT_RAYS
#define PT_MNEE_SWRT_RAYS 0
#endif
#ifndef PT_MNEE_OCCLUSION_PARITY
#define PT_MNEE_OCCLUSION_PARITY 0
#endif

constant float kInfinity = 1e20f;
constant float kPi = 3.14159265358979323846f;
constexpr sampler environmentSampler(filter::linear,
                                     mip_filter::linear,
                                     address::repeat,
                                     coord::normalized);
constexpr sampler environmentSamplerNearest(filter::nearest,
                                            mip_filter::nearest,
                                            address::repeat,
                                            coord::normalized);
constant float kEpsilon = 1e-3f;
constant float kRayOriginEpsilon = 1e-4f;
constant float kSssThroughputCutoff = 1e-3f;
constant float3 kLuminanceWeights = float3(0.2126f, 0.7152f, 0.0722f);
constant uint kInvalidIndex = 0xffffffffu;
constant uint kWorkingColorSpaceLinearSRGB = 0u;
constant uint kWorkingColorSpaceACEScg = 1u;
constant uint kBvhTraversalStackSize = 128u;
constant float kHardwareOcclusionEpsilon = 5.0e-3f;
constant float kAreaPrimitiveHitSlop = 2.0e-3f;
constant float kSpecularNeePdfFloor = 1.0e-4f;
constant float kSpecularNeeInvPdfClamp = 1.0e4f;
constant float kMisWeightClampMin = 1.0e-4f;
constant float kMisWeightClampMax = 0.9999f;
constant uint kParityReasonHitMiss = 1u;
constant uint kParityReasonT = 2u;
constant uint kParityReasonNormal = 4u;
constant uint kParityReasonId = 8u;
constant uint kParityReasonFrontFace = 16u;
// HWRT hit something at t < tMin (found only via the none→tMin=0 retry).
// SWRT skips it because its tMin stays at kEpsilon throughout.
constant uint kParityReasonBelowTMin = 32u;
constant uint kParityReasonShadingNormal = 64u;
constant uint kParityProbeMain = 0u;
constant uint kParityProbeRectShadow = 1u;
constant uint kParityProbeEnvShadow = 2u;
constant uint kParityProbeSpecEnv = 3u;
constant uint kParityProbeSpecRect = 4u;
constant uint kParityModeProbePixel = 1u;
constant uint kParityModeFirstInMedium = 2u;
constant uint kDebugViewNone = 0u;
constant uint kDebugViewBaseColor = 1u;
constant uint kDebugViewMetallic = 2u;
constant uint kDebugViewRoughness = 3u;
constant uint kDebugViewAO = 4u;
constant uint kDebugViewRestirCandidateSourceId = 10u;
constant uint kDebugViewRestirReservoirConfidence = 11u;
constant uint kDebugViewRestirRegirCellId = 12u;
constant uint kDebugViewRestirPathGuidingMask = 13u;
constant uint kDebugViewRestirPtReuseMask = 14u;
constant uint kDebugViewRestirSvgfVariance = 15u;
constant uint kDebugViewRestirNanInfMask = 16u;
constant uint kDebugViewRestirQueueStageFailure = 17u;
constant uint kDebugViewRadianceCache = 18u;
constant uint kWavefrontFlagEnvLodActive = 1u << 0;
constant uint kWavefrontFlagLastScatterDelta = 1u << 1;
constant uint kDebugEventScatter = 0u;
constant uint kDebugEventBackground = 1u;
constant uint kDebugEventVisibleEmitter = 2u;
constant uint kDebugEventRectNee = 3u;
constant uint kDebugEventEnvNee = 4u;
constant uint kDebugEventSpecEnvNee = 5u;
constant uint kDebugEventSpecRectNee = 6u;
constant uint kDebugEventThroughput = 7u;
constant uint kDebugEventRectSample = 8u;
constant uint kDebugEventRectEval = 9u;
constant uint kDebugEventRectShadow = 10u;
constant uint kDebugEventRay = 11u;
constant uint kDebugEventShadingNormal = 12u;
constant uint kDebugEventTbnBasis0 = 13u;
constant uint kDebugEventTbnBasis1 = 14u;
constant uint kDebugEventTbnBasis2 = 15u;
constant uint kDebugEventTbnBasis3 = 16u;
constant uint kDebugEventTbnBasis4 = 17u;
constant uint kDebugEventBsdfState = 18u;
constant uint kDebugEventBsdfRng = 19u;
constant uint kDebugEventBsdfMaterial0 = 20u;
constant uint kDebugEventBsdfMaterial1 = 21u;
constant uint kDebugEventBsdfMaterial2 = 22u;
constant uint kDebugEventBsdfGradients = 23u;
constant uint kDebugEventDirectLightAuditMeta = 26u;
constant uint kDebugEventDirectLightAuditEval = 27u;
constant uint kDebugEventDirectLightAuditContrib = 28u;
constant uint kDebugEventDirectLightAuditMaterial = 29u;
constant uint kDebugEventDirectLightAuditShadowRaw = 30u;
constant uint kDebugEventDirectLightAuditShadowMapped = 31u;
constant uint kDebugEventDirectLightAuditShadowEmbeddedSw = 32u;
constant uint kDebugEventDirectLightAuditShadowBinding = 33u;
constant uint kDebugEventDirectLightAuditShadowReject = 34u;
constant uint kDebugEventRisAuditState = 35u;
constant uint kDebugEventRisAuditWinnerA = 36u;
constant uint kDebugEventRisAuditWinnerB = 37u;
constant uint kDebugEventRisAuditWinnerC = 38u;
constant uint kDebugEventSpatialReuseAudit = 39u;
constant uint kDebugEventSpatialReuseWeights = 40u;
constant uint kDebugEventTemporalReuseAudit = 41u;
constant uint kDebugEventTemporalReuseWeights = 42u;
constant uint kDebugEventWorldReuseAudit = 43u;
constant uint kDebugEventWorldReuseWeights = 44u;
constant uint kDebugEventCacheReuseAudit = 45u;
constant uint kDebugEventCacheReuseWeights = 46u;
constant uint kDirectLightModeLegacyRect = 0u;
constant uint kDirectLightModeBaselineEmissive = 1u;
constant uint kDirectLightModeRis = 2u;
constant uint kDirectLightModeRisSpatialReuse = 3u;
constant uint kDirectLightModeRisTemporalReuse = 4u;
constant uint kDirectLightModeRisWorldReuse = 5u;
constant uint kDirectLightModeRisRegirCache = 6u;
constant uint kDirectLightModeRestirDi = 7u;
constant uint kDirectLightModeRestirDiRegirHybrid = 8u;
constant uint kRestirGiModeOff = 0u;
constant uint kRestirGiModeDiffusePrototype = 1u;
constant uint kPathGuidingModeOff = 0u;
constant uint kPathGuidingModePrototype = 1u;
constant uint kRestirPtModeOff = 0u;
constant uint kRestirPtModeResearch = 1u;
constant uint kRestirPtModeExperimentalPathReuse = 2u;
constant uint kRadianceCacheModeOff = 0u;
constant uint kRadianceCacheModePrototype = 1u;
constant uint kWavefrontShadowFlagEnvironment = 1u << 0;
constant uint kWavefrontShadowFlagEmissivePrimitive = 1u << 1;
constant int kDirectLightAuditKindRect = 0;
constant int kDirectLightAuditKindEmissivePrimitive = 1;

constant uint kShadowAuditRejectSelfLightRect = 1u << 0;
constant uint kShadowAuditRejectHwSelfHit = 1u << 1;
constant uint kShadowAuditRejectHwExcluded = 1u << 2;
constant uint kShadowAuditRejectHwOnlyNearMesh = 1u << 3;
constant uint kShadowAuditRejectHwNearRetryForeign = 1u << 4;
constant uint kShadowAuditRejectEmbeddedSwOverride = 1u << 5;
