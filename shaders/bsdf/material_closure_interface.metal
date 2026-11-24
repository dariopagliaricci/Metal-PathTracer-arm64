struct MaterialClosureEvalRecord {
    MaterialClosureMetadata metadata;
    BsdfEvalResult bsdf;
};

struct MaterialClosureSampleRecord {
    MaterialClosureMetadata metadata;
    BsdfSampleResult bsdf;
    BsdfEvalResult evalAtSample;
};

inline MaterialClosureMetadata material_closure_metadata_for_sample(const MaterialData material,
                                                                    const BsdfSampleResult sample) {
    MaterialClosureMetadata closure = material_primary_closure_metadata(material);
    closure.lobeIndex = sample.lobeType;
    closure.roughness = clamp(max(closure.roughness, sample.lobeRoughness), 0.0f, 1.0f);
    if (sample.isDelta) {
        closure.flags |= kMaterialClosureFlagDelta;
        closure.flags &= ~kMaterialClosureFlagConnectable;
    }
    if (sample.isBssrdf) {
        closure.flags |= kMaterialClosureFlagVolumetric;
    }
    return closure;
}

inline MaterialClosureEvalRecord material_closure_eval_record(const MaterialData material,
                                                              const float3 position,
                                                              const float3 normal,
                                                              const float3 wo,
                                                              const float3 wi,
                                                              const FireflyClampParams clampParams,
                                                              const uint sssMode,
                                                              const float diffuseOcclusion,
                                                              const bool specularOnly,
                                                              const bool referencePunctual = false) {
    MaterialClosureEvalRecord result;
    result.metadata = material_primary_closure_metadata(material);
    result.bsdf = evaluate_bsdf(material,
                                position,
                                normal,
                                wo,
                                wi,
                                clampParams,
                                sssMode,
                                diffuseOcclusion,
                                specularOnly,
                                referencePunctual);
    return result;
}

inline BsdfEvalResult material_closure_eval(const MaterialData material,
                                            const float3 position,
                                            const float3 normal,
                                            const float3 wo,
                                            const float3 wi,
                                            const FireflyClampParams clampParams,
                                            const uint sssMode,
                                            const float diffuseOcclusion,
                                            const bool specularOnly,
                                            const bool referencePunctual = false) {
    return material_closure_eval_record(material,
                                        position,
                                        normal,
                                        wo,
                                        wi,
                                        clampParams,
                                        sssMode,
                                        diffuseOcclusion,
                                        specularOnly,
                                        referencePunctual).bsdf;
}

inline MaterialClosureSampleRecord material_closure_sample_record(const MaterialData material,
                                                                  const float3 position,
                                                                  const float3 normal,
                                                                  const float3 wo,
                                                                  const float3 incidentDir,
                                                                  bool frontFace,
                                                                  thread uint& state,
                                                                  const FireflyClampParams clampParams,
                                                                  const uint sssMode,
                                                                  const float diffuseOcclusion,
                                                                  const bool specularOnly) {
    MaterialClosureSampleRecord result;
    result.metadata = material_primary_closure_metadata(material);
    result.bsdf = sample_bsdf(material,
                              position,
                              normal,
                              wo,
                              incidentDir,
                              frontFace,
                              state,
                              clampParams,
                              sssMode,
                              diffuseOcclusion,
                              specularOnly);
    result.evalAtSample = material_closure_eval(material,
                                                position,
                                                normal,
                                                wo,
                                                result.bsdf.direction,
                                                clampParams,
                                                sssMode,
                                                diffuseOcclusion,
                                                specularOnly);
    result.metadata = material_closure_metadata_for_sample(material, result.bsdf);
    if (!result.bsdf.isDelta && result.bsdf.pdf > 0.0f && result.evalAtSample.pdf > 0.0f) {
        result.bsdf.directionalPdf = result.evalAtSample.directionalPdf;
        result.bsdf.areaPdf = result.evalAtSample.areaPdf;
    }
    return result;
}

inline BsdfSampleResult material_closure_sample(const MaterialData material,
                                                const float3 position,
                                                const float3 normal,
                                                const float3 wo,
                                                const float3 incidentDir,
                                                bool frontFace,
                                                thread uint& state,
                                                const FireflyClampParams clampParams,
                                                const uint sssMode,
                                                const float diffuseOcclusion,
                                                const bool specularOnly) {
    return material_closure_sample_record(material,
                                          position,
                                          normal,
                                          wo,
                                          incidentDir,
                                          frontFace,
                                          state,
                                          clampParams,
                                          sssMode,
                                          diffuseOcclusion,
                                          specularOnly).bsdf;
}
