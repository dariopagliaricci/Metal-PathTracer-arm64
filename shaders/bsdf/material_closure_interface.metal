struct MaterialClosureEvalRecord {
    MaterialClosureMetadata metadata;
    BsdfEvalResult bsdf;
};

struct MaterialClosureSampleRecord {
    MaterialClosureMetadata metadata;
    BsdfSampleResult bsdf;
    BsdfEvalResult evalAtSample;
};

inline MaterialClosureEvalRecord material_closure_eval_record(const MaterialData material,
                                                              const float3 position,
                                                              const float3 normal,
                                                              const float3 wo,
                                                              const float3 wi,
                                                              const FireflyClampParams clampParams,
                                                              const uint sssMode,
                                                              const float diffuseOcclusion,
                                                              const bool specularOnly) {
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
                                specularOnly);
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
                                            const bool specularOnly) {
    return material_closure_eval_record(material,
                                        position,
                                        normal,
                                        wo,
                                        wi,
                                        clampParams,
                                        sssMode,
                                        diffuseOcclusion,
                                        specularOnly).bsdf;
}

inline float material_closure_pdf(const MaterialData material,
                                  const float3 position,
                                  const float3 normal,
                                  const float3 wo,
                                  const float3 wi,
                                  const FireflyClampParams clampParams,
                                  const uint sssMode,
                                  const float diffuseOcclusion,
                                  const bool specularOnly) {
    return material_closure_eval(material,
                                 position,
                                 normal,
                                 wo,
                                 wi,
                                 clampParams,
                                 sssMode,
                                 diffuseOcclusion,
                                 specularOnly).pdf;
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
    if (result.bsdf.pdf > 0.0f && result.evalAtSample.pdf > 0.0f) {
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
