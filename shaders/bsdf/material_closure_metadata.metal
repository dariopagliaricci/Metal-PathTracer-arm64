inline bool material_closure_has_flag(const MaterialClosureMetadata closure, const uint flag) {
    return (closure.flags & flag) != 0u;
}

inline float material_closure_emission_weight(const MaterialData material) {
    return luminance_rgb(max(material.emission.xyz, float3(0.0f)));
}

inline float material_closure_transmission_weight(const MaterialData material) {
    const float metallic = clamp(material.pbrParams.x, 0.0f, 1.0f);
    return clamp(material.pbrExtras.z, 0.0f, 1.0f) * (1.0f - metallic);
}

inline float3 material_closure_specular_aov(const MaterialData material) {
    const uint materialType = static_cast<uint>(material.typeEta.x);
    if (materialType == 1u) {
        return conductor_f0(material);
    }
    if (materialType == 6u) {
        return carpaint_base_f0(material);
    }
    if (materialType == 7u) {
        const float metallic = clamp(material.pbrParams.x, 0.0f, 1.0f);
        const float dielectricF0 = dielectric_f0_from_ior(material.typeEta.y);
        return mix(float3(dielectricF0), material_base_color(material), metallic);
    }
    if (materialType == 2u || materialType == 4u) {
        return float3(dielectric_f0_from_ior(material.typeEta.y));
    }
    return float3(0.0f);
}

inline uint material_closure_count(const MaterialData material) {
    const uint materialType = static_cast<uint>(material.typeEta.x);
    const bool hasEmission = material_closure_emission_weight(material) > 0.0f;
    if (materialType == 3u) {
        return 1u;
    }
    if (materialType == 4u || materialType == 6u) {
        return hasEmission ? 3u : 2u;
    }
    return hasEmission ? 2u : 1u;
}

inline MaterialClosureMetadata make_material_closure_metadata(const MaterialData material,
                                                             const uint type,
                                                             const uint flags,
                                                             const float weight,
                                                             const float roughness,
                                                             const float eta,
                                                             const uint lobeIndex) {
    MaterialClosureMetadata closure{};
    closure.type = type;
    closure.flags = flags;
    closure.weight = clamp(weight, 0.0f, 1.0f);
    closure.roughness = clamp(roughness, 0.0f, 1.0f);
    closure.eta = max(eta, 1.0f);
    closure.lobeIndex = lobeIndex;
    closure.reserved0 = 0u;
    closure.reserved1 = 0u;
    closure.aovAlbedoRoughness = float4(material_base_color(material), closure.roughness);
    closure.aovSpecularTransmission =
        float4(material_closure_specular_aov(material), material_closure_transmission_weight(material));
    closure.aovEmission = float4(max(material.emission.xyz, float3(0.0f)),
                                 material_closure_emission_weight(material));
    return closure;
}

inline MaterialClosureMetadata material_emission_closure_metadata(const MaterialData material,
                                                                 const uint lobeIndex) {
    return make_material_closure_metadata(material,
                                          kMaterialClosureTypeEmission,
                                          kMaterialClosureFlagEmissive |
                                              kMaterialClosureFlagConnectable,
                                          clamp(material_closure_emission_weight(material), 0.0f, 1.0f),
                                          1.0f,
                                          max(material.typeEta.y, 1.0f),
                                          lobeIndex);
}

inline MaterialClosureMetadata material_closure_metadata_at(const MaterialData material,
                                                           const uint requestedLobe) {
    const uint materialType = static_cast<uint>(material.typeEta.x);
    const bool hasEmission = material_closure_emission_weight(material) > 0.0f;
    const uint lobe = min(requestedLobe, material_closure_count(material) - 1u);

    if (materialType == 3u || (hasEmission && lobe == material_closure_count(material) - 1u)) {
        return material_emission_closure_metadata(material, lobe);
    }

    const float roughness = clamp(material.baseColorRoughness.w, 0.0f, 1.0f);
    const bool delta = material_is_delta(material);
    const uint deltaFlag = delta ? kMaterialClosureFlagDelta : 0u;
    const uint connectableFlag = delta ? 0u : kMaterialClosureFlagConnectable;
    const float eta = max(material.typeEta.y, 1.0f);

    switch (materialType) {
        case 0u:
            return make_material_closure_metadata(material,
                                                  kMaterialClosureTypeDiffuse,
                                                  kMaterialClosureFlagDiffuse |
                                                      kMaterialClosureFlagConnectable |
                                                      kMaterialClosureFlagGuideable |
                                                      kMaterialClosureFlagCacheable,
                                                  1.0f,
                                                  1.0f,
                                                  eta,
                                                  lobe);
        case 1u:
            return make_material_closure_metadata(material,
                                                  kMaterialClosureTypeConductor,
                                                  kMaterialClosureFlagSpecular |
                                                      kMaterialClosureFlagGlossy |
                                                      connectableFlag |
                                                      deltaFlag,
                                                  1.0f,
                                                  roughness,
                                                  eta,
                                                  lobe);
        case 2u:
            return make_material_closure_metadata(material,
                                                  kMaterialClosureTypeDielectric,
                                                  kMaterialClosureFlagSpecular |
                                                      kMaterialClosureFlagTransmissive |
                                                      deltaFlag,
                                                  1.0f,
                                                  roughness,
                                                  eta,
                                                  lobe);
        case 4u:
            if (lobe == 0u) {
                return make_material_closure_metadata(material,
                                                      kMaterialClosureTypeClearcoat,
                                                      kMaterialClosureFlagSpecular |
                                                          kMaterialClosureFlagGlossy |
                                                          connectableFlag |
                                                          deltaFlag,
                                                      max(material.coatParams.z, 0.0f),
                                                      plastic_coat_roughness(material),
                                                      max(material.typeEta.z, eta),
                                                      lobe);
            }
            return make_material_closure_metadata(material,
                                                  kMaterialClosureTypeDiffuse,
                                                  kMaterialClosureFlagDiffuse |
                                                      kMaterialClosureFlagConnectable,
                                                  1.0f,
                                                  1.0f,
                                                  eta,
                                                  lobe);
        case 5u:
            return make_material_closure_metadata(material,
                                                  kMaterialClosureTypeSubsurface,
                                                  kMaterialClosureFlagDiffuse |
                                                      kMaterialClosureFlagVolumetric |
                                                      kMaterialClosureFlagConnectable,
                                                  1.0f,
                                                  1.0f,
                                                  eta,
                                                  lobe);
        case 6u:
            if (lobe == 0u) {
                return make_material_closure_metadata(material,
                                                      kMaterialClosureTypeCarPaint,
                                                      kMaterialClosureFlagSpecular |
                                                          kMaterialClosureFlagGlossy |
                                                          connectableFlag |
                                                          deltaFlag,
                                                      1.0f,
                                                      carpaint_base_roughness(material),
                                                      eta,
                                                      lobe);
            }
            return make_material_closure_metadata(material,
                                                  kMaterialClosureTypeClearcoat,
                                                  kMaterialClosureFlagSpecular |
                                                      kMaterialClosureFlagGlossy |
                                                      connectableFlag,
                                                  carpaint_coat_sample_weight(material),
                                                  plastic_coat_roughness(material),
                                                  max(material.typeEta.z, eta),
                                                  lobe);
        case 7u: {
            const float metallic = clamp(material.pbrParams.x, 0.0f, 1.0f);
            const float transmission = material_closure_transmission_weight(material);
            if (transmission > 0.05f) {
                return make_material_closure_metadata(material,
                                                      kMaterialClosureTypeTransmission,
                                                      kMaterialClosureFlagSpecular |
                                                          kMaterialClosureFlagGlossy |
                                                          kMaterialClosureFlagTransmissive |
                                                          connectableFlag |
                                                          deltaFlag,
                                                      transmission,
                                                      roughness,
                                                      eta,
                                                      lobe);
            }
            if (metallic > 0.25f) {
                return make_material_closure_metadata(material,
                                                      kMaterialClosureTypePbrSpecular,
                                                      kMaterialClosureFlagSpecular |
                                                          kMaterialClosureFlagGlossy |
                                                          connectableFlag |
                                                          deltaFlag,
                                                      metallic,
                                                      roughness,
                                                      eta,
                                                      lobe);
            }
            const uint diffuseFlags =
                kMaterialClosureFlagDiffuse |
                kMaterialClosureFlagConnectable |
                (roughness >= 0.35f ? (kMaterialClosureFlagGuideable |
                                       kMaterialClosureFlagCacheable)
                                    : 0u);
            return make_material_closure_metadata(material,
                                                  kMaterialClosureTypePbrDiffuse,
                                                  diffuseFlags,
                                                  1.0f - metallic,
                                                  roughness,
                                                  eta,
                                                  lobe);
        }
        default:
            break;
    }

    return make_material_closure_metadata(material,
                                          kMaterialClosureTypeDiffuse,
                                          kMaterialClosureFlagDiffuse |
                                              kMaterialClosureFlagConnectable,
                                          1.0f,
                                          1.0f,
                                          eta,
                                          lobe);
}

inline MaterialClosureMetadata material_primary_closure_metadata(const MaterialData material) {
    return material_closure_metadata_at(material, 0u);
}

inline bool material_closure_restir_diffuse_eligible(const MaterialData material) {
    const MaterialClosureMetadata closure = material_primary_closure_metadata(material);
    return material_closure_has_flag(closure, kMaterialClosureFlagDiffuse) &&
           material_closure_has_flag(closure, kMaterialClosureFlagConnectable) &&
           !material_closure_has_flag(closure, kMaterialClosureFlagTransmissive) &&
           !material_closure_has_flag(closure, kMaterialClosureFlagEmissive);
}

inline bool material_closure_is_guideable(const MaterialData material) {
    const MaterialClosureMetadata closure = material_primary_closure_metadata(material);
    return material_closure_has_flag(closure, kMaterialClosureFlagGuideable) &&
           !material_closure_has_flag(closure, kMaterialClosureFlagDelta);
}

inline bool material_closure_is_cacheable(const MaterialData material) {
    const MaterialClosureMetadata closure = material_primary_closure_metadata(material);
    return material_closure_has_flag(closure, kMaterialClosureFlagCacheable) &&
           !material_closure_has_flag(closure, kMaterialClosureFlagDelta);
}

inline bool material_closure_is_connectable(const MaterialData material) {
    return material_closure_has_flag(material_primary_closure_metadata(material),
                                     kMaterialClosureFlagConnectable);
}

inline bool material_closure_is_emissive(const MaterialData material) {
    const uint closureCount = material_closure_count(material);
    for (uint lobe = 0u; lobe < closureCount; ++lobe) {
        if (material_closure_has_flag(material_closure_metadata_at(material, lobe),
                                      kMaterialClosureFlagEmissive)) {
            return true;
        }
    }
    return false;
}

inline float3 material_closure_aov_albedo(const MaterialData material) {
    return material_primary_closure_metadata(material).aovAlbedoRoughness.xyz;
}

inline float material_closure_aov_roughness(const MaterialData material) {
    return material_primary_closure_metadata(material).aovAlbedoRoughness.w;
}

inline float3 material_closure_aov_specular(const MaterialData material) {
    return material_primary_closure_metadata(material).aovSpecularTransmission.xyz;
}

inline float material_closure_aov_transmission(const MaterialData material) {
    return material_primary_closure_metadata(material).aovSpecularTransmission.w;
}

inline float4 material_closure_aov_features(const MaterialData material) {
    const MaterialClosureMetadata closure = material_primary_closure_metadata(material);
    const float specularLuma = luminance_rgb(max(closure.aovSpecularTransmission.xyz, float3(0.0f)));
    return float4(closure.aovAlbedoRoughness.w,
                  closure.aovSpecularTransmission.w,
                  specularLuma,
                  material.typeEta.x);
}

inline float3 material_closure_aov_emission(const MaterialData material) {
    return material_primary_closure_metadata(material).aovEmission.xyz;
}
