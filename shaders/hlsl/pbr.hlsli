// PBR (Physically Based Rendering) functions for HLSL shaders
// Implements Cook-Torrance BRDF with GGX distribution
//
// This file contains:
//   - Normal Distribution Function (GGX/Trowbridge-Reitz)
//   - Geometry Function (Smith GGX with Schlick approximation)
//   - Fresnel Function (Schlick approximation)
//   - Cook-Torrance specular BRDF
//
// Reference: "Real-Time Rendering" Chapter 9, "Physically Based Rendering"
// Industry standard: Disney/GGX model used by Unreal Engine, Unity, etc.

#ifndef PBR_HLSLI
#define PBR_HLSLI

// Mathematical constants
static const float PI = 3.14159265358979323846;
static const float EPSILON = 0.0001;

// ============================================================================
// PBR Material Structure
// ============================================================================

// PBR material properties for Metallic-Roughness Workflow
// Standard material representation compatible with glTF 2.0 and common PBR pipelines
//
// Members:
//   albedo: Base color (diffuse for dielectrics, F0 for metals)
//   metallic: Metalness factor (0 = dielectric, 1 = metal)
//   roughness: Surface roughness (0 = smooth, 1 = rough)
//   ao: Ambient occlusion factor (0 = fully occluded, 1 = no occlusion)
//   emissive: Self-emission color (added to final output)
struct PBRMaterial
{
    float3 albedo;
    float metallic;
    float roughness;
    float ao;
    float3 emissive;
};

// ============================================================================
// Normal Distribution Function (NDF)
// ============================================================================

// GGX/Trowbridge-Reitz Normal Distribution Function
// Describes the statistical distribution of microfacet orientations
//
// Parameters:
//   N: Surface normal (normalized)
//   H: Half vector between view and light (normalized)
//   roughness: Surface roughness (0 = smooth, 1 = rough)
//
// Returns: Probability that microfacets are oriented along H
float DistributionGGX(float3 N, float3 H, float roughness)
{
    // Disney's remapping: square the roughness for more perceptually linear control
    float a = roughness * roughness;
    float a2 = a * a;

    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;

    // GGX formula: D = a^2 / (PI * ((NdotH^2 * (a^2 - 1) + 1)^2))
    float denom = NdotH2 * (a2 - 1.0) + 1.0;
    denom = PI * denom * denom;

    return a2 / max(denom, EPSILON);
}

// ============================================================================
// Geometry Function
// ============================================================================

// Schlick-GGX Geometry Function (single direction)
// Approximates self-shadowing and masking of microfacets
//
// Parameters:
//   NdotV: Dot product of normal and view/light direction
//   roughness: Surface roughness
//
// Returns: Probability that microfacet is visible from the given direction
float GeometrySchlickGGX(float NdotV, float roughness)
{
    // Remapping for direct lighting (differs from IBL)
    // k = (roughness + 1)^2 / 8 for direct lighting
    // k = roughness^2 / 2 for IBL
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;

    float denom = NdotV * (1.0 - k) + k;
    return NdotV / max(denom, EPSILON);
}

// Smith's Method using Schlick-GGX
// Combines geometry shadowing (light blocked by microfacets) and
// geometry masking (view blocked by microfacets)
//
// Parameters:
//   N: Surface normal (normalized)
//   V: View direction (normalized)
//   L: Light direction (normalized)
//   roughness: Surface roughness
//
// Returns: Combined visibility term G(V) * G(L)
float GeometrySmith(float3 N, float3 V, float3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);

    float ggxV = GeometrySchlickGGX(NdotV, roughness);
    float ggxL = GeometrySchlickGGX(NdotL, roughness);

    return ggxV * ggxL;
}

// ============================================================================
// Fresnel Function
// ============================================================================

// Schlick's Fresnel Approximation
// Models how reflection increases at grazing angles
//
// Parameters:
//   cosTheta: Angle between view and half vector (VdotH or NdotV)
//   F0: Base reflectivity at normal incidence
//       - Dielectrics: typically 0.04 (4% reflectance)
//       - Metals: use the albedo color (tinted reflection)
//
// Returns: Fresnel reflectance (colored for metals)
float3 FresnelSchlick(float cosTheta, float3 F0)
{
    // Clamp to avoid numerical issues with pow
    float ct = saturate(cosTheta);
    return F0 + (1.0 - F0) * pow(1.0 - ct, 5.0);
}

// Fresnel with roughness factor for ambient/IBL lighting
// Rougher surfaces have less pronounced Fresnel effect
//
// Parameters:
//   cosTheta: Angle between view and half vector
//   F0: Base reflectivity at normal incidence
//   roughness: Surface roughness
//
// Returns: Roughness-attenuated Fresnel reflectance
float3 FresnelSchlickRoughness(float cosTheta, float3 F0, float roughness)
{
    float ct = saturate(cosTheta);
    float3 F90 = max(float3(1.0 - roughness, 1.0 - roughness, 1.0 - roughness), F0);
    return F0 + (F90 - F0) * pow(1.0 - ct, 5.0);
}

// ============================================================================
// Cook-Torrance BRDF
// ============================================================================

// Calculate F0 (base reflectivity) for metallic workflow
// Dielectrics have constant F0 of 0.04
// Metals use their albedo as F0 (tinted reflections)
//
// Parameters:
//   albedo: Surface base color
//   metallic: Metalness factor (0 = dielectric, 1 = metal)
//
// Returns: Base reflectivity for Fresnel calculations
float3 CalculateF0(float3 albedo, float metallic)
{
    // Standard dielectric F0 (4% reflectance, typical for non-metals)
    float3 dielectricF0 = float3(0.04, 0.04, 0.04);

    // Metals use albedo as F0, dielectrics use constant
    return lerp(dielectricF0, albedo, metallic);
}

// Cook-Torrance Specular BRDF
// Calculates the specular reflection for a single light
//
// Parameters:
//   N: Surface normal (normalized)
//   V: View direction (normalized)
//   L: Light direction (normalized)
//   roughness: Surface roughness
//   F0: Base reflectivity
//
// Returns: Specular BRDF value (multiply with NdotL and light color for final radiance)
float3 CookTorranceSpecular(float3 N, float3 V, float3 L, float roughness, float3 F0)
{
    float3 H = normalize(V + L);

    float NdotL = max(dot(N, L), 0.0);
    float NdotV = max(dot(N, V), 0.0);
    float VdotH = max(dot(V, H), 0.0);

    // Early out for surfaces facing away from light or view
    if (NdotL <= 0.0 || NdotV <= 0.0)
        return float3(0.0, 0.0, 0.0);

    // Calculate each component of Cook-Torrance
    float D = DistributionGGX(N, H, roughness);
    float G = GeometrySmith(N, V, L, roughness);
    float3 F = FresnelSchlick(VdotH, F0);

    // Cook-Torrance specular BRDF: DGF / (4 * NdotV * NdotL)
    float3 numerator = D * G * F;
    float denominator = 4.0 * NdotV * NdotL;
    float3 specular = numerator / max(denominator, EPSILON);

    return specular;
}

// Full PBR lighting calculation for a single light
// Combines diffuse (Lambertian) and specular (Cook-Torrance)
//
// Parameters:
//   N: Surface normal (normalized)
//   V: View direction (normalized)
//   L: Light direction (normalized)
//   albedo: Surface base color
//   metallic: Metalness factor
//   roughness: Surface roughness
//   lightColor: Light color * intensity
//
// Returns: Total reflected radiance for this light
float3 CalculatePBRLighting(
    float3 N,
    float3 V,
    float3 L,
    float3 albedo,
    float metallic,
    float roughness,
    float3 lightColor)
{
    float3 H = normalize(V + L);

    float NdotL = max(dot(N, L), 0.0);
    float NdotV = max(dot(N, V), 0.0);
    float VdotH = max(dot(V, H), 0.0);

    // Early out for surfaces facing away from light
    if (NdotL <= 0.0)
        return float3(0.0, 0.0, 0.0);

    // Calculate F0 based on metallic workflow
    float3 F0 = CalculateF0(albedo, metallic);

    // Calculate Cook-Torrance specular components
    float D = DistributionGGX(N, H, roughness);
    float G = GeometrySmith(N, V, L, roughness);
    float3 F = FresnelSchlick(VdotH, F0);

    // Specular BRDF
    float3 numerator = D * G * F;
    float denominator = 4.0 * NdotV * NdotL;
    float3 specular = numerator / max(denominator, EPSILON);

    // Energy conservation: what's not reflected is refracted (diffuse)
    // Metals have no diffuse (all light is reflected or absorbed)
    float3 kS = F;           // Specular reflection ratio
    float3 kD = 1.0 - kS;    // Diffuse refraction ratio
    kD *= (1.0 - metallic);  // Metals have no diffuse

    // Lambertian diffuse
    float3 diffuse = albedo / PI;

    // Combine diffuse and specular, apply light
    float3 Lo = (kD * diffuse + specular) * lightColor * NdotL;

    return Lo;
}

// Calculate PBR direct lighting using PBRMaterial struct
// Implements Metallic-Roughness Workflow with Cook-Torrance BRDF
//
// This function combines diffuse (Lambertian) and specular (Cook-Torrance)
// reflections with proper energy conservation for metallic surfaces.
//
// Key concepts:
//   - F0: Non-metals have constant 0.04 (4% reflectance), metals use albedo
//   - kD: Diffuse ratio (1 - kS), metals have no diffuse (kD = 0 when metallic = 1)
//   - Energy conservation: kS + kD = 1
//
// Parameters:
//   N: Surface normal (normalized)
//   V: View direction (normalized, from surface to camera)
//   L: Light direction (normalized, from surface to light)
//   radiance: Light color * intensity (incoming radiance from light)
//   material: PBR material properties
//
// Returns: Outgoing radiance (reflected light) for this light source
//          Note: Does NOT include ambient occlusion or emissive
float3 CalculatePBRDirect(
    float3 N,
    float3 V,
    float3 L,
    float3 radiance,
    PBRMaterial material)
{
    float3 H = normalize(V + L);

    // F0 - Surface reflection at zero incidence
    // Dielectrics: constant 0.04 (4% Fresnel reflectance)
    // Metals: use albedo color (tinted reflections)
    float3 F0 = float3(0.04, 0.04, 0.04);
    F0 = lerp(F0, material.albedo, material.metallic);

    // Cook-Torrance BRDF components
    float NDF = DistributionGGX(N, H, material.roughness);
    float G = GeometrySmith(N, V, L, material.roughness);
    float3 F = FresnelSchlick(max(dot(H, V), 0.0), F0);

    // Energy conservation
    // kS = F (energy of light that gets reflected as specular)
    // kD = 1 - kS (energy that gets refracted and becomes diffuse)
    float3 kS = F;
    float3 kD = float3(1.0, 1.0, 1.0) - kS;

    // Metals have no diffuse reflection (all light is either reflected or absorbed)
    kD *= 1.0 - material.metallic;

    // Cook-Torrance specular BRDF: DGF / (4 * NdotV * NdotL)
    float3 numerator = NDF * G * F;
    float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + EPSILON;
    float3 specular = numerator / denominator;

    // NdotL: Lambert's cosine law
    float NdotL = max(dot(N, L), 0.0);

    // Combine diffuse and specular contributions
    // Diffuse: Lambertian BRDF = albedo / PI
    // The result is multiplied by incoming radiance and NdotL
    return (kD * material.albedo / PI + specular) * radiance * NdotL;
}

// Calculate complete PBR direct lighting with ambient and emissive
// Convenience function that wraps CalculatePBRDirect with ambient occlusion and emission
//
// Parameters:
//   N: Surface normal (normalized)
//   V: View direction (normalized)
//   L: Light direction (normalized)
//   radiance: Light color * intensity
//   material: PBR material properties
//   ambientLight: Ambient/environment light contribution
//
// Returns: Final color including direct lighting, ambient (with AO), and emissive
float3 CalculatePBRDirectComplete(
    float3 N,
    float3 V,
    float3 L,
    float3 radiance,
    PBRMaterial material,
    float3 ambientLight)
{
    // Direct lighting from light source
    float3 Lo = CalculatePBRDirect(N, V, L, radiance, material);

    // Apply ambient with occlusion
    // Metals have no diffuse reflection, so ambient diffuse is zero for pure metals
    float3 ambient = ambientLight * material.albedo * material.ao * (1.0 - material.metallic);

    // Combine direct, ambient, and emissive
    return Lo + ambient + material.emissive;
}

// ============================================================================
// Image-Based Lighting (IBL) Functions
// ============================================================================

// Maximum mip level for prefiltered environment map
// Corresponds to the roughness range 0.0 (mip 0) to 1.0 (max mip)
// Value = floor(log2(PrefilteredMapSize)) = floor(log2(128)) = 7
static const float MAX_REFLECTION_LOD = 7.0;

// Calculate IBL (Image-Based Lighting) contribution
// Implements the split-sum approximation for real-time IBL
//
// The split-sum approximation separates the lighting integral into two parts:
// 1. Pre-filtered environment map (convolved with GGX for different roughness levels)
// 2. BRDF integration lookup table (pre-computed Fresnel and geometry terms)
//
// This allows us to compute IBL in real-time by combining:
//   - Diffuse IBL from irradiance map
//   - Specular IBL from prefiltered map + BRDF LUT
//
// Parameters:
//   N: Surface normal (normalized)
//   V: View direction (normalized, from surface to camera)
//   R: Reflection direction (normalized, reflect(-V, N))
//   material: PBR material properties
//   irradianceMap: Pre-convolved irradiance cubemap for diffuse IBL
//   prefilteredMap: Pre-filtered environment map with roughness-based mip levels
//   brdfLUT: 2D lookup table storing pre-integrated BRDF (scale, bias)
//   linearSampler: Linear filtering sampler for texture sampling
//
// Returns: Ambient lighting contribution from IBL
//
// Reference: "Real Shading in Unreal Engine 4" (Brian Karis, SIGGRAPH 2013)
float3 CalculateIBL(
    float3 N,
    float3 V,
    float3 R,
    PBRMaterial material,
    TextureCube<float4> irradianceMap,
    TextureCube<float4> prefilteredMap,
    Texture2D<float4> brdfLUT,
    SamplerState linearSampler)
{
    // Calculate F0 (base reflectivity at normal incidence)
    // Dielectrics: 0.04 (4% reflectance)
    // Metals: use albedo color (tinted reflections)
    float3 F0 = lerp(float3(0.04, 0.04, 0.04), material.albedo, material.metallic);

    // NdotV for Fresnel and BRDF LUT lookup
    float NdotV = max(dot(N, V), 0.0);

    // Fresnel with roughness consideration for IBL
    // Rougher surfaces have less pronounced Fresnel effect
    float3 F = FresnelSchlickRoughness(NdotV, F0, material.roughness);

    // Energy conservation for diffuse/specular split
    // kS = Fresnel term (specular reflection ratio)
    // kD = 1 - kS (diffuse refraction ratio)
    float3 kS = F;
    float3 kD = 1.0 - kS;

    // Metals have no diffuse reflection (all light is reflected or absorbed)
    kD *= 1.0 - material.metallic;

    // -------------------------------------------------------------------------
    // Diffuse IBL
    // -------------------------------------------------------------------------
    // Sample the irradiance map using the surface normal
    // The irradiance map contains pre-convolved diffuse lighting from all directions
    float3 irradiance = irradianceMap.Sample(linearSampler, N).rgb;
    float3 diffuse = irradiance * material.albedo;

    // -------------------------------------------------------------------------
    // Specular IBL (Split-Sum Approximation)
    // -------------------------------------------------------------------------
    // Sample the prefiltered environment map using the reflection direction
    // Higher roughness samples from higher mip levels (more blurred)
    float3 prefilteredColor = prefilteredMap.SampleLevel(
        linearSampler,
        R,
        material.roughness * MAX_REFLECTION_LOD
    ).rgb;

    // Sample the BRDF LUT to get pre-integrated scale and bias
    // U = NdotV, V = roughness
    // R channel = scale factor (1 - Fc) * G_Vis integral
    // G channel = bias factor Fc * G_Vis integral
    float2 brdf = brdfLUT.Sample(linearSampler, float2(NdotV, material.roughness)).rg;

    // Combine prefiltered color with BRDF integration
    // Split-sum approximation: Specular = PrefilterColor * (F0 * scale + bias)
    // Note: Use F0, not F. The BRDF LUT already encodes the Fresnel behavior.
    float3 specular = prefilteredColor * (F0 * brdf.x + brdf.y);

    // -------------------------------------------------------------------------
    // Final Ambient
    // -------------------------------------------------------------------------
    // Combine diffuse and specular IBL, apply ambient occlusion
    // AO is only applied to ambient/IBL lighting (not direct lighting)
    float3 ambient = (kD * diffuse + specular) * material.ao;

    return ambient;
}

// ============================================================================
// Helper Functions
// ============================================================================

// Clamp roughness to avoid numerical issues
// Very low roughness can cause division by zero or very bright spots
float ClampRoughness(float roughness)
{
    return max(roughness, 0.04);
}

// Simple hemisphere ambient approximation
// Blends between ground and sky color based on normal direction
float3 CalculateHemisphereAmbient(float3 N, float3 albedo, float ao)
{
    float3 skyColor = float3(0.15, 0.18, 0.25);    // Cool sky ambient
    float3 groundColor = float3(0.08, 0.06, 0.04); // Warm ground bounce

    float upFactor = N.y * 0.5 + 0.5; // Map -1..1 to 0..1
    float3 ambientColor = lerp(groundColor, skyColor, upFactor);

    return ambientColor * albedo * ao;
}

#endif // PBR_HLSLI
