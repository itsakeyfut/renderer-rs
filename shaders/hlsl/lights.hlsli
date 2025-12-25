// Light structure definitions for HLSL shaders
// These structures must match the C++ definitions in Scene/Light.h
//
// Usage:
//   #include "lights.hlsli"
//
// Binding:
//   cbuffer LightData : register(b2) - Contains LightUBO
//   StructuredBuffer<PointLight> pointLights : register(t0, space1)
//   StructuredBuffer<SpotLight> spotLights : register(t1, space1)

#ifndef LIGHTS_HLSLI
#define LIGHTS_HLSLI

// Directional light - infinitely distant light source (e.g., sun)
// Size: 32 bytes
struct DirectionalLight
{
    float3 Direction;   // Normalized direction from light to surface
    float  Intensity;   // Light intensity multiplier
    float3 Color;       // RGB color
    float  Padding;     // 16-byte alignment padding
};

// Point light - omnidirectional light from a position
// Size: 32 bytes
struct PointLight
{
    float3 Position;    // World position
    float  Radius;      // Maximum influence radius
    float3 Color;       // RGB color
    float  Intensity;   // Light intensity multiplier
};

// Spot light - cone-shaped light from a position
// Size: 48 bytes
struct SpotLight
{
    float3 Position;        // World position
    float  InnerConeAngle;  // cos(inner angle) - full intensity
    float3 Direction;       // Normalized direction
    float  OuterConeAngle;  // cos(outer angle) - falloff to zero
    float3 Color;           // RGB color
    float  Intensity;       // Light intensity multiplier
};

// Light uniform buffer object
// Bound to register(b2)
struct LightUBO
{
    DirectionalLight DirectionalLightData;
    uint NumPointLights;
    uint NumSpotLights;
    float2 Padding;
};

// ============================================================================
// Light Attenuation Functions
// ============================================================================

// Calculate smooth attenuation for point/spot lights
// Uses inverse square law with smooth falloff at radius boundary
float CalculateAttenuation(float distance, float radius)
{
    // Inverse square law with smooth cutoff
    float attenuation = 1.0 / (distance * distance + 1.0);

    // Smooth falloff at radius boundary
    float falloff = saturate(1.0 - distance / radius);
    falloff = falloff * falloff;

    return attenuation * falloff;
}

// Calculate spot light cone attenuation
// Uses smooth interpolation between inner and outer cone angles
float CalculateSpotAttenuation(float3 lightDir, float3 spotDir, float innerCos, float outerCos)
{
    float cosAngle = dot(-lightDir, spotDir);
    return saturate((cosAngle - outerCos) / (innerCos - outerCos));
}

// ============================================================================
// Blinn-Phong Shading Functions
// ============================================================================

// Calculate Blinn-Phong lighting contribution
// lightDir: normalized direction from surface to light
// viewDir: normalized direction from surface to camera
// normal: normalized surface normal (world space)
// lightColor: RGB color of the light
// albedo: surface diffuse color
// shininess: specular exponent (higher = tighter highlight)
// Returns: diffuse + specular contribution
float3 CalculateBlinnPhong(
    float3 lightDir,
    float3 viewDir,
    float3 normal,
    float3 lightColor,
    float3 albedo,
    float shininess)
{
    // Diffuse (Lambertian)
    float NdotL = max(dot(normal, lightDir), 0.0);
    float3 diffuse = NdotL * lightColor * albedo;

    // No specular if surface faces away from light
    if (NdotL <= 0.0)
        return diffuse;

    // Specular (Blinn-Phong)
    float3 halfDir = normalize(lightDir + viewDir);
    float NdotH = max(dot(normal, halfDir), 0.0);
    float3 specular = pow(NdotH, shininess) * lightColor;

    return diffuse + specular;
}

// Calculate only diffuse component (Lambertian)
float3 CalculateBlinnPhongDiffuse(
    float3 lightDir,
    float3 normal,
    float3 lightColor,
    float3 albedo)
{
    float NdotL = max(dot(normal, lightDir), 0.0);
    return NdotL * lightColor * albedo;
}

// Calculate only specular component (Blinn-Phong)
float3 CalculateBlinnPhongSpecular(
    float3 lightDir,
    float3 viewDir,
    float3 normal,
    float3 lightColor,
    float shininess)
{
    float NdotL = max(dot(normal, lightDir), 0.0);

    // No specular if surface faces away from light
    if (NdotL <= 0.0)
        return float3(0.0, 0.0, 0.0);

    float3 halfDir = normalize(lightDir + viewDir);
    float NdotH = max(dot(normal, halfDir), 0.0);
    return pow(NdotH, shininess) * lightColor;
}

// Convert roughness to shininess for Blinn-Phong
// roughness: 0 (smooth/shiny) to 1 (rough/matte)
// Returns: shininess exponent for pow()
float RoughnessToShininess(float roughness)
{
    // Map roughness 0..1 to shininess 2048..2
    // roughness 0 -> shininess 2048 (mirror-like)
    // roughness 1 -> shininess 2 (very matte)
    float r = clamp(roughness, 0.0, 1.0);
    return lerp(2048.0, 2.0, r);
}

// ============================================================================
// Light Calculation Helpers
// ============================================================================

// Calculate directional light contribution using Blinn-Phong
float3 CalculateDirectionalLight(
    DirectionalLight light,
    float3 normal,
    float3 viewDir,
    float3 albedo,
    float roughness,
    float metallic)
{
    float3 lightDir = normalize(-light.Direction);
    float3 lightColor = light.Color * light.Intensity;
    float shininess = RoughnessToShininess(roughness);

    return CalculateBlinnPhong(lightDir, viewDir, normal, lightColor, albedo, shininess);
}

// Calculate point light contribution using Blinn-Phong
float3 CalculatePointLight(
    PointLight light,
    float3 worldPos,
    float3 normal,
    float3 viewDir,
    float3 albedo,
    float roughness,
    float metallic)
{
    float3 lightVec = light.Position - worldPos;
    float distance = length(lightVec);
    float3 lightDir = lightVec / distance;

    float attenuation = CalculateAttenuation(distance, light.Radius);
    float3 lightColor = light.Color * light.Intensity * attenuation;
    float shininess = RoughnessToShininess(roughness);

    return CalculateBlinnPhong(lightDir, viewDir, normal, lightColor, albedo, shininess);
}

// Calculate spot light contribution using Blinn-Phong
float3 CalculateSpotLight(
    SpotLight light,
    float3 worldPos,
    float3 normal,
    float3 viewDir,
    float3 albedo,
    float roughness,
    float metallic)
{
    float3 lightVec = light.Position - worldPos;
    float distance = length(lightVec);
    float3 lightDir = lightVec / distance;

    // Distance attenuation (using radius from intensity for simplicity)
    float radius = 50.0; // Default radius for spot lights
    float distanceAttenuation = CalculateAttenuation(distance, radius);

    // Spot cone attenuation
    float spotAttenuation = CalculateSpotAttenuation(
        lightDir,
        normalize(light.Direction),
        light.InnerConeAngle,
        light.OuterConeAngle);

    float3 lightColor = light.Color * light.Intensity * distanceAttenuation * spotAttenuation;
    float shininess = RoughnessToShininess(roughness);

    return CalculateBlinnPhong(lightDir, viewDir, normal, lightColor, albedo, shininess);
}

#endif // LIGHTS_HLSLI
