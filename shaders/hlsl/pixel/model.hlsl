// Model Pixel (Fragment) Shader
// Blinn-Phong shading with Normal Mapping support
//
// This shader uses hardcoded fallback values for lighting and materials
// until the renderer binds LightUBO (b2) and MaterialData (b3).

#include "../lights.hlsli"

// Camera uniform buffer
cbuffer CameraData : register(b0)
{
    float4x4 view;
    float4x4 projection;
    float4x4 viewProjection;
    float3 cameraPosition;
    float cameraPadding;
};

struct PSInput
{
    float4 Position    : SV_POSITION;
    float3 WorldPos    : TEXCOORD0;
    float3 Normal      : TEXCOORD1;
    float2 TexCoord    : TEXCOORD2;
    float3 Tangent     : TEXCOORD3;
    float3 Bitangent   : TEXCOORD4;
};

float4 main(PSInput input) : SV_TARGET
{
    // =========================================================================
    // Fallback values (until proper UBOs and textures are bound)
    // =========================================================================

    // Default material properties
    float3 albedo = float3(0.7, 0.7, 0.7);  // Gray base color
    float roughness = 0.5;                    // Medium roughness
    float ambientOcclusion = 1.0;             // No occlusion

    // Default directional light (sun-like, from upper-right-front)
    float3 lightDirection = normalize(float3(1.0, 1.0, 1.0));
    float3 lightColor = float3(1.0, 1.0, 1.0);
    float lightIntensity = 1.0;

    // =========================================================================
    // Normal calculation
    // =========================================================================

    // Use interpolated vertex normal (no normal mapping without texture)
    float3 N = normalize(input.Normal);

    // View direction (from surface to camera)
    float3 V = normalize(cameraPosition - input.WorldPos);

    // =========================================================================
    // Blinn-Phong Lighting
    // =========================================================================

    // Ambient term
    float3 ambient = 0.03 * albedo * ambientOcclusion;

    // Calculate shininess from roughness
    float shininess = RoughnessToShininess(roughness);

    // Directional light contribution using Blinn-Phong
    float3 lighting = CalculateBlinnPhong(
        lightDirection,
        V,
        N,
        lightColor * lightIntensity,
        albedo,
        shininess
    );

    // =========================================================================
    // Final color composition
    // =========================================================================

    float3 color = ambient + lighting;

    return float4(color, 1.0);
}
