// Model Pixel (Fragment) Shader - Full Version
// Blinn-Phong shading with Normal Mapping and full light system support
//
// Requirements:
//   - LightData cbuffer (b2) with LightUBO
//   - MaterialData cbuffer (b3) with material properties
//   - albedoMap texture (t0)
//   - normalMap texture (t1)
//   - linearSampler (s0)
//   - pointLights StructuredBuffer (t0, space1)
//   - spotLights StructuredBuffer (t1, space1)
//
// Use this shader when the renderer has full descriptor set bindings ready.

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

// Light uniform buffer
cbuffer LightData : register(b2)
{
    LightUBO lightUBO;
};

// Material uniform buffer
cbuffer MaterialData : register(b3)
{
    float4 baseColor;
    float  metallic;
    float  roughness;
    float  ambientOcclusion;
    float  materialPadding;
};

// Texture samplers
Texture2D albedoMap : register(t0);
Texture2D normalMap : register(t1);
SamplerState linearSampler : register(s0);

// Point and spot light buffers
StructuredBuffer<PointLight> pointLights : register(t0, space1);
StructuredBuffer<SpotLight> spotLights : register(t1, space1);

struct PSInput
{
    float4 Position    : SV_POSITION;
    float3 WorldPos    : TEXCOORD0;
    float3 Normal      : TEXCOORD1;
    float2 TexCoord    : TEXCOORD2;
    float3 Tangent     : TEXCOORD3;
    float3 Bitangent   : TEXCOORD4;
};

// Sample and transform normal from normal map to world space
float3 GetWorldNormal(PSInput input, bool hasNormalMap)
{
    float3 N = normalize(input.Normal);

    if (!hasNormalMap)
        return N;

    // Sample normal map (tangent space normal)
    float3 normalSample = normalMap.Sample(linearSampler, input.TexCoord).rgb;

    // Convert from [0,1] to [-1,1]
    normalSample = normalSample * 2.0 - 1.0;

    // Build TBN matrix (Tangent, Bitangent, Normal)
    float3 T = normalize(input.Tangent);
    float3 B = normalize(input.Bitangent);
    float3x3 TBN = float3x3(T, B, N);

    // Transform from tangent space to world space
    return normalize(mul(normalSample, TBN));
}

float4 main(PSInput input) : SV_TARGET
{
    // Sample albedo texture
    float4 albedoSample = albedoMap.Sample(linearSampler, input.TexCoord);
    float3 albedo = albedoSample.rgb * baseColor.rgb;

    // Check if normal map is bound by sampling and checking if it's valid
    // (A default 1x1 white texture is used when no normal map is bound)
    float3 normalCheck = normalMap.Sample(linearSampler, input.TexCoord).rgb;
    bool hasNormalMap = length(normalCheck - float3(1.0, 1.0, 1.0)) > 0.01;

    // Get world space normal (with normal mapping if available)
    float3 N = GetWorldNormal(input, hasNormalMap);

    // View direction (from surface to camera)
    float3 V = normalize(cameraPosition - input.WorldPos);

    // Ambient lighting
    float3 ambient = 0.03 * albedo * ambientOcclusion;

    // Accumulate lighting
    float3 lighting = float3(0.0, 0.0, 0.0);

    // Directional light
    lighting += CalculateDirectionalLight(
        lightUBO.DirectionalLightData,
        N,
        V,
        albedo,
        roughness,
        metallic
    );

    // Point lights
    for (uint i = 0; i < lightUBO.NumPointLights; i++)
    {
        lighting += CalculatePointLight(
            pointLights[i],
            input.WorldPos,
            N,
            V,
            albedo,
            roughness,
            metallic
        );
    }

    // Spot lights
    for (uint j = 0; j < lightUBO.NumSpotLights; j++)
    {
        lighting += CalculateSpotLight(
            spotLights[j],
            input.WorldPos,
            N,
            V,
            albedo,
            roughness,
            metallic
        );
    }

    // Final color = ambient + all lighting contributions
    float3 color = ambient + lighting;

    return float4(color, albedoSample.a * baseColor.a);
}
