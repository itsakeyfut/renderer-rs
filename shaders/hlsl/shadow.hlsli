// Shadow sampling utilities for HLSL shaders
// Provides PCF (Percentage Closer Filtering) for soft shadows.
//
// Usage:
//   #include "shadow.hlsli"
//
// Required bindings (typically in light data set):
//   Texture2D<float> shadowMap + SamplerComparisonState
//   cbuffer ShadowData with ShadowParams
//
// Reference:
//   - Shadow acne prevention with bias
//   - PCF soft shadow sampling
//   - SamplerComparisonState for hardware depth comparison

#ifndef SHADOW_HLSLI
#define SHADOW_HLSLI

// Shadow parameters uniform buffer
// Size: 96 bytes (aligned for std140)
struct ShadowParams
{
    float4x4 LightSpaceMatrix;  // Combined light view-projection matrix
    float    ShadowBias;        // Depth bias for shadow acne prevention
    float    NormalBias;        // Normal-based bias offset
    float2   ShadowMapSize;     // Shadow map dimensions (width, height)
    float    ShadowStrength;    // Shadow darkness (0=no shadow, 1=full shadow)
    float3   Padding;           // Padding for std140 alignment
};

// ============================================================================
// Shadow Sampling Functions
// ============================================================================

/**
 * @brief Calculate shadow factor using PCF (Percentage Closer Filtering)
 *
 * Transforms world position to light space and samples the shadow map
 * with a 3x3 PCF kernel for soft shadow edges.
 *
 * @param shadowMap       Shadow depth texture
 * @param shadowSampler   Comparison sampler (VK_COMPARE_OP_LESS_OR_EQUAL)
 * @param shadowParams    Shadow parameters (light matrix, bias, map size)
 * @param worldPos        Fragment world position
 * @param normal          Fragment normal (world space, normalized)
 * @param lightDir        Light direction (normalized, toward light source)
 * @return Shadow factor: 1.0 = fully lit, 0.0 = fully shadowed
 */
float CalculateShadow(
    Texture2D<float> shadowMap,
    SamplerComparisonState shadowSampler,
    ShadowParams shadowParams,
    float3 worldPos,
    float3 normal,
    float3 lightDir)
{
    // Transform world position to light clip space
    float4 fragPosLightSpace = mul(shadowParams.LightSpaceMatrix, float4(worldPos, 1.0));

    // Perspective divide (orthographic projection: w = 1.0)
    float3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;

    // Transform from NDC [-1,1] to texture coordinates [0,1]
    // Vulkan: Y is inverted compared to OpenGL
    projCoords.xy = projCoords.xy * 0.5 + 0.5;
    projCoords.y = 1.0 - projCoords.y;  // Flip Y for Vulkan UV convention

    // Check if outside shadow map bounds
    // Return fully lit (1.0) for fragments outside the shadow frustum
    if (projCoords.x < 0.0 || projCoords.x > 1.0 ||
        projCoords.y < 0.0 || projCoords.y > 1.0 ||
        projCoords.z < 0.0 || projCoords.z > 1.0)
    {
        return 1.0;
    }

    // Calculate adaptive bias based on surface angle to light
    // Steeper angles require larger bias to prevent shadow acne
    float NdotL = dot(normal, lightDir);
    float bias = max(shadowParams.ShadowBias * (1.0 - NdotL), 0.0005);

    // Apply normal bias: offset sample position along surface normal
    // This helps prevent self-shadowing on curved surfaces
    float3 offsetPos = worldPos + normal * shadowParams.NormalBias;
    float4 offsetPosLightSpace = mul(shadowParams.LightSpaceMatrix, float4(offsetPos, 1.0));
    float3 offsetProjCoords = offsetPosLightSpace.xyz / offsetPosLightSpace.w;
    offsetProjCoords.xy = offsetProjCoords.xy * 0.5 + 0.5;
    offsetProjCoords.y = 1.0 - offsetProjCoords.y;

    // Use offset coordinates for both sampling and depth comparison
    float2 sampleCoords = offsetProjCoords.xy;
    float currentDepth = offsetProjCoords.z - bias;

    // PCF: 3x3 kernel sampling for soft shadow edges
    // SampleCmpLevelZero performs hardware depth comparison
    // Returns 0.0 if depth test fails (in shadow), 1.0 if passes (lit)
    float shadow = 0.0;
    float2 texelSize = 1.0 / shadowParams.ShadowMapSize;

    [unroll]
    for (int x = -1; x <= 1; ++x)
    {
        [unroll]
        for (int y = -1; y <= 1; ++y)
        {
            float2 offset = float2(x, y) * texelSize;
            shadow += shadowMap.SampleCmpLevelZero(
                shadowSampler,
                sampleCoords + offset,
                currentDepth
            );
        }
    }

    // Average the 9 samples
    shadow /= 9.0;

    // Apply shadow strength: lerp between fully lit (1.0) and shadow value
    // ShadowStrength=1.0 gives full shadow darkness, 0.0 gives no shadows
    return lerp(1.0, shadow, shadowParams.ShadowStrength);
}

/**
 * @brief Calculate shadow factor with simple bias (no PCF)
 *
 * Faster variant without PCF filtering. Produces hard shadow edges.
 *
 * @param shadowMap       Shadow depth texture
 * @param shadowSampler   Comparison sampler
 * @param shadowParams    Shadow parameters
 * @param worldPos        Fragment world position
 * @param normal          Fragment normal (world space, normalized)
 * @param lightDir        Light direction (normalized, toward light source)
 * @return Shadow factor: 1.0 = fully lit, 0.0 = fully shadowed
 */
float CalculateShadowHard(
    Texture2D<float> shadowMap,
    SamplerComparisonState shadowSampler,
    ShadowParams shadowParams,
    float3 worldPos,
    float3 normal,
    float3 lightDir)
{
    // Transform world position to light clip space
    float4 fragPosLightSpace = mul(shadowParams.LightSpaceMatrix, float4(worldPos, 1.0));

    // Perspective divide
    float3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;

    // Transform from NDC to texture coordinates
    projCoords.xy = projCoords.xy * 0.5 + 0.5;
    projCoords.y = 1.0 - projCoords.y;

    // Bounds check
    if (projCoords.x < 0.0 || projCoords.x > 1.0 ||
        projCoords.y < 0.0 || projCoords.y > 1.0 ||
        projCoords.z < 0.0 || projCoords.z > 1.0)
    {
        return 1.0;
    }

    // Calculate adaptive bias based on surface angle to light
    float NdotL = dot(normal, lightDir);
    float bias = max(shadowParams.ShadowBias * (1.0 - NdotL), 0.0005);

    // Apply normal bias: offset sample position along surface normal
    float3 offsetPos = worldPos + normal * shadowParams.NormalBias;
    float4 offsetPosLightSpace = mul(shadowParams.LightSpaceMatrix, float4(offsetPos, 1.0));
    float3 offsetProjCoords = offsetPosLightSpace.xyz / offsetPosLightSpace.w;
    offsetProjCoords.xy = offsetProjCoords.xy * 0.5 + 0.5;
    offsetProjCoords.y = 1.0 - offsetProjCoords.y;

    float currentDepth = offsetProjCoords.z - bias;

    // Single sample (hard shadows)
    float shadow = shadowMap.SampleCmpLevelZero(shadowSampler, offsetProjCoords.xy, currentDepth);

    // Apply shadow strength
    return lerp(1.0, shadow, shadowParams.ShadowStrength);
}

#endif // SHADOW_HLSLI
