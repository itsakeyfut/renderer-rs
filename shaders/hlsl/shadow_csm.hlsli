// Cascaded Shadow Map (CSM) sampling utilities for HLSL shaders
// Extends shadow.hlsli with multi-cascade shadow sampling support.
//
// Usage:
//   #include "shadow_csm.hlsli"
//
// Required bindings:
//   Texture2DArray<float> shadowMapArray + SamplerComparisonState
//   cbuffer CSMData with CSMParams
//
// Reference:
//   - "Parallel-Split Shadow Maps on Programmable GPUs" (Zhang et al.)
//   - UE: FShadowMapRenderTargets for CSM management

#ifndef SHADOW_CSM_HLSLI
#define SHADOW_CSM_HLSLI

// Number of cascades for CSM
#define CASCADE_COUNT 4

// Per-cascade data for CSM
// Size: 80 bytes (aligned to match C++ CascadeData struct)
struct CascadeData
{
    float4x4 ViewProjection;    // Light-space view-projection matrix for this cascade
    float    SplitDepth;        // Split depth in clip space
    float3   Padding;           // Padding for std140 alignment
};

// CSM parameters uniform buffer
// Size: 336 bytes (aligned to match C++ CascadedShadowMapUBO struct)
struct CSMParams
{
    CascadeData Cascades[CASCADE_COUNT];  // Per-cascade data
    float       ShadowBias;               // Depth bias for shadow acne prevention
    float       NormalBias;               // Normal-based bias offset
    float       ShadowMapSize;            // Shadow map size (width = height)
    float       Padding;                  // Padding for alignment
};

// ============================================================================
// Cascaded Shadow Map (CSM) Functions
// ============================================================================

/**
 * @brief Select the appropriate cascade based on fragment depth
 *
 * Compares fragment clip-space depth against cascade split depths
 * to determine which cascade to sample from.
 *
 * @param csmParams      CSM parameters with cascade split depths
 * @param clipSpaceDepth Fragment depth in clip space (SV_Position.z)
 * @return Cascade index (0 to CASCADE_COUNT-1)
 */
uint SelectCascade(CSMParams csmParams, float clipSpaceDepth)
{
    uint cascadeIndex = 0;

    // Find the cascade that contains this depth
    // Cascades are sorted by increasing depth (near to far)
    [unroll]
    for (uint i = 0; i < CASCADE_COUNT - 1; ++i)
    {
        if (clipSpaceDepth > csmParams.Cascades[i].SplitDepth)
        {
            cascadeIndex = i + 1;
        }
    }

    return cascadeIndex;
}

/**
 * @brief Sample shadow from a specific cascade with PCF
 *
 * Internal helper function for CSM shadow sampling.
 *
 * @param shadowMapArray  Shadow map array texture
 * @param shadowSampler   Comparison sampler
 * @param lightMatrix     Light-space view-projection matrix for the cascade
 * @param cascadeIndex    Index of the cascade to sample
 * @param worldPos        Fragment world position
 * @param normal          Fragment normal (world space)
 * @param lightDir        Light direction (toward light source)
 * @param bias            Shadow bias
 * @param normalBias      Normal-based bias offset
 * @param texelSize       1.0 / shadow map size
 * @return Shadow factor: 1.0 = fully lit, 0.0 = fully shadowed
 */
float SampleCascadePCF(
    Texture2DArray<float> shadowMapArray,
    SamplerComparisonState shadowSampler,
    float4x4 lightMatrix,
    uint cascadeIndex,
    float3 worldPos,
    float3 normal,
    float3 lightDir,
    float bias,
    float normalBias,
    float texelSize)
{
    // Apply normal bias offset
    float3 offsetPos = worldPos + normal * normalBias;

    // Transform to light space
    float4 fragPosLightSpace = mul(lightMatrix, float4(offsetPos, 1.0));
    float3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;

    // Transform to texture coordinates
    projCoords.xy = projCoords.xy * 0.5 + 0.5;
    projCoords.y = 1.0 - projCoords.y;

    // Bounds check
    if (projCoords.x < 0.0 || projCoords.x > 1.0 ||
        projCoords.y < 0.0 || projCoords.y > 1.0 ||
        projCoords.z < 0.0 || projCoords.z > 1.0)
    {
        return 1.0;
    }

    // Calculate adaptive bias
    float NdotL = dot(normal, lightDir);
    float adaptiveBias = max(bias * (1.0 - NdotL), 0.0005);
    float currentDepth = projCoords.z - adaptiveBias;

    // PCF: 3x3 kernel
    float shadow = 0.0;

    [unroll]
    for (int x = -1; x <= 1; ++x)
    {
        [unroll]
        for (int y = -1; y <= 1; ++y)
        {
            float2 offset = float2(x, y) * texelSize;
            float3 sampleUVW = float3(projCoords.xy + offset, (float)cascadeIndex);
            shadow += shadowMapArray.SampleCmpLevelZero(
                shadowSampler,
                sampleUVW,
                currentDepth
            );
        }
    }

    return shadow / 9.0;
}

/**
 * @brief Calculate shadow factor using Cascaded Shadow Maps (CSM)
 *
 * Automatically selects the appropriate cascade based on fragment depth
 * and samples with PCF for soft shadow edges.
 *
 * @param shadowMapArray  Shadow map array texture (4 layers)
 * @param shadowSampler   Comparison sampler
 * @param csmParams       CSM parameters (cascade matrices, split depths, bias)
 * @param worldPos        Fragment world position
 * @param normal          Fragment normal (world space, normalized)
 * @param lightDir        Light direction (normalized, toward light source)
 * @param clipSpaceDepth  Fragment depth in clip space (from SV_Position.z)
 * @return Shadow factor: 1.0 = fully lit, 0.0 = fully shadowed
 */
float CalculateShadowCSM(
    Texture2DArray<float> shadowMapArray,
    SamplerComparisonState shadowSampler,
    CSMParams csmParams,
    float3 worldPos,
    float3 normal,
    float3 lightDir,
    float clipSpaceDepth)
{
    // Select cascade based on depth
    uint cascadeIndex = SelectCascade(csmParams, clipSpaceDepth);

    // Get cascade-specific data
    float4x4 lightMatrix = csmParams.Cascades[cascadeIndex].ViewProjection;

    // Calculate texel size
    float texelSize = 1.0 / csmParams.ShadowMapSize;

    // Sample shadow with PCF
    return SampleCascadePCF(
        shadowMapArray,
        shadowSampler,
        lightMatrix,
        cascadeIndex,
        worldPos,
        normal,
        lightDir,
        csmParams.ShadowBias,
        csmParams.NormalBias,
        texelSize
    );
}

/**
 * @brief Calculate shadow factor with cascade blending for smooth transitions
 *
 * Blends between adjacent cascades at cascade boundaries to avoid
 * visible seams when transitioning between cascade levels.
 *
 * @param shadowMapArray  Shadow map array texture (4 layers)
 * @param shadowSampler   Comparison sampler
 * @param csmParams       CSM parameters
 * @param worldPos        Fragment world position
 * @param normal          Fragment normal (world space, normalized)
 * @param lightDir        Light direction (normalized, toward light source)
 * @param clipSpaceDepth  Fragment depth in clip space
 * @param blendThreshold  Blend region size (e.g., 0.1 = 10% of cascade range)
 * @return Shadow factor: 1.0 = fully lit, 0.0 = fully shadowed
 */
float CalculateShadowCSMBlended(
    Texture2DArray<float> shadowMapArray,
    SamplerComparisonState shadowSampler,
    CSMParams csmParams,
    float3 worldPos,
    float3 normal,
    float3 lightDir,
    float clipSpaceDepth,
    float blendThreshold)
{
    // Select primary cascade
    uint cascadeIndex = SelectCascade(csmParams, clipSpaceDepth);

    float texelSize = 1.0 / csmParams.ShadowMapSize;

    // Sample from primary cascade
    float shadow = SampleCascadePCF(
        shadowMapArray,
        shadowSampler,
        csmParams.Cascades[cascadeIndex].ViewProjection,
        cascadeIndex,
        worldPos,
        normal,
        lightDir,
        csmParams.ShadowBias,
        csmParams.NormalBias,
        texelSize
    );

    // Check if we should blend with the next cascade
    if (cascadeIndex < CASCADE_COUNT - 1)
    {
        float splitDepth = csmParams.Cascades[cascadeIndex].SplitDepth;
        float prevSplit = (cascadeIndex > 0) ?
            csmParams.Cascades[cascadeIndex - 1].SplitDepth : 0.0;
        float cascadeRange = splitDepth - prevSplit;
        float blendRegion = cascadeRange * blendThreshold;

        // Calculate blend factor for transition region
        float distanceToEdge = splitDepth - clipSpaceDepth;

        if (distanceToEdge < blendRegion && distanceToEdge > 0.0)
        {
            float blendFactor = distanceToEdge / blendRegion;

            // Sample from next cascade
            float nextShadow = SampleCascadePCF(
                shadowMapArray,
                shadowSampler,
                csmParams.Cascades[cascadeIndex + 1].ViewProjection,
                cascadeIndex + 1,
                worldPos,
                normal,
                lightDir,
                csmParams.ShadowBias,
                csmParams.NormalBias,
                texelSize
            );

            // Blend between cascades
            shadow = lerp(nextShadow, shadow, blendFactor);
        }
    }

    return shadow;
}

/**
 * @brief Get cascade index for debugging/visualization
 *
 * Returns a color based on which cascade the fragment falls into.
 * Useful for visualizing cascade distribution.
 *
 * @param csmParams      CSM parameters
 * @param clipSpaceDepth Fragment depth in clip space
 * @return Debug color (R=cascade 0, G=cascade 1, B=cascade 2, Y=cascade 3)
 */
float3 GetCascadeDebugColor(CSMParams csmParams, float clipSpaceDepth)
{
    uint cascadeIndex = SelectCascade(csmParams, clipSpaceDepth);

    float3 colors[CASCADE_COUNT] = {
        float3(1.0, 0.0, 0.0),  // Red - Cascade 0 (nearest)
        float3(0.0, 1.0, 0.0),  // Green - Cascade 1
        float3(0.0, 0.0, 1.0),  // Blue - Cascade 2
        float3(1.0, 1.0, 0.0)   // Yellow - Cascade 3 (farthest)
    };

    return colors[cascadeIndex];
}

#endif // SHADOW_CSM_HLSLI
