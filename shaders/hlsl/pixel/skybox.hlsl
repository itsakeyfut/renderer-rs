// Skybox Pixel Shader
// Samples the environment cubemap and outputs the sky color

// Environment cubemap
[[vk::binding(0, 0)]] TextureCube<float4> EnvironmentMap : register(t0);
[[vk::binding(1, 0)]] SamplerState CubemapSampler : register(s0);

struct PSInput
{
    float4 Position : SV_POSITION;
    float3 LocalPos : TEXCOORD0;
};

struct PSOutput
{
    float4 Color : SV_TARGET0;
};

PSOutput main(PSInput input)
{
    PSOutput output;

    // Sample the cubemap using the direction vector
    float3 direction = normalize(input.LocalPos);
    float4 envColor = EnvironmentMap.Sample(CubemapSampler, direction);

    // Output the environment color
    // Note: For HDR environment maps, tone mapping should be applied
    // in a post-processing pass, not here
    output.Color = envColor;

    return output;
}
