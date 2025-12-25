// Shadow Depth Vertex Shader
// Transforms vertices to light space for shadow map generation.
// Used in the depth-only shadow pass where only depth values are written.

// Shadow pass uniform buffer
// Contains matrices for transforming objects to light clip space
cbuffer ShadowConstants : register(b0)
{
    float4x4 lightSpaceMatrix;  // Combined light view-projection matrix
    float4x4 model;             // Object world transform
};

struct VSInput
{
    float3 Position : POSITION;
};

struct VSOutput
{
    float4 Position : SV_POSITION;
};

VSOutput main(VSInput input)
{
    VSOutput output;

    // Transform vertex position to light clip space
    // First apply model transform, then light space (view-projection)
    float4 worldPos = mul(model, float4(input.Position, 1.0));
    output.Position = mul(lightSpaceMatrix, worldPos);

    return output;
}
