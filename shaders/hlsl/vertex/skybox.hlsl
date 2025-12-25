// Skybox Vertex Shader
// Renders a fullscreen triangle and calculates cubemap sampling direction

// Push constants for inverse view-projection (rotation only)
[[vk::push_constant]]
struct PushConstants
{
    float4x4 inverseViewProjection;
} pushConstants;

struct VSOutput
{
    float4 Position : SV_POSITION;
    float3 LocalPos : TEXCOORD0;
};

// Generate fullscreen triangle vertices
// Uses vertex ID to generate positions (no vertex buffer needed)
// Triangle covers the entire screen with only 3 vertices
static const float2 positions[3] = {
    float2(-1.0, -1.0),
    float2( 3.0, -1.0),
    float2(-1.0,  3.0)
};

VSOutput main(uint vertexID : SV_VertexID)
{
    VSOutput output;

    // Generate fullscreen triangle position
    float2 pos = positions[vertexID];

    // Place at far plane so skybox is behind everything
    // Using 1.0 for z gives maximum depth (far plane)
    output.Position = float4(pos, 1.0, 1.0);

    // Transform NDC position to world direction
    // We use z=1 in clip space to get a point on the far plane
    // Flip Y because Vulkan has Y pointing down in NDC
    float4 clipPos = float4(pos.x, -pos.y, 1.0, 1.0);
    float4 worldPos = mul(pushConstants.inverseViewProjection, clipPos);
    output.LocalPos = worldPos.xyz / worldPos.w;

    return output;
}
