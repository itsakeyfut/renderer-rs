// Shadow Depth Pixel (Fragment) Shader
// Empty shader for depth-only shadow pass.
// No color output is produced - only depth values are written by the rasterizer.
// This enables fast shadow map generation with Early-Z optimization.

struct PSInput
{
    float4 Position : SV_POSITION;
};

// No output - depth-only pass
// The depth value is automatically written by the rasterizer
void main(PSInput input)
{
    // Intentionally empty
    // Depth is written automatically without fragment shader output
}
