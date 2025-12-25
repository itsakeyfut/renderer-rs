// Equirectangular to Cubemap Conversion Compute Shader
// Converts an HDR equirectangular environment map to a cubemap texture

#define PI 3.14159265359

// Input equirectangular HDR texture
[[vk::binding(0, 0)]] Texture2D<float4> EquirectTexture : register(t0);
[[vk::binding(1, 0)]] SamplerState LinearSampler : register(s0);

// Output cubemap (6 faces as array)
[[vk::binding(2, 0)]] RWTexture2DArray<float4> OutputCubemap : register(u0);

// Push constants for cubemap size
[[vk::push_constant]]
struct PushConstants
{
    uint CubemapSize;
} pushConstants;

// Convert cubemap face coordinates to 3D direction vector
// Face indices: 0=+X, 1=-X, 2=+Y, 3=-Y, 4=+Z, 5=-Z
float3 GetCubemapDirection(uint face, float2 uv)
{
    // Convert UV [0,1] to [-1,1] for direction calculation
    float u = uv.x * 2.0 - 1.0;
    float v = uv.y * 2.0 - 1.0;

    float3 dir;

    switch (face)
    {
        case 0: // +X (right)
            dir = float3(1.0, -v, -u);
            break;
        case 1: // -X (left)
            dir = float3(-1.0, -v, u);
            break;
        case 2: // +Y (top)
            dir = float3(u, 1.0, v);
            break;
        case 3: // -Y (bottom)
            dir = float3(u, -1.0, -v);
            break;
        case 4: // +Z (front)
            dir = float3(u, -v, 1.0);
            break;
        case 5: // -Z (back)
            dir = float3(-u, -v, -1.0);
            break;
        default:
            dir = float3(0.0, 0.0, 1.0);
            break;
    }

    return normalize(dir);
}

// Convert 3D direction to equirectangular UV coordinates
float2 DirectionToEquirectUV(float3 dir)
{
    // Calculate spherical coordinates
    // phi: azimuthal angle around Y axis (longitude)
    // theta: polar angle from Y axis (latitude)
    float phi = atan2(dir.z, dir.x);
    float theta = asin(clamp(dir.y, -1.0, 1.0));

    // Map to UV coordinates
    // phi: [-PI, PI] -> [0, 1]
    // theta: [-PI/2, PI/2] -> [0, 1]
    float2 uv;
    uv.x = (phi + PI) / (2.0 * PI);
    uv.y = (theta + PI * 0.5) / PI;

    return uv;
}

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint2 pixelCoord = dispatchThreadId.xy;
    uint faceIndex = dispatchThreadId.z;

    // Check bounds
    if (pixelCoord.x >= pushConstants.CubemapSize ||
        pixelCoord.y >= pushConstants.CubemapSize ||
        faceIndex >= 6)
    {
        return;
    }

    // Calculate UV coordinates for this pixel (center of pixel)
    float2 uv = (float2(pixelCoord) + 0.5) / float(pushConstants.CubemapSize);

    // Get the 3D direction for this cubemap texel
    float3 direction = GetCubemapDirection(faceIndex, uv);

    // Convert direction to equirectangular UV
    float2 equirectUV = DirectionToEquirectUV(direction);

    // Sample the equirectangular texture
    float4 color = EquirectTexture.SampleLevel(LinearSampler, equirectUV, 0);

    // Write to the cubemap face
    OutputCubemap[uint3(pixelCoord, faceIndex)] = color;
}
