// Irradiance Map Generation Compute Shader
// Generates a diffuse irradiance map from an HDR environment cubemap
// Used for Image-Based Lighting (IBL) diffuse component
//
// The irradiance map stores the precomputed integral of incoming radiance
// over the hemisphere weighted by cos(theta) for Lambertian diffuse reflection.

#define PI 3.14159265359

// Input environment cubemap
[[vk::binding(0, 0)]] TextureCube<float4> EnvironmentMap : register(t0);
[[vk::binding(1, 0)]] SamplerState LinearSampler : register(s0);

// Output irradiance cubemap (6 faces as array)
[[vk::binding(2, 0)]] RWTexture2DArray<float4> IrradianceMap : register(u0);

// Push constants for irradiance map size
[[vk::push_constant]]
struct PushConstants
{
    uint IrradianceMapSize;
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

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint2 pixelCoord = dispatchThreadId.xy;
    uint faceIndex = dispatchThreadId.z;

    // Check bounds
    if (pixelCoord.x >= pushConstants.IrradianceMapSize ||
        pixelCoord.y >= pushConstants.IrradianceMapSize ||
        faceIndex >= 6)
    {
        return;
    }

    // Calculate UV coordinates for this pixel (center of pixel)
    float2 uv = (float2(pixelCoord) + 0.5) / float(pushConstants.IrradianceMapSize);

    // Get the surface normal direction for this cubemap texel
    float3 N = GetCubemapDirection(faceIndex, uv);

    // Build a tangent-space basis around N
    // Choose an up vector that is not parallel to N
    float3 up = abs(N.y) < 0.999 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
    float3 right = normalize(cross(up, N));
    up = normalize(cross(N, right));

    // Integrate over the hemisphere using uniform sampling
    // The irradiance integral is: E = integral over hemisphere of Li * cos(theta) * dw
    // For discrete sampling: E = (1/N) * sum(Li * cos(theta))
    // With spherical coordinates: dw = sin(theta) * dtheta * dphi
    // Final: E = PI * (1/N) * sum(Li * cos(theta) * sin(theta))
    float3 irradiance = float3(0.0, 0.0, 0.0);

    // Sample step for hemisphere integration
    // Smaller values give more accurate results but are slower
    float sampleDelta = 0.025;
    float sampleCount = 0.0;

    // Integrate over hemisphere (theta: 0 to PI/2, phi: 0 to 2*PI)
    for (float phi = 0.0; phi < 2.0 * PI; phi += sampleDelta)
    {
        for (float theta = 0.0; theta < 0.5 * PI; theta += sampleDelta)
        {
            // Convert spherical coordinates to Cartesian (in tangent space)
            // x = sin(theta) * cos(phi)
            // y = sin(theta) * sin(phi)
            // z = cos(theta) (aligned with N)
            float sinTheta = sin(theta);
            float cosTheta = cos(theta);
            float cosPhi = cos(phi);
            float sinPhi = sin(phi);

            float3 tangentSample = float3(
                sinTheta * cosPhi,
                sinTheta * sinPhi,
                cosTheta
            );

            // Transform from tangent space to world space
            float3 sampleVec = tangentSample.x * right +
                               tangentSample.y * up +
                               tangentSample.z * N;

            // Sample the environment map
            float3 sampleColor = EnvironmentMap.SampleLevel(LinearSampler, sampleVec, 0).rgb;

            // Weight by cos(theta) * sin(theta):
            // - cos(theta): Lambertian BRDF (n dot l)
            // - sin(theta): Jacobian for spherical coordinates (solid angle differential)
            irradiance += sampleColor * cosTheta * sinTheta;
            sampleCount += 1.0;
        }
    }

    // Normalize and apply PI factor
    // The integral of cos(theta) * sin(theta) over hemisphere is PI
    // So we multiply by PI and divide by sample count
    irradiance = PI * irradiance / sampleCount;

    // Write to the irradiance cubemap
    IrradianceMap[uint3(pixelCoord, faceIndex)] = float4(irradiance, 1.0);
}
