// Prefiltered Environment Map Generation Compute Shader
// Generates a pre-filtered environment map for specular IBL
// Each mip level stores the convolved environment for a specific roughness value
//
// Uses GGX importance sampling with Hammersley sequence for efficient sampling.
// Roughness = 0 at mip 0 (mirror reflection), increasing to 1 at max mip (fully rough)
//
// Reference: "Real Shading in Unreal Engine 4" - Brian Karis, Epic Games

#define PI 3.14159265359

// Input environment cubemap
[[vk::binding(0, 0)]] TextureCube<float4> EnvironmentMap : register(t0);
[[vk::binding(1, 0)]] SamplerState LinearSampler : register(s0);

// Output prefiltered cubemap (single mip level as 6-layer array)
[[vk::binding(2, 0)]] RWTexture2DArray<float4> PrefilteredMap : register(u0);

// Push constants for prefilter parameters
[[vk::push_constant]]
struct PushConstants
{
    uint MipSize;         // Size of current mip level
    float Roughness;      // Roughness value for this mip level
    uint SampleCount;     // Number of samples for convolution
    uint SourceMipLevel;  // Source mip level for sampling (0 for base cubemap)
} pushConstants;

// Van der Corput radical inverse for Hammersley sequence
// Bit manipulation for efficient quasi-random sequence generation
float RadicalInverse_VdC(uint bits)
{
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10; // / 0x100000000
}

// Hammersley 2D sequence - low-discrepancy quasi-random sampling
// Returns a 2D point in [0,1]^2 for importance sampling
float2 Hammersley(uint i, uint N)
{
    return float2(float(i) / float(N), RadicalInverse_VdC(i));
}

// Importance sample GGX distribution
// Returns a half vector H distributed according to GGX NDF
//
// Parameters:
//   Xi: 2D quasi-random sample point in [0,1]^2
//   N: Surface normal direction
//   roughness: Surface roughness (0 = smooth, 1 = rough)
float3 ImportanceSampleGGX(float2 Xi, float3 N, float roughness)
{
    // Disney's remapping: alpha = roughness^2
    float a = roughness * roughness;

    // Convert uniform random sample to spherical coordinates
    // Using the inverse CDF of GGX distribution for importance sampling
    float phi = 2.0 * PI * Xi.x;
    float cosTheta = sqrt((1.0 - Xi.y) / (1.0 + (a * a - 1.0) * Xi.y));
    float sinTheta = sqrt(1.0 - cosTheta * cosTheta);

    // Spherical to Cartesian (half vector in tangent space)
    float3 H;
    H.x = cos(phi) * sinTheta;
    H.y = sin(phi) * sinTheta;
    H.z = cosTheta;

    // Build tangent space basis from normal
    // Choose up vector that is not parallel to N
    float3 up = abs(N.z) < 0.999 ? float3(0.0, 0.0, 1.0) : float3(1.0, 0.0, 0.0);
    float3 tangent = normalize(cross(up, N));
    float3 bitangent = cross(N, tangent);

    // Transform from tangent space to world space
    float3 sampleVec = tangent * H.x + bitangent * H.y + N * H.z;
    return normalize(sampleVec);
}

// GGX Normal Distribution Function for PDF calculation
float DistributionGGX(float NdotH, float roughness)
{
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH2 = NdotH * NdotH;

    float denom = NdotH2 * (a2 - 1.0) + 1.0;
    denom = PI * denom * denom;

    return a2 / max(denom, 0.0001);
}

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
    if (pixelCoord.x >= pushConstants.MipSize ||
        pixelCoord.y >= pushConstants.MipSize ||
        faceIndex >= 6)
    {
        return;
    }

    // Calculate UV coordinates for this pixel (center of pixel)
    float2 uv = (float2(pixelCoord) + 0.5) / float(pushConstants.MipSize);

    // Get the reflection direction for this cubemap texel
    // For prefiltering, this direction represents the view/reflection direction R
    float3 R = GetCubemapDirection(faceIndex, uv);

    // For the split-sum approximation, we assume N = V = R
    // This is the "isotropic" assumption that works well in practice
    float3 N = R;
    float3 V = R;

    float totalWeight = 0.0;
    float3 prefilteredColor = float3(0.0, 0.0, 0.0);

    // Use importance sampling with Hammersley sequence
    uint sampleCount = pushConstants.SampleCount;

    // For very smooth surfaces (roughness near 0), use fewer samples
    // as the result is nearly a mirror reflection
    if (pushConstants.Roughness < 0.01)
    {
        // Nearly perfect mirror - just sample in the reflection direction
        prefilteredColor = EnvironmentMap.SampleLevel(LinearSampler, R, 0).rgb;
        totalWeight = 1.0;
    }
    else
    {
        for (uint i = 0; i < sampleCount; ++i)
        {
            // Generate quasi-random sample point
            float2 Xi = Hammersley(i, sampleCount);

            // Importance sample to get half vector H
            float3 H = ImportanceSampleGGX(Xi, N, pushConstants.Roughness);

            // Compute light direction L by reflecting V around H
            float3 L = normalize(2.0 * dot(V, H) * H - V);

            float NdotL = dot(N, L);

            // Only consider samples where light is in the upper hemisphere
            if (NdotL > 0.0)
            {
                // Sample the environment map
                // Use mip level based on roughness and PDF for better quality
                // This reduces fireflies from very bright spots
                float NdotH = max(dot(N, H), 0.0);
                float HdotV = max(dot(H, V), 0.0);

                // Calculate the PDF of this sample direction
                float D = DistributionGGX(NdotH, pushConstants.Roughness);
                float pdf = (D * NdotH) / (4.0 * HdotV) + 0.0001;

                // Calculate the solid angle of the texel for filtering
                // This helps reduce aliasing artifacts
                float resolution = 512.0; // Assuming base cubemap is 512
                float saTexel = 4.0 * PI / (6.0 * resolution * resolution);
                float saSample = 1.0 / (float(sampleCount) * pdf + 0.0001);

                // Calculate mip level based on roughness and sample PDF
                float mipLevel = pushConstants.Roughness == 0.0 ? 0.0 : 0.5 * log2(saSample / saTexel);
                mipLevel = max(0.0, mipLevel);

                float3 sampleColor = EnvironmentMap.SampleLevel(LinearSampler, L, mipLevel).rgb;

                // Weight by NdotL for energy conservation
                prefilteredColor += sampleColor * NdotL;
                totalWeight += NdotL;
            }
        }
    }

    // Normalize the result
    if (totalWeight > 0.0)
    {
        prefilteredColor = prefilteredColor / totalWeight;
    }

    // Write to the prefiltered cubemap
    PrefilteredMap[uint3(pixelCoord, faceIndex)] = float4(prefilteredColor, 1.0);
}
