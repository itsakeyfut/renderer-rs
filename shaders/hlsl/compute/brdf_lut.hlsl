// BRDF Look-Up Table Generation Compute Shader
// Pre-integrates the BRDF for the split-sum approximation in IBL
//
// The LUT stores:
// - U axis: NdotV (view angle from 0.0 to 1.0)
// - V axis: roughness (from 0.0 to 1.0)
// - R channel: F0 scale factor (1 - Fc) * G_Vis integrated
// - G channel: F0 bias factor (Fc * G_Vis integrated)
//
// Usage: specularIBL = prefilteredColor * (F0 * brdfLUT.r + brdfLUT.g)
//
// Reference: "Real Shading in Unreal Engine 4" - Brian Karis, Epic Games

#define PI 3.14159265359

// Output BRDF LUT (2D texture, not cubemap)
// Explicit format specification to match VK_FORMAT_R16G16_SFLOAT
[[vk::binding(0, 0)]] [[vk::image_format("rg16f")]]
RWTexture2D<float2> BRDFLut : register(u0);

// Push constants for BRDF LUT size
[[vk::push_constant]]
struct PushConstants
{
    uint LutSize;  // Size of the LUT (e.g., 512)
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

// Schlick-GGX Geometry function for IBL
// Uses k = (roughness^2) / 2 for IBL (different from direct lighting)
float GeometrySchlickGGX(float NdotV, float roughness)
{
    // IBL uses a different k value than direct lighting
    // For IBL: k = roughness^2 / 2
    // For direct: k = (roughness + 1)^2 / 8
    float a = roughness;
    float k = (a * a) / 2.0;

    float nom = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return nom / max(denom, 0.0001);
}

// Smith's method for geometry function
// Combines geometry shadowing (light) and masking (view)
float GeometrySmith(float3 N, float3 V, float3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = GeometrySchlickGGX(NdotV, roughness);
    float ggx1 = GeometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}

// Integrate the BRDF for a given NdotV and roughness
// Returns (scale, bias) factors for the Fresnel term
//
// The split-sum approximation:
// L_o = integral(L_i * f_r * cos(theta)) = L_c * (F0 * scale + bias)
// where L_c is the prefiltered environment color
float2 IntegrateBRDF(float NdotV, float roughness)
{
    // Construct view vector from NdotV
    // We're in tangent space where N = (0, 0, 1)
    float3 V;
    V.x = sqrt(1.0 - NdotV * NdotV);  // sin(theta)
    V.y = 0.0;
    V.z = NdotV;                       // cos(theta)

    float A = 0.0;  // Scale factor (1 - Fc) * G_Vis sum
    float B = 0.0;  // Bias factor (Fc * G_Vis) sum

    // Normal in tangent space
    float3 N = float3(0.0, 0.0, 1.0);

    // Number of samples for Monte Carlo integration
    // 1024 samples provides good quality vs performance
    const uint SAMPLE_COUNT = 1024u;

    for (uint i = 0u; i < SAMPLE_COUNT; ++i)
    {
        // Generate quasi-random sample point
        float2 Xi = Hammersley(i, SAMPLE_COUNT);

        // Importance sample to get half vector H according to GGX NDF
        float3 H = ImportanceSampleGGX(Xi, N, roughness);

        // Compute light direction L by reflecting V around H
        float3 L = normalize(2.0 * dot(V, H) * H - V);

        float NdotL = max(L.z, 0.0);
        float NdotH = max(H.z, 0.0);
        float VdotH = max(dot(V, H), 0.0);

        // Only consider samples in the upper hemisphere
        if (NdotL > 0.0)
        {
            // Geometry term using Smith's method
            float G = GeometrySmith(N, V, L, roughness);

            // Visibility term: G / (4 * NdotL * NdotV)
            // Simplified: G_Vis = (G * VdotH) / (NdotH * NdotV)
            float G_Vis = (G * VdotH) / max(NdotH * NdotV, 0.0001);

            // Schlick Fresnel approximation: F = F0 + (1 - F0) * (1 - VdotH)^5
            // Fc = (1 - VdotH)^5 is the Fresnel coefficient
            float Fc = pow(1.0 - VdotH, 5.0);

            // Accumulate contributions
            // A = integral((1 - Fc) * G_Vis)
            // B = integral(Fc * G_Vis)
            A += (1.0 - Fc) * G_Vis;
            B += Fc * G_Vis;
        }
    }

    // Average over all samples
    A /= float(SAMPLE_COUNT);
    B /= float(SAMPLE_COUNT);

    return float2(A, B);
}

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint2 pixelCoord = dispatchThreadId.xy;

    // Check bounds
    if (pixelCoord.x >= pushConstants.LutSize ||
        pixelCoord.y >= pushConstants.LutSize)
    {
        return;
    }

    // Calculate UV coordinates (center of pixel)
    // U = NdotV (0.0 to 1.0)
    // V = roughness (0.0 to 1.0)
    float2 uv = (float2(pixelCoord) + 0.5) / float(pushConstants.LutSize);

    // Clamp NdotV to avoid edge cases at grazing angles
    // Using a small epsilon to prevent division by zero issues
    float NdotV = max(uv.x, 0.001);
    float roughness = uv.y;

    // Integrate BRDF for this (NdotV, roughness) pair
    float2 result = IntegrateBRDF(NdotV, roughness);

    // Store result (scale, bias) in RG channels
    BRDFLut[pixelCoord] = result;
}
