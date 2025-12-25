// Model Vertex Shader
// Transforms model vertices with MVP matrix and passes data to pixel shader

// Camera uniform buffer
cbuffer CameraData : register(b0)
{
    float4x4 view;
    float4x4 projection;
    float4x4 viewProjection;
    float3 cameraPosition;
    float padding;
};

// Object uniform buffer
cbuffer ObjectData : register(b1)
{
    float4x4 model;
    float4x4 normalMatrix;
};

struct VSInput
{
    float3 Position : POSITION;
    float3 Normal   : NORMAL;
    float2 TexCoord : TEXCOORD;
    float4 Tangent  : TANGENT;
};

struct VSOutput
{
    float4 Position    : SV_POSITION;
    float3 WorldPos    : TEXCOORD0;
    float3 Normal      : TEXCOORD1;
    float2 TexCoord    : TEXCOORD2;
    float3 Tangent     : TEXCOORD3;
    float3 Bitangent   : TEXCOORD4;
};

VSOutput main(VSInput input)
{
    VSOutput output;

    // Transform position to world space
    float4 worldPos = mul(model, float4(input.Position, 1.0));
    output.WorldPos = worldPos.xyz;

    // Transform to clip space
    output.Position = mul(viewProjection, worldPos);

    // Transform normal and tangent to world space
    float3 N = normalize(mul((float3x3)normalMatrix, input.Normal));
    float3 T = normalize(mul((float3x3)model, input.Tangent.xyz));

    // Re-orthogonalize T with respect to N (Gram-Schmidt)
    T = normalize(T - dot(T, N) * N);

    // Calculate bitangent with handedness
    float3 B = cross(N, T) * input.Tangent.w;

    output.Normal = N;
    output.Tangent = T;
    output.Bitangent = B;

    // Pass through texture coordinates
    output.TexCoord = input.TexCoord;

    return output;
}
