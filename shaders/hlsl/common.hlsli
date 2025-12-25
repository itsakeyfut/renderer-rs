// Common HLSL definitions shared across all shaders
// This file contains structures, macros, and utility functions

#ifndef COMMON_HLSLI
#define COMMON_HLSLI

// Vertex input structure
struct VSInput
{
    float3 Position : POSITION;
    float3 Normal   : NORMAL;
    float2 TexCoord : TEXCOORD0;
    float4 Color    : COLOR0;
};

// Per-frame constant buffer
struct FrameData
{
    float4x4 ViewMatrix;
    float4x4 ProjectionMatrix;
    float4x4 ViewProjectionMatrix;
    float3   CameraPosition;
    float    Time;
    float2   ScreenSize;
    float2   InvScreenSize;
};

// Per-object constant buffer
struct ObjectData
{
    float4x4 WorldMatrix;
    float4x4 NormalMatrix;
};

// Material data
struct MaterialData
{
    float4 BaseColor;
    float  Metallic;
    float  Roughness;
    float  AO;
    float  Padding;
};

// Utility functions
float3 LinearToSRGB(float3 color)
{
    return pow(color, 1.0 / 2.2);
}

float3 SRGBToLinear(float3 color)
{
    return pow(color, 2.2);
}

#endif // COMMON_HLSLI
