//! Vertex data structures and input descriptions.
//!
//! This module defines vertex formats used in the renderer.

use ash::vk;
use bytemuck::{Pod, Zeroable};
use glam::{Vec2, Vec3, Vec4};

/// Standard vertex format with position, normal, UV, and tangent.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct Vertex {
    pub position: Vec3,
    pub normal: Vec3,
    pub tex_coord: Vec2,
    pub tangent: Vec4, // w = handedness
}

impl Vertex {
    /// Get the vertex input binding description.
    pub fn binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription {
            binding: 0,
            stride: std::mem::size_of::<Self>() as u32,
            input_rate: vk::VertexInputRate::VERTEX,
        }
    }

    /// Get the vertex attribute descriptions.
    pub fn attribute_descriptions() -> [vk::VertexInputAttributeDescription; 4] {
        [
            // Position at location 0
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 0,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: 0,
            },
            // Normal at location 1
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 1,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: 12,
            },
            // TexCoord at location 2
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 2,
                format: vk::Format::R32G32_SFLOAT,
                offset: 24,
            },
            // Tangent at location 3
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 3,
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: 32,
            },
        ]
    }
}
