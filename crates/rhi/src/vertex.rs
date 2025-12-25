//! Vertex data structures and input descriptions.
//!
//! This module defines vertex formats used in the renderer.
//!
//! # Vertex Types
//!
//! - [`TriangleVertex`] - Simple vertex with position and color for Hello Triangle
//! - [`Vertex`] - Full PBR vertex with position, normal, UV, and tangent

use ash::vk;
use bytemuck::{Pod, Zeroable};
use glam::{Vec2, Vec3, Vec4};

/// Simple vertex format with position and color.
///
/// Used for basic rendering like the Hello Triangle example.
/// Each vertex contains:
/// - Position (Vec3): 3D position in clip space
/// - Color (Vec3): RGB color
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct TriangleVertex {
    pub position: Vec3,
    pub color: Vec3,
}

impl TriangleVertex {
    /// Creates a new triangle vertex.
    #[inline]
    pub const fn new(position: Vec3, color: Vec3) -> Self {
        Self { position, color }
    }

    /// Get the vertex input binding description.
    pub fn binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription {
            binding: 0,
            stride: std::mem::size_of::<Self>() as u32,
            input_rate: vk::VertexInputRate::VERTEX,
        }
    }

    /// Get the vertex attribute descriptions.
    pub fn attribute_descriptions() -> [vk::VertexInputAttributeDescription; 2] {
        [
            // Position at location 0
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 0,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: 0,
            },
            // Color at location 1
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 1,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: 12, // offset of color field (after position: 3 * 4 = 12 bytes)
            },
        ]
    }
}

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
