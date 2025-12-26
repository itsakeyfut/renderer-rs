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
///
/// This is the primary vertex format used for 3D mesh rendering with PBR materials.
/// Each vertex contains:
/// - `position` (Vec3): 3D position in object space
/// - `normal` (Vec3): Surface normal vector (should be normalized)
/// - `tex_coord` (Vec2): Texture coordinates (UV)
/// - `tangent` (Vec4): Tangent vector with handedness in w component
///
/// # Memory Layout
///
/// The struct uses `#[repr(C)]` to ensure predictable memory layout:
/// - Offset 0: position (12 bytes)
/// - Offset 12: normal (12 bytes)
/// - Offset 24: tex_coord (8 bytes)
/// - Offset 32: tangent (16 bytes)
/// - Total size: 48 bytes
///
/// # Shader Locations
///
/// - location 0: position (vec3)
/// - location 1: normal (vec3)
/// - location 2: tex_coord (vec2)
/// - location 3: tangent (vec4)
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct Vertex {
    /// 3D position in object space.
    pub position: Vec3,
    /// Surface normal vector (should be normalized).
    pub normal: Vec3,
    /// Texture coordinates (UV).
    pub tex_coord: Vec2,
    /// Tangent vector with handedness in w component.
    /// The w component is typically 1.0 or -1.0 for handedness.
    pub tangent: Vec4,
}

impl Vertex {
    /// Creates a new vertex with the specified attributes.
    ///
    /// # Arguments
    ///
    /// * `position` - 3D position in object space
    /// * `normal` - Surface normal vector
    /// * `tex_coord` - Texture coordinates (UV)
    /// * `tangent` - Tangent vector with handedness in w
    #[inline]
    pub const fn new(position: Vec3, normal: Vec3, tex_coord: Vec2, tangent: Vec4) -> Self {
        Self {
            position,
            normal,
            tex_coord,
            tangent,
        }
    }

    /// Returns the size of the vertex in bytes.
    #[inline]
    pub const fn size() -> usize {
        std::mem::size_of::<Self>()
    }

    /// Get the vertex input binding description.
    ///
    /// Returns a binding description for binding 0 with per-vertex input rate.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_triangle_vertex_size() {
        // TriangleVertex: 2 x Vec3 = 2 x 12 = 24 bytes
        assert_eq!(std::mem::size_of::<TriangleVertex>(), 24);
    }

    #[test]
    fn test_triangle_vertex_binding_description() {
        let binding = TriangleVertex::binding_description();
        assert_eq!(binding.binding, 0);
        assert_eq!(binding.stride, 24);
        assert_eq!(binding.input_rate, vk::VertexInputRate::VERTEX);
    }

    #[test]
    fn test_triangle_vertex_attribute_descriptions() {
        let attrs = TriangleVertex::attribute_descriptions();
        assert_eq!(attrs.len(), 2);

        // Position attribute
        assert_eq!(attrs[0].binding, 0);
        assert_eq!(attrs[0].location, 0);
        assert_eq!(attrs[0].format, vk::Format::R32G32B32_SFLOAT);
        assert_eq!(attrs[0].offset, 0);

        // Color attribute
        assert_eq!(attrs[1].binding, 0);
        assert_eq!(attrs[1].location, 1);
        assert_eq!(attrs[1].format, vk::Format::R32G32B32_SFLOAT);
        assert_eq!(attrs[1].offset, 12);
    }

    #[test]
    fn test_triangle_vertex_new() {
        let position = Vec3::new(1.0, 2.0, 3.0);
        let color = Vec3::new(0.5, 0.6, 0.7);
        let vertex = TriangleVertex::new(position, color);

        assert_eq!(vertex.position, position);
        assert_eq!(vertex.color, color);
    }

    #[test]
    fn test_vertex_size() {
        // Vertex: Vec3 (12) + Vec3 (12) + Vec2 (8) + Vec4 (16) = 48 bytes
        assert_eq!(std::mem::size_of::<Vertex>(), 48);
        assert_eq!(Vertex::size(), 48);
    }

    #[test]
    fn test_vertex_binding_description() {
        let binding = Vertex::binding_description();
        assert_eq!(binding.binding, 0);
        assert_eq!(binding.stride, 48);
        assert_eq!(binding.input_rate, vk::VertexInputRate::VERTEX);
    }

    #[test]
    fn test_vertex_attribute_descriptions() {
        let attrs = Vertex::attribute_descriptions();
        assert_eq!(attrs.len(), 4);

        // Position attribute (location 0)
        assert_eq!(attrs[0].binding, 0);
        assert_eq!(attrs[0].location, 0);
        assert_eq!(attrs[0].format, vk::Format::R32G32B32_SFLOAT);
        assert_eq!(attrs[0].offset, 0);

        // Normal attribute (location 1)
        assert_eq!(attrs[1].binding, 0);
        assert_eq!(attrs[1].location, 1);
        assert_eq!(attrs[1].format, vk::Format::R32G32B32_SFLOAT);
        assert_eq!(attrs[1].offset, 12);

        // TexCoord attribute (location 2)
        assert_eq!(attrs[2].binding, 0);
        assert_eq!(attrs[2].location, 2);
        assert_eq!(attrs[2].format, vk::Format::R32G32_SFLOAT);
        assert_eq!(attrs[2].offset, 24);

        // Tangent attribute (location 3)
        assert_eq!(attrs[3].binding, 0);
        assert_eq!(attrs[3].location, 3);
        assert_eq!(attrs[3].format, vk::Format::R32G32B32A32_SFLOAT);
        assert_eq!(attrs[3].offset, 32);
    }

    #[test]
    fn test_vertex_new() {
        let position = Vec3::new(1.0, 2.0, 3.0);
        let normal = Vec3::new(0.0, 1.0, 0.0);
        let tex_coord = Vec2::new(0.5, 0.5);
        let tangent = Vec4::new(1.0, 0.0, 0.0, 1.0);

        let vertex = Vertex::new(position, normal, tex_coord, tangent);

        assert_eq!(vertex.position, position);
        assert_eq!(vertex.normal, normal);
        assert_eq!(vertex.tex_coord, tex_coord);
        assert_eq!(vertex.tangent, tangent);
    }

    #[test]
    fn test_vertex_default() {
        let vertex = Vertex::default();

        assert_eq!(vertex.position, Vec3::ZERO);
        assert_eq!(vertex.normal, Vec3::ZERO);
        assert_eq!(vertex.tex_coord, Vec2::ZERO);
        assert_eq!(vertex.tangent, Vec4::ZERO);
    }

    #[test]
    fn test_vertex_pod_zeroable() {
        // Verify Pod and Zeroable traits work correctly
        let vertex = Vertex::new(
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec2::new(0.5, 0.5),
            Vec4::new(1.0, 0.0, 0.0, 1.0),
        );

        // Test bytemuck cast to bytes and back
        let bytes: &[u8] = bytemuck::bytes_of(&vertex);
        assert_eq!(bytes.len(), 48);

        let vertex_back: &Vertex = bytemuck::from_bytes(bytes);
        assert_eq!(vertex_back.position, vertex.position);
        assert_eq!(vertex_back.normal, vertex.normal);
        assert_eq!(vertex_back.tex_coord, vertex.tex_coord);
        assert_eq!(vertex_back.tangent, vertex.tangent);
    }

    #[test]
    fn test_vertex_offsets() {
        // Verify field offsets match what we specify in attribute descriptions
        use std::mem::offset_of;

        assert_eq!(offset_of!(Vertex, position), 0);
        assert_eq!(offset_of!(Vertex, normal), 12);
        assert_eq!(offset_of!(Vertex, tex_coord), 24);
        assert_eq!(offset_of!(Vertex, tangent), 32);
    }
}
