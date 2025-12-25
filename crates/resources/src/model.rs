//! Model and mesh loading from glTF files.

use glam::Vec3;

/// A mesh containing vertex and index data.
#[derive(Debug, Default)]
pub struct Mesh {
    /// Raw vertex data (will be converted to Vertex structs)
    pub positions: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    pub tex_coords: Vec<[f32; 2]>,
    pub tangents: Vec<[f32; 4]>,
    /// Index data
    pub indices: Vec<u32>,
}

/// A model containing one or more meshes.
#[derive(Debug, Default)]
pub struct Model {
    /// Meshes in this model
    pub meshes: Vec<Mesh>,
    /// Axis-aligned bounding box minimum
    pub aabb_min: Vec3,
    /// Axis-aligned bounding box maximum
    pub aabb_max: Vec3,
}

impl Model {
    /// Load a model from a glTF file.
    ///
    /// # Arguments
    /// * `path` - Path to the .gltf or .glb file
    ///
    /// # Returns
    /// The loaded model or an error
    pub fn load(_path: &std::path::Path) -> renderer_core::Result<Self> {
        // TODO: Implement in Task 2.5
        Ok(Self::default())
    }
}
