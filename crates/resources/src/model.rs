//! Model and mesh loading from glTF files.
//!
//! This module provides functionality to load 3D models from glTF 2.0 files.
//! It extracts mesh data including positions, normals, texture coordinates,
//! tangents, and indices.
//!
//! # Example
//!
//! ```no_run
//! use std::path::Path;
//! use renderer_resources::Model;
//!
//! let model = Model::load(Path::new("assets/model.gltf")).unwrap();
//! println!("Loaded {} meshes", model.meshes.len());
//! println!("AABB: {:?} to {:?}", model.aabb_min, model.aabb_max);
//! ```

use std::path::Path;

use glam::{Vec2, Vec3, Vec4};
use tracing::{debug, info, warn};

use crate::Material;
use crate::error::{ResourceError, ResourceResult};

/// A mesh containing vertex and index data.
///
/// Each mesh represents a single primitive from a glTF file.
/// Vertex attributes are stored in separate arrays for flexibility.
#[derive(Debug, Default, Clone)]
pub struct Mesh {
    /// Vertex positions in object space.
    pub positions: Vec<Vec3>,
    /// Vertex normals (normalized).
    pub normals: Vec<Vec3>,
    /// Texture coordinates (UV).
    pub tex_coords: Vec<Vec2>,
    /// Tangent vectors with handedness in the w component.
    pub tangents: Vec<Vec4>,
    /// Index data for indexed rendering.
    pub indices: Vec<u32>,
    /// Index of the material used by this mesh (if any).
    pub material_index: Option<usize>,
}

impl Mesh {
    /// Returns the number of vertices in this mesh.
    #[inline]
    pub fn vertex_count(&self) -> usize {
        self.positions.len()
    }

    /// Returns the number of indices in this mesh.
    #[inline]
    pub fn index_count(&self) -> usize {
        self.indices.len()
    }

    /// Returns the number of triangles in this mesh.
    #[inline]
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }
}

/// A loaded 3D model containing one or more meshes.
///
/// The model also stores an axis-aligned bounding box (AABB) that
/// encompasses all vertices in all meshes.
#[derive(Debug, Default, Clone)]
pub struct Model {
    /// Meshes in this model.
    pub meshes: Vec<Mesh>,
    /// Materials used by meshes in this model.
    pub materials: Vec<Material>,
    /// Axis-aligned bounding box minimum corner.
    pub aabb_min: Vec3,
    /// Axis-aligned bounding box maximum corner.
    pub aabb_max: Vec3,
}

impl Model {
    /// Loads a model from a glTF file.
    ///
    /// Supports both `.gltf` (JSON + separate binary) and `.glb` (binary) formats.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the `.gltf` or `.glb` file
    ///
    /// # Returns
    ///
    /// The loaded model with all meshes and materials extracted.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The file cannot be read
    /// - The file is not a valid glTF file
    /// - The file contains no meshes
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::path::Path;
    /// use renderer_resources::Model;
    ///
    /// let model = Model::load(Path::new("assets/model.gltf"))?;
    /// # Ok::<(), renderer_resources::ResourceError>(())
    /// ```
    pub fn load(path: &Path) -> ResourceResult<Self> {
        // Check if file exists
        if !path.exists() {
            return Err(ResourceError::FileNotFound(path.to_path_buf()));
        }

        info!("Loading glTF model: {}", path.display());

        // Import the glTF file
        let (document, buffers, _images) =
            gltf::import(path).map_err(|e| ResourceError::GltfLoad {
                path: path.to_path_buf(),
                message: e.to_string(),
            })?;

        let mut meshes = Vec::new();
        let mut aabb_min = Vec3::splat(f32::MAX);
        let mut aabb_max = Vec3::splat(f32::MIN);

        // Extract materials
        let materials = Self::extract_materials(&document);
        debug!("Extracted {} materials", materials.len());

        // Process each mesh in the document
        for (mesh_idx, mesh) in document.meshes().enumerate() {
            debug!(
                "Processing mesh {}: {:?}",
                mesh_idx,
                mesh.name().unwrap_or("unnamed")
            );

            // Process each primitive in the mesh
            for (prim_idx, primitive) in mesh.primitives().enumerate() {
                let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

                // Extract positions (required)
                let positions: Vec<Vec3> = reader
                    .read_positions()
                    .map(|iter| iter.map(Vec3::from).collect())
                    .ok_or(ResourceError::NoPositionData)?;

                if positions.is_empty() {
                    warn!(
                        "Mesh {} primitive {} has no vertices, skipping",
                        mesh_idx, prim_idx
                    );
                    continue;
                }

                let vertex_count = positions.len();

                // Extract normals (default to up vector if missing)
                let normals: Vec<Vec3> = reader
                    .read_normals()
                    .map(|iter| iter.map(Vec3::from).collect())
                    .unwrap_or_else(|| {
                        warn!(
                            "Mesh {} primitive {} has no normals, using default Y-up",
                            mesh_idx, prim_idx
                        );
                        vec![Vec3::Y; vertex_count]
                    });

                // Extract texture coordinates (default to zero if missing)
                let tex_coords: Vec<Vec2> = reader
                    .read_tex_coords(0)
                    .map(|iter| iter.into_f32().map(Vec2::from).collect())
                    .unwrap_or_else(|| {
                        debug!(
                            "Mesh {} primitive {} has no texture coordinates",
                            mesh_idx, prim_idx
                        );
                        vec![Vec2::ZERO; vertex_count]
                    });

                // Extract tangents (default to X axis if missing)
                let tangents: Vec<Vec4> = reader
                    .read_tangents()
                    .map(|iter| iter.map(Vec4::from).collect())
                    .unwrap_or_else(|| {
                        debug!(
                            "Mesh {} primitive {} has no tangents, using default X-axis",
                            mesh_idx, prim_idx
                        );
                        // Default tangent: X axis with positive handedness
                        vec![Vec4::new(1.0, 0.0, 0.0, 1.0); vertex_count]
                    });

                // Extract indices (generate sequential if missing)
                let indices: Vec<u32> = reader
                    .read_indices()
                    .map(|iter| iter.into_u32().collect())
                    .unwrap_or_else(|| {
                        debug!(
                            "Mesh {} primitive {} has no indices, generating sequential",
                            mesh_idx, prim_idx
                        );
                        (0..vertex_count as u32).collect()
                    });

                // Update global AABB
                for pos in &positions {
                    aabb_min = aabb_min.min(*pos);
                    aabb_max = aabb_max.max(*pos);
                }

                // Get material index
                let material_index = primitive.material().index();

                debug!(
                    "Loaded primitive: {} vertices, {} indices, {} triangles",
                    vertex_count,
                    indices.len(),
                    indices.len() / 3
                );

                meshes.push(Mesh {
                    positions,
                    normals,
                    tex_coords,
                    tangents,
                    indices,
                    material_index,
                });
            }
        }

        if meshes.is_empty() {
            return Err(ResourceError::NoMeshes(path.to_path_buf()));
        }

        // Handle case where no vertices were found (empty AABB)
        if aabb_min.x == f32::MAX {
            aabb_min = Vec3::ZERO;
            aabb_max = Vec3::ZERO;
        }

        let total_vertices: usize = meshes.iter().map(|m| m.vertex_count()).sum();
        let total_triangles: usize = meshes.iter().map(|m| m.triangle_count()).sum();

        info!(
            "Model loaded: {} meshes, {} vertices, {} triangles, AABB: [{:.2}, {:.2}, {:.2}] to [{:.2}, {:.2}, {:.2}]",
            meshes.len(),
            total_vertices,
            total_triangles,
            aabb_min.x,
            aabb_min.y,
            aabb_min.z,
            aabb_max.x,
            aabb_max.y,
            aabb_max.z
        );

        Ok(Self {
            meshes,
            materials,
            aabb_min,
            aabb_max,
        })
    }

    /// Extracts materials from a glTF document.
    fn extract_materials(document: &gltf::Document) -> Vec<Material> {
        document
            .materials()
            .map(|mat| {
                let pbr = mat.pbr_metallic_roughness();

                // Extract base color
                let base_color_factor = pbr.base_color_factor();
                let base_color = Vec4::from(base_color_factor);

                // Extract metallic and roughness
                let metallic = pbr.metallic_factor();
                let roughness = pbr.roughness_factor();

                // Extract emissive color
                let emissive_factor = mat.emissive_factor();
                let emissive = Vec4::new(
                    emissive_factor[0],
                    emissive_factor[1],
                    emissive_factor[2],
                    1.0,
                );

                // Note: AO is typically in a texture, not a factor
                // Default to 1.0 (no occlusion)
                let ao = 1.0;

                Material {
                    base_color,
                    metallic,
                    roughness,
                    ao,
                    emissive,
                }
            })
            .collect()
    }

    /// Returns the total number of vertices across all meshes.
    #[inline]
    pub fn total_vertex_count(&self) -> usize {
        self.meshes.iter().map(|m| m.vertex_count()).sum()
    }

    /// Returns the total number of indices across all meshes.
    #[inline]
    pub fn total_index_count(&self) -> usize {
        self.meshes.iter().map(|m| m.index_count()).sum()
    }

    /// Returns the total number of triangles across all meshes.
    #[inline]
    pub fn total_triangle_count(&self) -> usize {
        self.meshes.iter().map(|m| m.triangle_count()).sum()
    }

    /// Returns the center of the model's bounding box.
    #[inline]
    pub fn center(&self) -> Vec3 {
        (self.aabb_min + self.aabb_max) * 0.5
    }

    /// Returns the size of the model's bounding box.
    #[inline]
    pub fn size(&self) -> Vec3 {
        self.aabb_max - self.aabb_min
    }

    /// Returns the diagonal length of the bounding box.
    #[inline]
    pub fn diagonal(&self) -> f32 {
        self.size().length()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mesh_default() {
        let mesh = Mesh::default();
        assert!(mesh.positions.is_empty());
        assert!(mesh.normals.is_empty());
        assert!(mesh.tex_coords.is_empty());
        assert!(mesh.tangents.is_empty());
        assert!(mesh.indices.is_empty());
        assert!(mesh.material_index.is_none());
    }

    #[test]
    fn test_mesh_counts() {
        let mesh = Mesh {
            positions: vec![Vec3::ZERO; 4],
            normals: vec![Vec3::Y; 4],
            tex_coords: vec![Vec2::ZERO; 4],
            tangents: vec![Vec4::X; 4],
            indices: vec![0, 1, 2, 2, 3, 0],
            material_index: None,
        };

        assert_eq!(mesh.vertex_count(), 4);
        assert_eq!(mesh.index_count(), 6);
        assert_eq!(mesh.triangle_count(), 2);
    }

    #[test]
    fn test_model_default() {
        let model = Model::default();
        assert!(model.meshes.is_empty());
        assert!(model.materials.is_empty());
        assert_eq!(model.aabb_min, Vec3::ZERO);
        assert_eq!(model.aabb_max, Vec3::ZERO);
    }

    #[test]
    fn test_model_counts() {
        let model = Model {
            meshes: vec![
                Mesh {
                    positions: vec![Vec3::ZERO; 4],
                    normals: vec![Vec3::Y; 4],
                    tex_coords: vec![Vec2::ZERO; 4],
                    tangents: vec![Vec4::X; 4],
                    indices: vec![0, 1, 2, 2, 3, 0],
                    material_index: None,
                },
                Mesh {
                    positions: vec![Vec3::ZERO; 3],
                    normals: vec![Vec3::Y; 3],
                    tex_coords: vec![Vec2::ZERO; 3],
                    tangents: vec![Vec4::X; 3],
                    indices: vec![0, 1, 2],
                    material_index: Some(0),
                },
            ],
            materials: vec![Material::default()],
            aabb_min: Vec3::new(-1.0, -1.0, -1.0),
            aabb_max: Vec3::new(1.0, 1.0, 1.0),
        };

        assert_eq!(model.total_vertex_count(), 7);
        assert_eq!(model.total_index_count(), 9);
        assert_eq!(model.total_triangle_count(), 3);
    }

    #[test]
    fn test_model_geometry_helpers() {
        let model = Model {
            meshes: vec![],
            materials: vec![],
            aabb_min: Vec3::new(-2.0, -1.0, -3.0),
            aabb_max: Vec3::new(2.0, 1.0, 3.0),
        };

        assert_eq!(model.center(), Vec3::ZERO);
        assert_eq!(model.size(), Vec3::new(4.0, 2.0, 6.0));
        assert!(
            (model.diagonal() - (4.0_f32.powi(2) + 2.0_f32.powi(2) + 6.0_f32.powi(2)).sqrt()).abs()
                < 0.001
        );
    }

    #[test]
    fn test_load_nonexistent_file() {
        let result = Model::load(Path::new("nonexistent/path/model.gltf"));
        assert!(result.is_err());

        match result {
            Err(ResourceError::FileNotFound(path)) => {
                assert!(path.to_string_lossy().contains("nonexistent"));
            }
            _ => panic!("Expected FileNotFound error"),
        }
    }
}
