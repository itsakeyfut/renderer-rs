//! Resource loading and management.
//!
//! This crate handles loading of external assets:
//! - glTF model loading
//! - Image/texture loading
//! - Material definitions
//! - Uniform Buffer Object definitions
//!
//! # Example
//!
//! ```no_run
//! use std::path::Path;
//! use renderer_resources::Model;
//!
//! // Load a glTF model
//! let model = Model::load(Path::new("assets/model.gltf")).unwrap();
//!
//! // Access mesh data
//! for mesh in &model.meshes {
//!     println!("Mesh has {} vertices", mesh.vertex_count());
//! }
//! ```

pub mod error;
pub mod material;
pub mod model;
pub mod ubo;

pub use error::{ResourceError, ResourceResult};
pub use material::Material;
pub use model::{Mesh, Model};
pub use ubo::{CameraUbo, DirectionalLightUbo, ObjectUbo, SceneUbo};
