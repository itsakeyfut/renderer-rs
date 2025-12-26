//! Resource loading and management.
//!
//! This crate handles loading of external assets:
//! - glTF model loading
//! - Image/texture loading
//! - Material definitions
//! - Uniform Buffer Object definitions

pub mod material;
pub mod model;
pub mod ubo;

pub use material::Material;
pub use model::{Mesh, Model};
pub use ubo::{CameraUbo, DirectionalLightUbo, ObjectUbo, SceneUbo};
