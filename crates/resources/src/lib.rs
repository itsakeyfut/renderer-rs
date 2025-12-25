//! Resource loading and management.
//!
//! This crate handles loading of external assets:
//! - glTF model loading
//! - Image/texture loading
//! - Material definitions

pub mod material;
pub mod model;

pub use material::Material;
pub use model::{Mesh, Model};
