//! Scene graph and components.
//!
//! This crate provides scene management:
//! - Transform hierarchy
//! - Camera systems
//! - Light definitions

pub mod camera;
pub mod light;
pub mod transform;

pub use camera::{Camera, FpsController, OrbitController, Projection};
pub use light::{DirectionalLight, PointLight, SpotLight};
pub use transform::Transform;
