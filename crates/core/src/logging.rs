//! Logging initialization and configuration.

use tracing_subscriber::{EnvFilter, fmt, prelude::*};

/// Initialize the logging system with tracing.
///
/// This sets up tracing-subscriber with:
/// - Environment-based filtering (RUST_LOG)
/// - Pretty printing for development
///
/// # Example
/// ```
/// renderer_core::init_logging();
/// tracing::info!("Renderer initialized");
/// ```
pub fn init_logging() {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,renderer=debug,wgpu=warn,naga=warn"));

    tracing_subscriber::registry()
        .with(filter)
        .with(fmt::layer().with_target(true).with_thread_ids(true))
        .init();
}
