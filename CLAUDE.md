# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Vulkan 1.3-based 3D renderer in Rust for learning Unreal Engine graphics architecture. Uses ash for Vulkan bindings, winit for windowing, and gpu-allocator for memory management.

## Build Commands

This project uses `just` for build automation:

```bash
just dev-build     # Debug build
just dev-run       # Run with RUST_LOG=debug
just dev           # Build and run debug

just release-build # Release build
just release       # Build and run release

just fmt           # Format code
just clippy        # Lint (strict: -D warnings)
just check         # Format + clippy
just build         # Build workspace

just test          # Unit tests only
just test-all      # All tests including integration
just test-seq      # Sequential test execution
```

Direct cargo commands:
```bash
cargo test --workspace --lib                    # Unit tests
cargo test --workspace --lib -- test_name       # Single test
cargo clippy --workspace -- -D warnings         # Strict linting
```

## Architecture

### Crate Dependency Order (bottom-up)

```
app → renderer → scene → resources → rhi → platform → core
```

- **core**: Logging, error types, timer utilities
- **platform**: Window/surface creation (winit), input handling
- **rhi**: Vulkan abstraction layer (ash, gpu-allocator)
- **resources**: Asset loading (glTF, textures)
- **scene**: Transform, camera, light components
- **renderer**: Render passes, frame management
- **app**: Main entry point and event loop

### Key Design Patterns

1. **Handle-based resources**: `Handle<T>` with generation counters for GPU resources
2. **Deferred deletion**: Per-frame deletion queues for in-flight resource management
3. **Builder pattern**: `GraphicsPipelineBuilder<'a>` for complex object construction
4. **ManuallyDrop for drop order**: Allocator must be dropped before Device (see `device.rs`)

### Important Files

- `specs/architecture.md` - Complete architecture documentation (937 lines)
- `specs/implementation-phases.md` - Detailed phase tasks with code examples
- `crates/rhi/src/device.rs` - Logical device with gpu-allocator integration
- `crates/app/src/main.rs` - ApplicationHandler event loop

## Code Conventions

- All code comments and documentation in English
- Use `tracing` macros (`info!`, `warn!`, `error!`) not `println!`
- Error handling: `Result<T, E>` with `?` operator, avoid `unwrap()`
- Thread safety: `Arc<Mutex<T>>` or `Arc<RwLock<T>>` for shared mutable state
- Type aliases: `RhiResult<T> = Result<T, RhiError>`

## Shaders

HLSL shaders in `shaders/hlsl/` compiled to SPIR-V via DXC:
- `vertex/`, `pixel/`, `compute/` subdirectories
- Shared headers: `common.hlsli`, `lights.hlsli`, `pbr.hlsli`, `shadow.hlsli`
