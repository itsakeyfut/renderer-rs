# Update local main branch
new:
    git checkout main && git fetch && git pull origin main

# === Build Commands ===

# Build workspace in debug mode
build:
    cargo build --workspace

# Build renderer in debug mode (faster compile, more logging)
dev-build:
    cargo build -p renderer-app

# Build renderer in release mode (optimized, minimal logging)
release-build:
    cargo build -p renderer-app --release

# === Run Commands ===

# Run renderer in debug mode with debug logging
dev-run:
    RUST_LOG=debug,ash=warn,gpu_allocator=warn cargo run -p renderer-app

# Run renderer in release mode with info logging
release-run:
    cargo run -p renderer-app --release

# === Shortcuts ===

# Build and run in debug mode
dev: dev-build dev-run

# Build and run in release mode
release: release-build release-run

# === Code Quality ===

# Format code
fmt:
    cargo fmt --all

# Run clippy
clippy:
    cargo clippy --workspace -- -D warnings

# Quick check (format + clippy)
check:
    cargo fmt --all -- --check && cargo clippy --workspace -- -D warnings

# === Testing ===

# Run unit tests (fast)
test:
    cargo test --workspace --lib

# Run all tests including integration tests
test-all:
    cargo test --workspace

# Run tests sequentially (saves memory)
test-seq:
    cargo test --workspace --lib -- --test-threads=1
