//! Vulkan Renderer - Main Entry Point
//!
//! This is a Vulkan-based renderer implemented in Rust, following modern
//! rendering techniques including PBR, IBL, and deferred shading.

use anyhow::Result;
use tracing::info;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::WindowId;

use renderer_core::Timer;
use renderer_platform::{InputState, Window};

struct App {
    window: Option<Window>,
    input: InputState,
    timer: Timer,
}

impl App {
    fn new() -> Self {
        Self {
            window: None,
            input: InputState::new(),
            timer: Timer::new(),
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            match Window::new(event_loop, 1280, 720, "Vulkan Renderer") {
                Ok(window) => {
                    info!("Initialization complete, entering main loop");
                    self.window = Some(window);
                }
                Err(e) => {
                    tracing::error!("Failed to create window: {}", e);
                    event_loop.exit();
                }
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                info!("Close requested, shutting down");
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                info!("Window resized to {}x{}", size.width, size.height);
                // TODO: Handle resize
            }
            WindowEvent::RedrawRequested => {
                let _delta = self.timer.delta_secs();
                // TODO: Render frame
            }
            WindowEvent::KeyboardInput { event, .. } => {
                use winit::keyboard::PhysicalKey;
                if let PhysicalKey::Code(key) = event.physical_key {
                    if event.state.is_pressed() {
                        self.input.on_key_pressed(key);
                    } else {
                        self.input.on_key_released(key);
                    }
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        self.input.begin_frame();
        if let Some(ref window) = self.window {
            window.request_redraw();
        }
    }
}

fn main() -> Result<()> {
    // Initialize logging
    renderer_core::init_logging();
    info!("Starting Vulkan Renderer");

    // Create event loop
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);

    // Create app and run
    let mut app = App::new();
    event_loop.run_app(&mut app)?;

    Ok(())
}
