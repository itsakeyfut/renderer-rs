//! Vulkan Renderer - Main Entry Point
//!
//! This is a Vulkan-based renderer implemented in Rust, following modern
//! rendering techniques including PBR, IBL, and deferred shading.

use anyhow::Result;
use tracing::{error, info};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::WindowId;

use renderer_core::Timer;
use renderer_platform::{InputState, Window};
use renderer_renderer::Renderer;

struct App {
    window: Option<Window>,
    renderer: Option<Renderer>,
    input: InputState,
    timer: Timer,
}

impl App {
    fn new() -> Self {
        Self {
            window: None,
            renderer: None,
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
                    // Create renderer after window is created
                    match Renderer::new(&window) {
                        Ok(renderer) => {
                            info!("Initialization complete, entering main loop");
                            self.renderer = Some(renderer);
                            self.window = Some(window);
                        }
                        Err(e) => {
                            error!("Failed to create renderer: {:?}", e);
                            event_loop.exit();
                        }
                    }
                }
                Err(e) => {
                    error!("Failed to create window: {}", e);
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
                if let Some(ref mut window) = self.window {
                    window.resize(size.width, size.height);
                }
                if let Some(ref mut renderer) = self.renderer {
                    renderer.resize(size.width, size.height);
                }
            }
            WindowEvent::RedrawRequested => {
                let _delta = self.timer.delta_secs();

                if let Some(ref mut renderer) = self.renderer
                    && let Err(e) = renderer.render_frame()
                {
                    error!("Render error: {:?}", e);
                }
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
