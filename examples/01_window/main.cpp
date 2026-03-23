// examples/01_window/main.cpp
// Demonstrates: creating a window and running the engine event loop.
// No rendering is performed — just shows how Window works in isolation.

#include <vkgfx/window.h>
#include <iostream>

int main() {
    // Create an 800×600 window. Title, width, height.
    vkgfx::Window window("01 – Window", 800, 600);

    // Unlock the cursor so it stays visible (no FPS-camera capture)
    window.setCursorLocked(false);

    std::cout << "Window created. Press Escape to exit, F11 to toggle fullscreen.\n";

    // ── Main loop ─────────────────────────────────────────────────────────────
    while (!window.shouldClose()) {
        // Poll OS events (keyboard, mouse, resize, close)
        window.pollEvents();

        // Escape = quit
        if (window.keyPressed(GLFW_KEY_ESCAPE))
            break;

        // F11 = toggle borderless fullscreen
        if (window.keyPressed(GLFW_KEY_F11))
            window.toggleFullscreen();

        // Print window dimensions on resize
        if (window.resized()) {
            std::cout << "Resized to " << window.width() << "×" << window.height() << "\n";
            window.clearResized();
        }
    }

    std::cout << "Clean exit.\n";
    return 0;
}
