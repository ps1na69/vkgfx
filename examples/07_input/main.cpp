// examples/07_input/main.cpp
// Demonstrates: the full input API — keyboard (held vs. pressed),
// mouse delta, cursor lock/unlock, fullscreen toggle, and printing
// all active inputs to stdout each frame.

#include <vkgfx/window.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <sstream>

int main() {
    using namespace vkgfx;

    Window window("07 – Input", 800, 600);
    // Start unlocked so the cursor is visible
    window.setCursorLocked(false);

    std::cout << "=== Input example ===\n"
              << "  Mouse       : moves camera when cursor is LOCKED\n"
              << "  Tab         : toggle cursor lock (locked = FPS mode)\n"
              << "  F11         : toggle fullscreen\n"
              << "  Arrow keys  : prints 'held: UP/DOWN/LEFT/RIGHT'\n"
              << "  Space       : prints 'JUMP' (pressed — fires once)\n"
              << "  Escape      : quit\n\n";

    // Track how many frames each direction key has been held
    int holdFrames[4] = {0, 0, 0, 0};  // up, down, left, right

    while (!window.shouldClose()) {
        window.pollEvents();

        // ── keyPressed: fires exactly once on the frame the key goes down ──────
        if (window.keyPressed(GLFW_KEY_ESCAPE))  break;
        if (window.keyPressed(GLFW_KEY_TAB))     window.toggleCursorLock();
        if (window.keyPressed(GLFW_KEY_F11))     window.toggleFullscreen();
        if (window.keyPressed(GLFW_KEY_SPACE))   std::cout << "  [pressed] SPACE → JUMP!\n";
        if (window.keyPressed(GLFW_KEY_ENTER))   std::cout << "  [pressed] ENTER\n";

        // ── keyHeld: true every frame while the key is down ────────────────────
        const int dirs[4] = {GLFW_KEY_UP, GLFW_KEY_DOWN, GLFW_KEY_LEFT, GLFW_KEY_RIGHT};
        const char* names[4] = {"UP", "DOWN", "LEFT", "RIGHT"};
        bool anyHeld = false;
        for (int i = 0; i < 4; ++i) {
            if (window.keyHeld(dirs[i])) {
                ++holdFrames[i];
                std::cout << "  [held " << holdFrames[i] << "f] " << names[i] << "\n";
                anyHeld = true;
            } else {
                holdFrames[i] = 0;
            }
        }

        // ── Mouse delta: relative movement since last frame ────────────────────
        float dx = window.mouseDX(), dy = window.mouseDY();
        if (window.isCursorLocked() && (dx != 0.f || dy != 0.f))
            std::cout << "  [mouse] dx=" << dx << "  dy=" << dy << "\n";

        // Update title to show lock state
        window.setTitle(window.isCursorLocked()
            ? "07 – Input  [LOCKED — Tab to unlock]"
            : "07 – Input  [UNLOCKED — Tab to lock]");
    }

    std::cout << "\nClean exit.\n";
    return 0;
}
