#pragma once
// include/vkgfx/window.h

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <string>
#include <vector>
#include <functional>
#include <unordered_set>

namespace vkgfx {

class Window {
public:
    Window(const std::string& title, uint32_t width, uint32_t height);
    ~Window();

    Window(const Window&)            = delete;
    Window& operator=(const Window&) = delete;

    void pollEvents();
    [[nodiscard]] bool shouldClose() const;

    // ── Input ─────────────────────────────────────────────────────────────────
    // keyPressed: true for exactly one frame when the key transitions down
    [[nodiscard]] bool keyPressed(int glfwKey);
    [[nodiscard]] bool keyHeld   (int glfwKey) const;

    [[nodiscard]] float mouseDX() const { return m_mouseDX; }
    [[nodiscard]] float mouseDY() const { return m_mouseDY; }

    // ── Size ──────────────────────────────────────────────────────────────────
    [[nodiscard]] uint32_t width()  const { return m_width; }
    [[nodiscard]] uint32_t height() const { return m_height; }
    [[nodiscard]] float    aspect() const {
        return m_height > 0 ? static_cast<float>(m_width) / m_height : 1.f;
    }
    [[nodiscard]] bool resized() const { return m_resized; }
    void clearResized() { m_resized = false; }

    // ── Window control ────────────────────────────────────────────────────────
    void setTitle(const std::string& title);
    void setTitle(const char* title);

    // Toggle or explicitly enter/exit fullscreen.
    // Remembers windowed size/position so restore works correctly.
    void setFullscreen(bool enable);
    void toggleFullscreen();
    [[nodiscard]] bool isFullscreen() const { return m_fullscreen; }

    // Cursor lock (captured + hidden vs. normal)
    void setCursorLocked(bool locked);
    void toggleCursorLock();
    [[nodiscard]] bool isCursorLocked() const { return m_cursorLocked; }

    // Resize the window (only meaningful in windowed mode)
    void resize(uint32_t w, uint32_t h);

    // ── Vulkan ────────────────────────────────────────────────────────────────
    [[nodiscard]] VkSurfaceKHR createSurface(VkInstance instance) const;
    [[nodiscard]] static std::vector<const char*> requiredExtensions();
    [[nodiscard]] GLFWwindow* handle() const { return m_window; }

private:
    static void framebufferResizeCb(GLFWwindow* w, int width, int height);
    static void cursorPosCb        (GLFWwindow* w, double x,  double y);
    static void keyCb              (GLFWwindow* w, int key, int scancode,
                                    int action, int mods);

    GLFWwindow* m_window     = nullptr;
    uint32_t    m_width      = 0;
    uint32_t    m_height     = 0;
    bool        m_resized    = false;
    bool        m_fullscreen = false;
    bool        m_cursorLocked = true;

    // Saved windowed state for restoring after leaving fullscreen
    int m_windowedX = 100, m_windowedY = 100;
    int m_windowedW = 1280, m_windowedH = 720;

    // Key tracking — pressed-this-frame uses the GLFW key callback
    // so every key works without needing a static allowlist.
    std::unordered_set<int> m_pressedThisFrame;
    std::unordered_set<int> m_heldKeys;

    double m_lastMouseX = 0, m_lastMouseY = 0;
    float  m_mouseDX    = 0, m_mouseDY    = 0;
    bool   m_firstMouse = true;
};

} // namespace vkgfx
