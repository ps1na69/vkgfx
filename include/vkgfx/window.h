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

    // True for exactly one frame after the key transitions pressed → released
    [[nodiscard]] bool keyPressed(int glfwKey);
    [[nodiscard]] bool keyHeld(int glfwKey) const;

    [[nodiscard]] uint32_t width()  const { return m_width; }
    [[nodiscard]] uint32_t height() const { return m_height; }
    [[nodiscard]] bool     resized() const { return m_resized; }
    void clearResized() { m_resized = false; }

    [[nodiscard]] GLFWwindow* handle() const { return m_window; }

    // Vulkan surface creation
    [[nodiscard]] VkSurfaceKHR createSurface(VkInstance instance) const;
    [[nodiscard]] static std::vector<const char*> requiredExtensions();

    // Mouse delta (relative, zeroed each frame after poll)
    [[nodiscard]] float mouseDX() const { return m_mouseDX; }
    [[nodiscard]] float mouseDY() const { return m_mouseDY; }

private:
    static void framebufferResizeCb(GLFWwindow* w, int width, int height);
    static void cursorPosCb(GLFWwindow* w, double x, double y);

    GLFWwindow* m_window   = nullptr;
    uint32_t    m_width;
    uint32_t    m_height;
    bool        m_resized  = false;

    // Per-frame key tracking
    std::unordered_set<int> m_pressedThisFrame;
    std::unordered_set<int> m_heldKeys;

    double m_lastMouseX = 0, m_lastMouseY = 0;
    float  m_mouseDX    = 0, m_mouseDY    = 0;
    bool   m_firstMouse = true;
};

} // namespace vkgfx
