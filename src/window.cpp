// src/window.cpp
#include <vkgfx/window.h>
#include <array>
#include <stdexcept>
#include <iostream>

namespace vkgfx {

Window::Window(const std::string& title, uint32_t width, uint32_t height)
    : m_width(width), m_height(height)
{
    if (!glfwInit())
        throw std::runtime_error("[vkgfx] glfwInit failed");

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE,  GLFW_TRUE);

    m_window = glfwCreateWindow(
        static_cast<int>(width), static_cast<int>(height),
        title.c_str(), nullptr, nullptr);

    if (!m_window)
        throw std::runtime_error("[vkgfx] glfwCreateWindow failed");

    glfwSetWindowUserPointer(m_window, this);
    glfwSetFramebufferSizeCallback(m_window, framebufferResizeCb);
    glfwSetCursorPosCallback(m_window, cursorPosCb);
    glfwSetInputMode(m_window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
}

Window::~Window() {
    if (m_window) glfwDestroyWindow(m_window);
    glfwTerminate();
}

void Window::pollEvents() {
    m_pressedThisFrame.clear();
    m_mouseDX = 0.f;
    m_mouseDY = 0.f;

    glfwPollEvents();

    // Build held-key set from GLFW state
    // Track transitions for keyPressed()
    static const std::array<int,12> TRACKED_KEYS = {
        GLFW_KEY_W, GLFW_KEY_A, GLFW_KEY_S, GLFW_KEY_D,
        GLFW_KEY_Q, GLFW_KEY_E, GLFW_KEY_ESCAPE,
        GLFW_KEY_F1, GLFW_KEY_F2, GLFW_KEY_F3,
        GLFW_KEY_F4, GLFW_KEY_F5
    };

    for (int k : TRACKED_KEYS) {
        bool down = glfwGetKey(m_window, k) == GLFW_PRESS;
        bool wasHeld = m_heldKeys.count(k) > 0;
        if (down) {
            m_heldKeys.insert(k);
            if (!wasHeld) m_pressedThisFrame.insert(k); // just pressed
        } else {
            m_heldKeys.erase(k);
        }
    }
}

bool Window::shouldClose() const {
    return glfwWindowShouldClose(m_window);
}

bool Window::keyPressed(int glfwKey) {
    return m_pressedThisFrame.count(glfwKey) > 0;
}

bool Window::keyHeld(int glfwKey) const {
    return m_heldKeys.count(glfwKey) > 0;
}

VkSurfaceKHR Window::createSurface(VkInstance instance) const {
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VkResult res = glfwCreateWindowSurface(instance, m_window, nullptr, &surface);
    if (res != VK_SUCCESS)
        throw std::runtime_error("[vkgfx] glfwCreateWindowSurface failed");
    return surface;
}

std::vector<const char*> Window::requiredExtensions() {
    uint32_t count = 0;
    const char** ext = glfwGetRequiredInstanceExtensions(&count);
    return std::vector<const char*>(ext, ext + count);
}

// ── Static callbacks ──────────────────────────────────────────────────────────

void Window::framebufferResizeCb(GLFWwindow* w, int, int) {
    auto* self = static_cast<Window*>(glfwGetWindowUserPointer(w));
    self->m_resized = true;
    int fw, fh;
    glfwGetFramebufferSize(w, &fw, &fh);
    self->m_width  = static_cast<uint32_t>(fw);
    self->m_height = static_cast<uint32_t>(fh);
}

void Window::cursorPosCb(GLFWwindow* w, double x, double y) {
    auto* self = static_cast<Window*>(glfwGetWindowUserPointer(w));
    if (self->m_firstMouse) {
        self->m_lastMouseX = x;
        self->m_lastMouseY = y;
        self->m_firstMouse = false;
    }
    self->m_mouseDX += static_cast<float>(x - self->m_lastMouseX);
    self->m_mouseDY += static_cast<float>(y - self->m_lastMouseY);
    self->m_lastMouseX = x;
    self->m_lastMouseY = y;
}

} // namespace vkgfx
