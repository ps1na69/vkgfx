// src/window.cpp
#include <vkgfx/window.h>
#include <stdexcept>
#include <iostream>

namespace vkgfx {

// ── Constructor / Destructor ──────────────────────────────────────────────────

Window::Window(const std::string& title, uint32_t width, uint32_t height)
    : m_width(width), m_height(height),
      m_windowedW(static_cast<int>(width)),
      m_windowedH(static_cast<int>(height))
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

    // Save initial windowed position
    glfwGetWindowPos(m_window, &m_windowedX, &m_windowedY);

    glfwSetWindowUserPointer(m_window, this);
    glfwSetFramebufferSizeCallback(m_window, framebufferResizeCb);
    glfwSetCursorPosCallback      (m_window, cursorPosCb);
    glfwSetKeyCallback            (m_window, keyCb);

    // Start with cursor locked (FPS camera)
    glfwSetInputMode(m_window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
}

Window::~Window() {
    if (m_window) glfwDestroyWindow(m_window);
    glfwTerminate();
}

// ── pollEvents ────────────────────────────────────────────────────────────────

void Window::pollEvents() {
    m_pressedThisFrame.clear();
    m_mouseDX = 0.f;
    m_mouseDY = 0.f;
    glfwPollEvents();

    // Rebuild held-key set from live GLFW state.
    // We iterate the currently held set — any key that is no longer pressed
    // gets removed. New presses are detected by the keyCb.
    std::unordered_set<int> stillHeld;
    for (int k : m_heldKeys)
        if (glfwGetKey(m_window, k) == GLFW_PRESS)
            stillHeld.insert(k);
    m_heldKeys = std::move(stillHeld);
}

// ── Queries ───────────────────────────────────────────────────────────────────

bool Window::shouldClose() const { return glfwWindowShouldClose(m_window); }
bool Window::keyPressed(int k)   { return m_pressedThisFrame.count(k) > 0; }
bool Window::keyHeld   (int k) const { return m_heldKeys.count(k) > 0; }

// ── Title ─────────────────────────────────────────────────────────────────────

void Window::setTitle(const std::string& title) {
    glfwSetWindowTitle(m_window, title.c_str());
}
void Window::setTitle(const char* title) {
    glfwSetWindowTitle(m_window, title);
}

// ── Fullscreen ────────────────────────────────────────────────────────────────

void Window::setFullscreen(bool enable) {
    if (enable == m_fullscreen) return;

    if (enable) {
        // Save windowed state
        glfwGetWindowPos (m_window, &m_windowedX, &m_windowedY);
        glfwGetWindowSize(m_window, &m_windowedW, &m_windowedH);

        GLFWmonitor*       monitor = glfwGetPrimaryMonitor();
        const GLFWvidmode* mode    = glfwGetVideoMode(monitor);
        glfwSetWindowMonitor(m_window, monitor,
                             0, 0, mode->width, mode->height,
                             mode->refreshRate);
    } else {
        // Restore windowed state
        glfwSetWindowMonitor(m_window, nullptr,
                             m_windowedX, m_windowedY,
                             m_windowedW, m_windowedH,
                             GLFW_DONT_CARE);
    }
    m_fullscreen = enable;
    m_firstMouse = true; // reset mouse delta after mode switch
}

void Window::toggleFullscreen() { setFullscreen(!m_fullscreen); }

// ── Cursor lock ───────────────────────────────────────────────────────────────

void Window::setCursorLocked(bool locked) {
    m_cursorLocked = locked;
    glfwSetInputMode(m_window, GLFW_CURSOR,
                     locked ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
    m_firstMouse = true; // avoid jump when cursor warps
}

void Window::toggleCursorLock() { setCursorLocked(!m_cursorLocked); }

// ── Resize (windowed mode only) ───────────────────────────────────────────────

void Window::resize(uint32_t w, uint32_t h) {
    if (m_fullscreen) return;
    glfwSetWindowSize(m_window, static_cast<int>(w), static_cast<int>(h));
    m_width  = w;
    m_height = h;
}

// ── Vulkan ────────────────────────────────────────────────────────────────────

VkSurfaceKHR Window::createSurface(VkInstance instance) const {
    VkSurfaceKHR s = VK_NULL_HANDLE;
    if (glfwCreateWindowSurface(instance, m_window, nullptr, &s) != VK_SUCCESS)
        throw std::runtime_error("[vkgfx] glfwCreateWindowSurface failed");
    return s;
}

std::vector<const char*> Window::requiredExtensions() {
    uint32_t    count = 0;
    const char** ext  = glfwGetRequiredInstanceExtensions(&count);
    return {ext, ext + count};
}

// ── Static callbacks ──────────────────────────────────────────────────────────

void Window::framebufferResizeCb(GLFWwindow* w, int fw, int fh) {
    auto* self     = static_cast<Window*>(glfwGetWindowUserPointer(w));
    self->m_resized = true;
    self->m_width   = static_cast<uint32_t>(fw);
    self->m_height  = static_cast<uint32_t>(fh);
}

void Window::cursorPosCb(GLFWwindow* w, double x, double y) {
    auto* self = static_cast<Window*>(glfwGetWindowUserPointer(w));
    if (!self->m_cursorLocked) return; // ignore mouse when cursor is free
    if (self->m_firstMouse) {
        self->m_lastMouseX = x;
        self->m_lastMouseY = y;
        self->m_firstMouse = false;
        return;
    }
    self->m_mouseDX += static_cast<float>(x - self->m_lastMouseX);
    self->m_mouseDY += static_cast<float>(y - self->m_lastMouseY);
    self->m_lastMouseX = x;
    self->m_lastMouseY = y;
}

void Window::keyCb(GLFWwindow* w, int key, int /*scancode*/,
                   int action, int /*mods*/) {
    auto* self = static_cast<Window*>(glfwGetWindowUserPointer(w));
    if (action == GLFW_PRESS) {
        self->m_heldKeys.insert(key);
        self->m_pressedThisFrame.insert(key);
    } else if (action == GLFW_RELEASE) {
        self->m_heldKeys.erase(key);
    }
    // GLFW_REPEAT: don't add to pressedThisFrame (keeps single-press semantics)
}

} // namespace vkgfx
