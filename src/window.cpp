#include "vkgfx/window.h"

namespace vkgfx {

int Window::s_windowCount = 0;

Window::Window(const WindowSettings& settings)
    : m_settings(settings)
{
    if (s_windowCount == 0) {
        if (!glfwInit())
            throw std::runtime_error("[VKGFX] Failed to initialise GLFW");
    }
    ++s_windowCount;

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, settings.resizable ? GLFW_TRUE : GLFW_FALSE);

    GLFWmonitor* monitor = settings.fullscreen ? glfwGetPrimaryMonitor() : nullptr;
    m_handle = glfwCreateWindow(
        static_cast<int>(settings.width),
        static_cast<int>(settings.height),
        settings.title.c_str(),
        monitor, nullptr);

    if (!m_handle)
        throw std::runtime_error("[VKGFX] Failed to create GLFW window");

    glfwSetWindowUserPointer(m_handle, this);
    glfwSetFramebufferSizeCallback(m_handle, framebufferResizeCallback);
    glfwSetKeyCallback(m_handle, keyCallback);
    glfwSetCursorPosCallback(m_handle, cursorPosCallback);
}

Window::Window(std::string_view title, uint32_t w, uint32_t h, bool resizable)
    : Window(WindowSettings{w, h, std::string(title), resizable})
{}

Window::~Window() {
    if (m_handle) glfwDestroyWindow(m_handle);
    if (--s_windowCount == 0) glfwTerminate();
}

VkSurfaceKHR Window::createSurface(VkInstance instance) const {
    VkSurfaceKHR surface;
    VK_CHECK(glfwCreateWindowSurface(instance, m_handle, nullptr, &surface),
             "Failed to create window surface");
    return surface;
}

std::pair<uint32_t, uint32_t> Window::getFramebufferSize() const {
    int w, h;
    glfwGetFramebufferSize(m_handle, &w, &h);
    return { static_cast<uint32_t>(w), static_cast<uint32_t>(h) };
}

float Window::getAspectRatio() const {
    auto [w, h] = getFramebufferSize();
    return h > 0 ? static_cast<float>(w) / static_cast<float>(h) : 1.f;
}

Vec2 Window::getCursorPos() const {
    return m_cursor;
}

void Window::setCursorVisible(bool visible) {
    glfwSetInputMode(m_handle, GLFW_CURSOR,
                     visible ? GLFW_CURSOR_NORMAL : GLFW_CURSOR_DISABLED);
}

void Window::setTitle(std::string_view title) {
    m_settings.title = std::string(title);
    glfwSetWindowTitle(m_handle, m_settings.title.c_str());
}

std::vector<const char*> Window::getRequiredInstanceExtensions() {
    uint32_t count = 0;
    const char** exts = glfwGetRequiredInstanceExtensions(&count);
    return std::vector<const char*>(exts, exts + count);
}

void Window::framebufferResizeCallback(GLFWwindow* w, int, int) {
    auto* self = reinterpret_cast<Window*>(glfwGetWindowUserPointer(w));
    self->m_resized = true;
}

void Window::keyCallback(GLFWwindow* w, int key, int, int action, int) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(w, GLFW_TRUE);
}

void Window::cursorPosCallback(GLFWwindow* w, double x, double y) {
    auto* self = reinterpret_cast<Window*>(glfwGetWindowUserPointer(w));
    self->m_cursor = { static_cast<float>(x), static_cast<float>(y) };
}

} // namespace vkgfx
