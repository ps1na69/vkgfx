#pragma once
#include "types.h"

namespace vkgfx {

struct WindowSettings {
    uint32_t    width       = 1280;
    uint32_t    height      = 720;
    std::string title       = "VKGFX Application";
    bool        resizable   = true;
    bool        fullscreen  = false;
    bool        vsync       = true;
};

class Window {
public:
    explicit Window(const WindowSettings& settings = {});
    Window(std::string_view title, uint32_t w, uint32_t h, bool resizable = true);
    ~Window();

    // Non-copyable, movable
    Window(const Window&)            = delete;
    Window& operator=(const Window&) = delete;
    Window(Window&&)                 = default;
    Window& operator=(Window&&)      = default;

    void pollEvents()  const { glfwPollEvents(); }
    [[nodiscard]] bool shouldClose() const { return glfwWindowShouldClose(m_handle); }
    void close() { glfwSetWindowShouldClose(m_handle, GLFW_TRUE); }

    //Surface
    [[nodiscard]] VkSurfaceKHR createSurface(VkInstance instance) const;


    [[nodiscard]] std::pair<uint32_t, uint32_t> getFramebufferSize() const;
    [[nodiscard]] float getAspectRatio() const;
    [[nodiscard]] float getTime() const { return static_cast<float>(glfwGetTime()); }
    [[nodiscard]] GLFWwindow* handle() const { return m_handle; }
    [[nodiscard]] bool wasResized() const { return m_resized; }
    void resetResizeFlag() { m_resized = false; }
    [[nodiscard]] const WindowSettings& settings() const { return m_settings; }


    [[nodiscard]] bool isKeyPressed(int key) const { return glfwGetKey(m_handle, key) == GLFW_PRESS; }
    [[nodiscard]] Vec2 getCursorPos() const;
    void setCursorVisible(bool visible);
    void setTitle(std::string_view title);

    //Vulkan extensions
    [[nodiscard]] static std::vector<const char*> getRequiredInstanceExtensions();

private:
    static void framebufferResizeCallback(GLFWwindow* w, int, int);
    static void keyCallback(GLFWwindow* w, int key, int, int action, int);
    static void cursorPosCallback(GLFWwindow* w, double x, double y);

    GLFWwindow*    m_handle   = nullptr;
    WindowSettings m_settings;
    bool           m_resized  = false;
    Vec2           m_cursor   = {0.f, 0.f};

    static int s_windowCount;
};

} // namespace vkgfx
