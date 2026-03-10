#include <vkgfx/vkgfx.h>

using namespace vkgfx;

int main() {
    Window window("Назва", 800, 600);
    RendererSettings rs;
    rs.clearColor = {0, 1, 1, 1};
    rs.msaa = MSAASamples::x1;
    rs.wireframe = false;
    rs.vsync = true;
    rs.workerThreads = 0;
    rs.validation = true;
    Renderer renderer(window, rs);

    Camera camera;
    camera.setPosition({0.f, 1.5f, -5.f})
          .setYaw(90.f)
          .setPitch(-15.f)
          .setFov(70.f)
          .setAspect(window.getAspectRatio());

    Scene scene(&camera);
    scene.setAmbient({0,1,1}, 1);

    while (!window.shouldClose()) {
        window.pollEvents();
        renderer.render(scene);
    }
    renderer.shutdown(&scene);
    return 0;
}