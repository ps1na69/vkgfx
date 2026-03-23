// examples/02_triangle/main.cpp
// Demonstrates: rendering a basic 3D shape using the engine renderer.
// Shows the minimal setup: Window → RendererConfig → Renderer → Mesh → Scene → render loop.
// A rotating cube is used because the engine is deferred/3D and culls back faces.

#include <vkgfx/vkgfx.h>
#include <chrono>

int main() {
    using namespace vkgfx;

    // ── Window ────────────────────────────────────────────────────────────────
    Window window("02 – Basic Shape", 800, 600);
    window.setCursorLocked(false);

    // ── Renderer config ───────────────────────────────────────────────────────
    RendererConfig cfg;
    cfg.ibl.enabled   = false;         // no sky.hdr needed for this example
    cfg.sun.enabled   = true;
    cfg.sun.intensity = 3.f;
    cfg.sun.direction[0] = -0.4f;
    cfg.sun.direction[1] = -1.0f;
    cfg.sun.direction[2] = -0.3f;

    Renderer renderer(window, cfg);

    // ── Camera ────────────────────────────────────────────────────────────────
    Camera cam;
    cam.setPosition({0.f, 0.5f, -3.f}).setFov(60.f);

    // ── Cube mesh ─────────────────────────────────────────────────────────────
    // createBox(halfExtents, context) — halfExtents is half the size on each axis.
    auto cube = Mesh::createBox({0.7f, 0.7f, 0.7f}, renderer.context());

    // Assign a PBR material: reddish, slightly rough, non-metallic
    auto mat = std::make_shared<PBRMaterial>();
    mat->setAlbedo(0.9f, 0.25f, 0.15f)
        .setRoughness(0.4f)
        .setMetallic(0.f);
    cube->setMaterial(mat);

    // ── Scene ─────────────────────────────────────────────────────────────────
    Scene scene;
    scene.setCamera(&cam);
    scene.add(cube);               // add the cube to the scene graph

    // ── Render loop ───────────────────────────────────────────────────────────
    auto last = std::chrono::high_resolution_clock::now();
    float time = 0.f;

    while (!window.shouldClose()) {
        auto now = std::chrono::high_resolution_clock::now();
        float dt = std::chrono::duration<float>(now - last).count();
        last = now;
        time += dt;

        window.pollEvents();
        if (window.keyPressed(GLFW_KEY_ESCAPE)) break;

        // Rotate the cube over time so all faces are visible
        cube->setRotation(glm::quat(glm::vec3(time * 0.5f, time * 0.8f, 0.f)));

        renderer.render(scene);    // draw G-buffer → lighting → tonemap
    }

    // Free GPU resources before shutdown
    vkDeviceWaitIdle(renderer.context().device());
    cube->destroy(renderer.context());
    renderer.shutdown();
    return 0;
}
