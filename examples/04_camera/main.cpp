// examples/04_camera/main.cpp
// Demonstrates: all Camera controls — WASD movement, mouse look,
// sprint (Shift), vertical movement (Q/E), and dynamic FOV zoom (scroll).

#include <vkgfx/vkgfx.h>
#include <GLFW/glfw3.h>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <iostream>

int main() {
    using namespace vkgfx;

    Window window("04 – Camera", 1280, 720);
    // Cursor captured for first-person look
    window.setCursorLocked(true);

    RendererConfig cfg;
    cfg.ibl.enabled  = false;
    cfg.sun.enabled  = true;
    cfg.sun.intensity= 3.f;

    Renderer renderer(window, cfg);
    Context& ctx = renderer.context();

    // ── Scene: a grid of boxes to navigate around ─────────────────────────────
    Scene scene;
    Camera cam;
    cam.setPosition({0.f, 1.f, -6.f}).setFov(60.f);
    scene.setCamera(&cam);

    std::vector<std::shared_ptr<Mesh>> boxes;
    for (int x = -4; x <= 4; x += 2) {
        for (int z = -4; z <= 4; z += 2) {
            auto b = Mesh::createBox({0.4f, 0.4f, 0.4f}, ctx);
            b->setPosition({(float)x, 0.f, (float)z});
            auto m = std::make_shared<PBRMaterial>();
            float t = (x + z + 8.f) / 16.f;
            m->setAlbedo(t, 0.5f, 1.f - t).setRoughness(0.3f).setMetallic(0.f);
            b->setMaterial(m);
            scene.add(b);
            boxes.push_back(b);
        }
    }

    auto sun = std::make_shared<DirectionalLight>();
    sun->setDirection(-0.4f, -1.f, -0.3f);
    scene.add(sun);

    // ── Camera state ──────────────────────────────────────────────────────────
    float fov      = 60.f;
    float baseSpeed= 4.f;

    auto last = std::chrono::high_resolution_clock::now();

    while (!window.shouldClose()) {
        auto now = std::chrono::high_resolution_clock::now();
        float dt = std::chrono::duration<float>(now - last).count();
        last = now;

        window.pollEvents();
        if (window.keyPressed(GLFW_KEY_ESCAPE)) break;

        // Sprint with Left Shift — doubles movement speed
        float speed = window.keyHeld(GLFW_KEY_LEFT_SHIFT) ? baseSpeed * 2.5f : baseSpeed;
        float dist  = speed * dt;

        // WASD + QE movement
        if (window.keyHeld(GLFW_KEY_W)) cam.moveForward( dist);
        if (window.keyHeld(GLFW_KEY_S)) cam.moveForward(-dist);
        if (window.keyHeld(GLFW_KEY_A)) cam.moveRight  (-dist);
        if (window.keyHeld(GLFW_KEY_D)) cam.moveRight  ( dist);
        if (window.keyHeld(GLFW_KEY_Q)) cam.moveUp     (-dist);
        if (window.keyHeld(GLFW_KEY_E)) cam.moveUp     ( dist);

        // Mouse look
        cam.rotateYaw  ( window.mouseDX() * 0.1f);
        cam.rotatePitch(-window.mouseDY() * 0.1f);

        // FOV zoom (simulate scroll with Z/X keys)
        if (window.keyHeld(GLFW_KEY_Z)) fov = std::max(20.f, fov - 40.f * dt);
        if (window.keyHeld(GLFW_KEY_X)) fov = std::min(120.f, fov + 40.f * dt);
        cam.setFov(fov);

        // Tab toggles cursor lock (useful for debugging)
        if (window.keyPressed(GLFW_KEY_TAB))
            window.toggleCursorLock();

        // Update title with position
        auto p = cam.position();
        std::ostringstream ss;
        ss << "04 – Camera  pos("
           << std::fixed << std::setprecision(1)
           << p.x << ", " << p.y << ", " << p.z << ")  fov=" << (int)fov;
        window.setTitle(ss.str());

        renderer.render(scene);
    }

    vkDeviceWaitIdle(ctx.device());
    for (auto& b : boxes) b->destroy(ctx);
    renderer.shutdown();
    return 0;
}
