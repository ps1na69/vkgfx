// examples/05_lighting/main.cpp
// Demonstrates: directional sun light + multiple point lights.
// Press 1–4 to toggle each point light. Press S to toggle the sun.
// Press F1 to toggle IBL if sky.hdr is present in assets/.
// The 5×5 sphere grid shows how roughness and metallic interact with lighting.

#include <vkgfx/vkgfx.h>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <cmath>

int main() {
    using namespace vkgfx;

    Window window("05 – Lighting", 1280, 720);
    window.setCursorLocked(true);

    // Detect sky.hdr
    bool hdrFound = std::filesystem::exists("assets/sky.hdr") ||
                    std::filesystem::exists("../../assets/sky.hdr");

    RendererConfig cfg;
    cfg.ibl.enabled   = hdrFound;
    cfg.ibl.hdrPath   = hdrFound ? "assets/sky.hdr" : "";
    cfg.ibl.intensity = 0.5f;
    cfg.sun.enabled   = true;
    cfg.sun.intensity = 3.f;

    Renderer renderer(window, cfg);
    Context& ctx = renderer.context();

    // ── Camera ────────────────────────────────────────────────────────────────
    Camera cam;
    cam.setPosition({0.f, 2.f, -8.f}).setFov(60.f);

    Scene scene;
    scene.setCamera(&cam);

    // ── 4×4 metallic/roughness sphere grid ────────────────────────────────────
    std::vector<std::shared_ptr<Mesh>> meshes;
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            auto s = Mesh::createSphere(0.38f, 24, 24, ctx);
            s->setPosition({(col - 1.5f) * 1.1f, (row - 1.5f) * 1.1f, 0.f});
            auto m = std::make_shared<PBRMaterial>();
            m->setAlbedo(0.8f, 0.3f, 0.1f)
             .setRoughness(0.1f + col * 0.28f)   // 0.1 → 0.95 left→right
             .setMetallic (row / 3.f);            // 0.0 → 1.0 bottom→top
            s->setMaterial(m);
            scene.add(s);
            meshes.push_back(s);
        }
    }

    // ── Directional sun ───────────────────────────────────────────────────────
    auto sun = std::make_shared<DirectionalLight>();
    sun->setDirection(-0.4f, -1.f, -0.3f).setColor(1.f, 0.95f, 0.85f).setIntensity(3.f);
    scene.add(sun);

    // ── 4 coloured point lights (toggleable) ──────────────────────────────────
    const glm::vec3 ptCol[4] = {{1.f,0.1f,0.1f},{0.1f,0.3f,1.f},
                                 {0.1f,1.f,0.2f},{1.f,0.8f,0.1f}};
    std::shared_ptr<PointLight> ptLights[4];
    for (int i = 0; i < 4; ++i) {
        ptLights[i] = std::make_shared<PointLight>();
        ptLights[i]->setColor(ptCol[i].r, ptCol[i].g, ptCol[i].b)
                    .setIntensity(50.f).setRadius(6.f);
        scene.add(ptLights[i]);
    }

    std::cout << "Keys: 1-4 = toggle point lights, S = toggle sun, F1 = toggle IBL\n";

    auto lastTime = std::chrono::high_resolution_clock::now();
    float time = 0.f;

    while (!window.shouldClose()) {
        auto now = std::chrono::high_resolution_clock::now();
        float dt = std::chrono::duration<float>(now - lastTime).count();
        lastTime = now;
        time += dt;

        window.pollEvents();
        if (window.keyPressed(GLFW_KEY_ESCAPE)) break;

        // Toggle each point light with 1-4
        for (int i = 0; i < 4; ++i)
            if (window.keyPressed(GLFW_KEY_1 + i))
                ptLights[i]->setEnabled(!ptLights[i]->enabled());

        // Toggle sun
        if (window.keyPressed(GLFW_KEY_S))
            sun->setEnabled(!sun->enabled());

        // Toggle IBL (only if sky.hdr is present)
        if (window.keyPressed(GLFW_KEY_F1) && hdrFound) {
            cfg.ibl.enabled = !cfg.ibl.enabled;
            renderer.applyConfig(cfg);
        }

        // Camera controls
        float d = 4.f * dt;
        if (window.keyHeld(GLFW_KEY_W)) cam.moveForward( d);
        if (window.keyHeld(GLFW_KEY_S)) cam.moveForward(-d);
        if (window.keyHeld(GLFW_KEY_A)) cam.moveRight  (-d);
        if (window.keyHeld(GLFW_KEY_D)) cam.moveRight  ( d);
        cam.rotateYaw  ( window.mouseDX() * 0.1f);
        cam.rotatePitch(-window.mouseDY() * 0.1f);

        // Orbit point lights
        for (int i = 0; i < 4; ++i) {
            float angle = time * 0.7f + i * 1.57f;
            ptLights[i]->setPosition({std::cos(angle) * 3.5f,
                                       1.5f + std::sin(time * 0.4f + i) * 1.f,
                                       std::sin(angle) * 3.5f});
        }

        renderer.render(scene);
    }

    vkDeviceWaitIdle(ctx.device());
    for (auto& m : meshes) m->destroy(ctx);
    renderer.shutdown();
    return 0;
}
