// examples/09_scene/main.cpp
// Demonstrates: building a small scene with multiple mesh types, materials,
// lights, and a collision world — all composed using the Scene API.

#include <vkgfx/vkgfx.h>
#include <chrono>
#include <filesystem>
#include <cmath>
#include <iostream>

int main() {
    using namespace vkgfx;

    Window window("09 – Scene", 1280, 720);
    window.setCursorLocked(true);

    bool hdrFound = std::filesystem::exists("assets/sky.hdr") ||
                    std::filesystem::exists("../../assets/sky.hdr");

    RendererConfig cfg;
    cfg.ibl.enabled   = hdrFound;
    cfg.ibl.intensity = 0.8f;
    cfg.sun.enabled   = true;
    cfg.sun.intensity = 3.f;

    Renderer renderer(window, cfg);
    Context& ctx = renderer.context();

    // ── Camera ────────────────────────────────────────────────────────────────
    Camera cam;
    cam.setPosition({0.f, 2.f, -10.f}).setFov(60.f);

    Scene scene;
    scene.setCamera(&cam);

    std::vector<std::shared_ptr<Mesh>> meshes;

    // ── Floor ─────────────────────────────────────────────────────────────────
    auto floor = Mesh::createBox({8.f, 0.15f, 8.f}, ctx);
    floor->setPosition({0.f, -0.15f, 0.f});
    auto floorMat = std::make_shared<PBRMaterial>();
    floorMat->setAlbedo(0.55f, 0.55f, 0.55f).setRoughness(0.85f);
    floor->setMaterial(floorMat);
    scene.add(floor); meshes.push_back(floor);

    // ── 3 showcase objects ────────────────────────────────────────────────────
    // Gold sphere
    auto goldSphere = Mesh::createSphere(0.7f, 32, 32, ctx);
    goldSphere->setPosition({-3.f, 0.7f, 0.f});
    auto goldMat = std::make_shared<PBRMaterial>();
    goldMat->setAlbedo(1.f, 0.76f, 0.34f).setRoughness(0.1f).setMetallic(1.f);
    goldSphere->setMaterial(goldMat);
    scene.add(goldSphere); meshes.push_back(goldSphere);

    // Rubber box (non-metallic, rough)
    auto rubberBox = Mesh::createBox({0.6f, 0.6f, 0.6f}, ctx);
    rubberBox->setPosition({0.f, 0.6f, 0.f});
    auto rubberMat = std::make_shared<PBRMaterial>();
    rubberMat->setAlbedo(0.1f, 0.6f, 0.9f).setRoughness(0.95f).setMetallic(0.f);
    rubberBox->setMaterial(rubberMat);
    scene.add(rubberBox); meshes.push_back(rubberBox);

    // Emissive beacon sphere
    auto beacon = Mesh::createSphere(0.4f, 16, 16, ctx);
    beacon->setPosition({3.f, 0.4f, 0.f});
    auto beaconMat = std::make_shared<PBRMaterial>();
    beaconMat->setAlbedo(1.f, 0.2f, 0.1f)
              .setRoughness(0.3f)
              .setEmissive(1.f, 0.3f, 0.1f, 4.f); // bright orange emission
    beacon->setMaterial(beaconMat);
    scene.add(beacon); meshes.push_back(beacon);

    // ── Lights ────────────────────────────────────────────────────────────────
    auto sun = std::make_shared<DirectionalLight>();
    sun->setDirection(-0.3f, -1.f, -0.5f).setIntensity(3.f);
    scene.add(sun);

    // Two orbiting point lights
    auto ptA = std::make_shared<PointLight>();
    ptA->setColor(0.2f, 0.5f, 1.f).setIntensity(80.f).setRadius(8.f);
    scene.add(ptA);

    auto ptB = std::make_shared<PointLight>();
    ptB->setColor(1.f, 0.4f, 0.1f).setIntensity(80.f).setRadius(8.f);
    scene.add(ptB);

    // ── Collision world ───────────────────────────────────────────────────────
    CollisionWorld physics(4.f);
    physics.add(floor.get(),      Collider::makeAABB({8.f, 0.15f, 8.f}), true);
    physics.add(goldSphere.get(), Collider::fitSphere(*goldSphere));
    physics.add(rubberBox.get(),  Collider::fitAABB(*rubberBox));
    physics.add(beacon.get(),     Collider::fitSphere(*beacon));

    physics.setOnContact([](const CollisionEvent& ev) {
        (void)ev; // in a game you'd respond here
    });

    std::cout << "WASD/QE = fly, Mouse = look, F1 = IBL, F9 = debug albedo, Esc = quit\n";

    auto last = std::chrono::high_resolution_clock::now();
    float time = 0.f;

    while (!window.shouldClose()) {
        auto now = std::chrono::high_resolution_clock::now();
        float dt = std::chrono::duration<float>(now - last).count();
        last = now;
        time += dt;

        window.pollEvents();
        if (window.keyPressed(GLFW_KEY_ESCAPE)) break;
        if (window.keyPressed(GLFW_KEY_TAB))    window.toggleCursorLock();
        if (window.keyPressed(GLFW_KEY_F11))    window.toggleFullscreen();

        if (window.keyPressed(GLFW_KEY_F1) && hdrFound) {
            cfg.ibl.enabled = !cfg.ibl.enabled;
            renderer.applyConfig(cfg);
        }
        if (window.keyPressed(GLFW_KEY_F9)) {
            cfg.gbufferDebug = (cfg.gbufferDebug == GBufferDebugView::None)
                             ? GBufferDebugView::Albedo : GBufferDebugView::None;
            renderer.applyConfig(cfg);
        }

        // Camera fly
        float d = 5.f * dt;
        if (window.keyHeld(GLFW_KEY_W)) cam.moveForward( d);
        if (window.keyHeld(GLFW_KEY_S)) cam.moveForward(-d);
        if (window.keyHeld(GLFW_KEY_A)) cam.moveRight  (-d);
        if (window.keyHeld(GLFW_KEY_D)) cam.moveRight  ( d);
        if (window.keyHeld(GLFW_KEY_Q)) cam.moveUp     (-d);
        if (window.keyHeld(GLFW_KEY_E)) cam.moveUp     ( d);
        cam.rotateYaw  ( window.mouseDX() * 0.1f);
        cam.rotatePitch(-window.mouseDY() * 0.1f);

        // Spin the box
        rubberBox->setRotation(glm::quat(glm::vec3(0.f, time * 0.8f, 0.f)));

        // Pulse beacon scale
        float pulse = 1.f + 0.15f * std::sin(time * 4.f);
        beacon->setScale({pulse, pulse, pulse});

        // Orbit point lights
        ptA->setPosition({std::cos(time * 0.6f) * 4.f, 2.f, std::sin(time * 0.6f) * 4.f});
        ptB->setPosition({std::cos(time * 0.6f + 3.14f) * 4.f, 1.f,
                           std::sin(time * 0.6f + 3.14f) * 4.f});

        // Run collision
        physics.update(false); // false = no automatic position fix; scene is static

        renderer.render(scene);
    }

    vkDeviceWaitIdle(ctx.device());
    for (auto& m : meshes) m->destroy(ctx);
    renderer.shutdown();
    return 0;
}
