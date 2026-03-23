// examples/06_textures/main.cpp
// Demonstrates: loading textures via TextureCache and applying them to a mesh
// using PBRMaterial's texture slots (albedo, normal, RMA).
//
// Usage: ex_06_textures [albedo.png] [normal.png] [rma.png]
// Defaults to procedural solid-colour fallbacks if files are missing.

#include <vkgfx/vkgfx.h>
#include <filesystem>
#include <iostream>
#include <chrono>

int main(int argc, char** argv) {
    using namespace vkgfx;

    Window   window("06 – Textures", 1280, 720);
    window.setCursorLocked(true);

    RendererConfig cfg;
    cfg.ibl.enabled  = false;
    cfg.sun.enabled  = true;
    cfg.sun.intensity= 4.f;

    Renderer   renderer(window, cfg);
    Context&   ctx = renderer.context();

    // ── TextureCache: deduplicates loads, handles missing files gracefully ────
    TextureCache textures(ctx);

    // Resolve texture paths from command line or use defaults
    std::string albedoPath = (argc > 1) ? argv[1] : "albedo.png";
    std::string normalPath = (argc > 2) ? argv[2] : "normal.png";
    std::string rmaPath    = (argc > 3) ? argv[3] : "rma.png";

    // load() returns a magenta 1×1 placeholder if the file is not found
    auto albedoTex = textures.load(albedoPath);
    auto normalTex = textures.load(normalPath);
    auto rmaTex    = textures.load(rmaPath);

    // ── 3 spheres: textured, albedo-only, plain material ─────────────────────
    Scene scene;
    Camera cam;
    cam.setPosition({0.f, 0.f, -5.f}).setFov(60.f);
    scene.setCamera(&cam);

    std::vector<std::shared_ptr<Mesh>> meshes;

    // Left sphere: full texture set
    {
        auto s = Mesh::createSphere(0.8f, 32, 32, ctx);
        s->setPosition({-2.f, 0.f, 0.f});
        auto m = std::make_shared<PBRMaterial>();
        m->setAlbedoTexture(albedoTex)
         .setNormalTexture(normalTex)
         .setRMATexture(rmaTex);
        s->setMaterial(m);
        scene.add(s); meshes.push_back(s);
    }

    // Centre sphere: albedo texture only, manual roughness/metallic
    {
        auto s = Mesh::createSphere(0.8f, 32, 32, ctx);
        s->setPosition({0.f, 0.f, 0.f});
        auto m = std::make_shared<PBRMaterial>();
        m->setAlbedoTexture(albedoTex)
         .setRoughness(0.3f)
         .setMetallic(0.7f);
        s->setMaterial(m);
        scene.add(s); meshes.push_back(s);
    }

    // Right sphere: no textures, plain orange with emissive glow
    {
        auto s = Mesh::createSphere(0.8f, 32, 32, ctx);
        s->setPosition({2.f, 0.f, 0.f});
        auto m = std::make_shared<PBRMaterial>();
        m->setAlbedo(1.f, 0.4f, 0.1f)
         .setRoughness(0.5f)
         .setMetallic(0.f)
         .setEmissive(1.f, 0.3f, 0.f, 1.5f);  // orange emissive glow
        s->setMaterial(m);
        scene.add(s); meshes.push_back(s);
    }

    auto sun = std::make_shared<DirectionalLight>();
    sun->setDirection(-0.4f, -1.f, -0.3f).setIntensity(4.f);
    scene.add(sun);

    std::cout << "Left: full textures | Centre: albedo only | Right: emissive\n";
    std::cout << "WASD = move, Mouse = look, Escape = quit\n";

    auto last = std::chrono::high_resolution_clock::now();
    while (!window.shouldClose()) {
        auto now = std::chrono::high_resolution_clock::now();
        float dt = std::chrono::duration<float>(now - last).count();
        last = now;

        window.pollEvents();
        if (window.keyPressed(GLFW_KEY_ESCAPE)) break;

        float d = 4.f * dt;
        if (window.keyHeld(GLFW_KEY_W)) cam.moveForward( d);
        if (window.keyHeld(GLFW_KEY_S)) cam.moveForward(-d);
        if (window.keyHeld(GLFW_KEY_A)) cam.moveRight  (-d);
        if (window.keyHeld(GLFW_KEY_D)) cam.moveRight  ( d);
        cam.rotateYaw  ( window.mouseDX() * 0.1f);
        cam.rotatePitch(-window.mouseDY() * 0.1f);

        renderer.render(scene);
    }

    vkDeviceWaitIdle(ctx.device());
    for (auto& m : meshes) m->destroy(ctx);
    textures.clear();          // destroy all loaded textures
    renderer.shutdown();
    return 0;
}
