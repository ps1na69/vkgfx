// examples/03_model_loading/main.cpp
// Demonstrates: loading a 3D OBJ model from disk and rendering it.
// Place any .obj file next to the executable (or supply the path as argv[1]).
// A fallback sphere is shown if no model is found.

#include <vkgfx/vkgfx.h>
#include <filesystem>
#include <iostream>

int main(int argc, char** argv) {
    using namespace vkgfx;

    Window   window("03 – Model Loading", 1280, 720);
    window.setCursorLocked(true);

    RendererConfig cfg;
    cfg.ibl.enabled  = false;
    cfg.sun.enabled  = true;
    cfg.sun.intensity= 4.f;

    Renderer renderer(window, cfg);
    Context& ctx = renderer.context();

    // ── Load model ───────────────────────────────────────────────────────────
    std::string modelPath = (argc > 1) ? argv[1] : "model.obj";

    std::shared_ptr<Mesh> model;
    if (std::filesystem::exists(modelPath)) {
        std::cout << "Loading: " << modelPath << "\n";
        model = Mesh::loadOBJ(modelPath, ctx);  // parses OBJ, uploads to GPU
    } else {
        std::cout << "Model not found — showing sphere fallback.\n"
                  << "Usage: ex_03_model_loading <path.obj>\n";
        model = Mesh::createSphere(1.f, 32, 32, ctx);
    }

    // Apply a default grey metallic material
    auto mat = std::make_shared<PBRMaterial>();
    mat->setAlbedo(0.7f, 0.7f, 0.7f)
        .setRoughness(0.4f)
        .setMetallic(0.6f);
    model->setMaterial(mat);

    // ── Camera (orbiting with mouse) ──────────────────────────────────────────
    Camera cam;
    cam.setPosition({0.f, 1.f, -4.f}).setFov(60.f);

    Scene scene;
    scene.setCamera(&cam);
    scene.add(model);

    auto sun = std::make_shared<DirectionalLight>();
    sun->setDirection(-0.4f, -1.f, -0.3f).setIntensity(4.f);
    scene.add(sun);

    // ── Render loop ───────────────────────────────────────────────────────────
    const float MOVE = 4.f, LOOK = 0.1f;
    while (!window.shouldClose()) {
        window.pollEvents();
        if (window.keyPressed(GLFW_KEY_ESCAPE)) break;

        if (window.keyHeld(GLFW_KEY_W)) cam.moveForward( MOVE * 0.016f);
        if (window.keyHeld(GLFW_KEY_S)) cam.moveForward(-MOVE * 0.016f);
        if (window.keyHeld(GLFW_KEY_A)) cam.moveRight  (-MOVE * 0.016f);
        if (window.keyHeld(GLFW_KEY_D)) cam.moveRight  ( MOVE * 0.016f);
        cam.rotateYaw  (window.mouseDX() * LOOK);
        cam.rotatePitch(-window.mouseDY() * LOOK);

        renderer.render(scene);
    }

    vkDeviceWaitIdle(ctx.device());
    model->destroy(ctx);
    renderer.shutdown();
    return 0;
}
