// examples/03_model_loading/main.cpp
// Demonstrates: auto-scanning assets/models/ for an OBJ file and loading it.
//
// USAGE:
//   1. Create:  assets/models/  in the project root
//   2. Put any  .obj  file inside (with its .mtl if it has one)
//   3. Run — the first .obj found is loaded automatically
//
// Falls back to a sphere if no model is found.

#include <vkgfx/vkgfx.h>
#include <filesystem>
#include <iostream>
#include <chrono>

namespace fs = std::filesystem;

// Scan a directory for the first .obj file, searching several candidate paths
// relative to the executable (which is deep inside the build tree).
static std::string findModel() {
    for (auto& candidate : {
            "assets/models",
            "../assets/models",
            "../../assets/models",
            "../../../assets/models",
            "../../../../assets/models",
            "../../../../../assets/models",
            "assets",           // flat fallback
            "../../../../assets",
        }) {
        if (!fs::exists(candidate)) continue;
        for (auto& entry : fs::directory_iterator(candidate)) {
            if (!entry.is_regular_file()) continue;
            std::string ext = entry.path().extension().string();
            // lowercase comparison
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".obj") {
                std::cout << "[03_model] Found: " << entry.path() << "\n";
                return entry.path().string();
            }
        }
    }
    return "";
}

int main() {
    using namespace vkgfx;

    Window window("03 – Model Loading", 1920, 1080);
    window.setCursorLocked(true);
    window.setFullscreen(true);

    // Detect sky.hdr for IBL
    bool hdrFound = false;
    std::string hdrPath;
    for (auto& c : {"assets/sky.hdr","../../assets/sky.hdr",
                    "../../../../assets/sky.hdr","../../../../../assets/sky.hdr"}) {
        if (fs::exists(c)) { hdrPath = c; hdrFound = true; break; }
    }

    RendererConfig cfg;
	cfg.vsync = false;
    cfg.ibl.enabled   = hdrFound;
    cfg.ibl.hdrPath   = hdrPath;
    cfg.ibl.intensity = 0.8f;
    cfg.sun.enabled   = true;
    cfg.sun.intensity = 3.f;
    cfg.sun.direction[0] = -0.4f;
    cfg.sun.direction[1] = -1.0f;
    cfg.sun.direction[2] = -0.3f;
	cfg.msaa = MSAASamples::x8;

    Renderer renderer(window, cfg);
    Context& ctx = renderer.context();

    // ── Find and load model ───────────────────────────────────────────────────
    std::string modelPath = findModel();

    std::shared_ptr<Mesh> model;
    if (!modelPath.empty()) {
        model = Mesh::loadOBJ(modelPath, ctx);
        std::cout << "[03_model] Loaded successfully.\n";
    } else {
        std::cout << "[03_model] No .obj found — showing sphere fallback.\n"
                  << "  Create assets/models/ and place a .obj file there.\n";
        model = Mesh::createSphere(1.f, 32, 32, ctx);
    }

    // Default grey metallic material (overridden if the OBJ has its own materials)
    auto mat = std::make_shared<PBRMaterial>();
    mat->setAlbedo(0.7f, 0.7f, 0.72f)
        .setRoughness(0.5f)
        .setMetallic(1.0f);
    model->setMaterial(mat);

    // Centre the model at origin
    model->setPosition({0.f, 0.f, 0.f});

    // ── Scene ─────────────────────────────────────────────────────────────────
    Camera cam;
    cam.setPosition({0.f, 0.f, 4.f}).setFov(60.f);

    Scene scene;
    scene.setCamera(&cam);
    scene.add(model);

    auto sun = std::make_shared<DirectionalLight>();
    sun->setDirection(-0.4f, -1.f, -0.3f).setIntensity(10.f);
    scene.add(sun);

	auto ptLight = std::make_shared<PointLight>();
	ptLight->setPosition({ 1.f, 1.f, 1.f }).setColor(1.f, 0.8f, 0.6f).setIntensity(30.f).setRadius(10.f);
	scene.add(ptLight);

    std::cout << "WASD = move  Mouse = look  F11 = fullscreen  Esc = quit\n";

    auto last = std::chrono::high_resolution_clock::now();
    // ── Main loop ─────────────────────────────────────────────────────────────
    const float MOVE = 5.0f;
    const float LOOK = 0.10f;
    auto lastTime = std::chrono::high_resolution_clock::now();
    float totalTime = 0.f;

    // FPS counter state
    float  fpsTimer = 0.f;
    int    fpsFrames = 0;
    float  fpsDisplay = 0.f;

    while (!window.shouldClose()) {
        auto now = std::chrono::high_resolution_clock::now();
        float dt = std::chrono::duration<float>(now - last).count();
        last = now;

        ++fpsFrames;
        fpsTimer += dt;
        if (fpsTimer >= 0.5f) {
            fpsDisplay = fpsFrames / fpsTimer;
            fpsTimer = 0.f;
            fpsFrames = 0;
            std::ostringstream oss;
            oss << "vkgfx — deferred PBR + IBL  |  "
                << std::fixed << std::setprecision(1) << fpsDisplay << " fps  |  "
                << std::setprecision(2) << (1000.f / fpsDisplay) << " ms";
            window.setTitle(oss.str());
        }

        window.pollEvents();
        if (window.keyPressed(GLFW_KEY_ESCAPE)) break;
        if (window.keyPressed(GLFW_KEY_F11))    window.toggleFullscreen();
        if (window.keyPressed(GLFW_KEY_TAB))    window.toggleCursorLock();

        float d = 4.f * dt;
        if (window.keyHeld(GLFW_KEY_W)) cam.moveForward( d);
        if (window.keyHeld(GLFW_KEY_S)) cam.moveForward(-d);
        if (window.keyHeld(GLFW_KEY_A)) cam.moveRight  (-d);
        if (window.keyHeld(GLFW_KEY_D)) cam.moveRight  ( d);
        if (window.keyHeld(GLFW_KEY_Q)) cam.moveUp     (-d);
        if (window.keyHeld(GLFW_KEY_E)) cam.moveUp     ( d);
        cam.rotateYaw  ( window.mouseDX() * 0.1f);
        cam.rotatePitch(-window.mouseDY() * 0.1f);

        renderer.render(scene);
    }

    vkDeviceWaitIdle(ctx.device());
    model->destroy(ctx);
    renderer.shutdown();
    return 0;
}
