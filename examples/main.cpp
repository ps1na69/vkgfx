// examples/main.cpp
// sky.hdr is OPTIONAL. Without it IBL is disabled and the scene renders
// with sun + ambient only. Press F1 at runtime to toggle IBL on/off
// (only works if sky.hdr was found at startup).

#include <vkgfx/vkgfx.h>
#include <iostream>
#include <filesystem>
#include <chrono>

static void printControls(bool iblAvailable) {
    std::cout
        << "\n=== vkgfx example ===\n"
        << "  WASD / QE   Move\n"
        << "  Mouse       Look\n"
        << "  F1          Toggle IBL"
        << (iblAvailable ? "" : "  [DISABLED — sky.hdr not found]") << "\n"
        << "  F2          Toggle Sun\n"
        << "  F3          G-buffer: Albedo\n"
        << "  F4          G-buffer: Normal\n"
        << "  F5          G-buffer: Roughness\n"
        << "  F6          G-buffer: Metallic\n"
        << "  F7          G-buffer: Depth\n"
        << "  F8          G-buffer: AO\n"
        << "  F9          Full lighting (reset debug)\n"
        << "  Esc         Quit\n\n";
}

// Resolve a path relative to the executable first, then CWD.
// This handles both running from build/Debug/ (VS default) and from CWD.
static std::string resolveAssetPath(const std::string& rel) {
    // 1. Relative to CWD
    if (std::filesystem::exists(rel)) return rel;
    // 2. Give up — return original (caller will handle absence)
    return rel;
}

int main(int argc, char** argv) {
    using namespace vkgfx;

    // ── Build config ──────────────────────────────────────────────────────────
    RendererConfig cfg;
    cfg.shaderDir          = "shaders";
    cfg.assetDir           = "assets";
    cfg.ibl.hdrPath        = cfg.assetDir + "/sky.hdr";
    cfg.ibl.envMapSize     = 512;
    cfg.ibl.irradianceSize = 32;
    cfg.ibl.intensity      = 1.0f;
    cfg.sun.enabled        = true;
    cfg.sun.direction[0]   = -0.4f;
    cfg.sun.direction[1]   = -1.0f;
    cfg.sun.direction[2]   = -0.3f;
    cfg.sun.intensity      = 5.0f;
    cfg.msaa               = MSAASamples::x4;
    cfg.vsync              = true;
    cfg.gbufferDebug       = GBufferDebugView::None;

    // Override from JSON file if provided as first argument
    if (argc > 1) {
        if (!std::filesystem::exists(argv[1])) {
            std::cerr << "[example] Config not found: " << argv[1] << "\n";
            return 1;
        }
        cfg = RendererConfig::fromFile(argv[1]);
    }

    // ── Resolve shaderDir ────────────────────────────────────────────────────
    // The renderer also does this, but resolving here gives a clear startup log.
    {
        const std::string probe = "/gbuffer.vert.spv";
        std::vector<std::string> candidates = {
            cfg.shaderDir, "shaders", "../shaders",
            "build/shaders", "build/Debug/shaders", "build/Release/shaders",
            "../build/shaders", "../build/Debug/shaders", "../build/Release/shaders",
        };
        for (auto& c : candidates) {
            if (std::filesystem::exists(c + probe)) {
                cfg.shaderDir = c;
                std::cout << "[example] Resolved shaderDir: " << c << "\n";
                break;
            }
        }
    }

    // ── sky.hdr is OPTIONAL ───────────────────────────────────────────────────
    bool hdrFound = false;
    {
        std::vector<std::string> candidates = {
            cfg.ibl.hdrPath,
            "assets/sky.hdr",
            "../assets/sky.hdr",
            "../../assets/sky.hdr",
        };
        for (auto& c : candidates) {
            if (std::filesystem::exists(c)) {
                cfg.ibl.hdrPath = c;
                hdrFound = true;
                std::cout << "[example] Found sky.hdr at: " << c << "\n";
                break;
            }
        }
    }

    if (!hdrFound) {
        std::cout << "[example] sky.hdr not found — IBL disabled.\n"
                  << "          Place an equirectangular HDR at assets/sky.hdr to enable IBL.\n"
                  << "          Download free HDRs from https://polyhaven.com/hdris\n\n";
        cfg.ibl.enabled = false;
    } else {
        cfg.ibl.enabled = true;
    }

    printControls(hdrFound);

    // ── Window + Renderer ─────────────────────────────────────────────────────
    Window   window("vkgfx — deferred PBR + IBL", 1280, 720);
    Renderer renderer(window, cfg);

    // ── Scene ─────────────────────────────────────────────────────────────────
    Camera cam;
    cam.setPosition({0.f, 0.f, -4.f}).setFov(60.f);

    Scene scene;
    scene.setCamera(&cam);

    // 4x4 sphere grid: roughness increases right, metallic increases up
    // Keep handles so we can explicitly free GPU resources before shutdown.
    std::vector<std::shared_ptr<vkgfx::Mesh>> spheres;
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            auto sphere = vkgfx::Mesh::createSphere(0.3f, 32, 32, renderer.context());
            sphere->setPosition({(col - 1.5f) * 0.9f,
                                  (row - 1.5f) * 0.9f, 0.f});
            auto mat = std::make_shared<vkgfx::PBRMaterial>();
            mat->setAlbedo(0.8f, 0.3f, 0.2f)
                .setRoughness(static_cast<float>(col) / 3.f)
                .setMetallic (static_cast<float>(row) / 3.f);
            sphere->setMaterial(mat);
            spheres.push_back(sphere);
            scene.add(sphere);
        }
    }

    auto sun = std::make_shared<DirectionalLight>();
    sun->setDirection(cfg.sun.direction[0], cfg.sun.direction[1], cfg.sun.direction[2])
        .setIntensity(cfg.sun.intensity)
        .setEnabled(cfg.sun.enabled);
    scene.add(sun);

    // ── Main loop ─────────────────────────────────────────────────────────────
    const float MOVE  = 3.0f;
    const float LOOK  = 0.1f;
    auto lastTime = std::chrono::high_resolution_clock::now();

    while (!window.shouldClose()) {
        auto now  = std::chrono::high_resolution_clock::now();
        float dt  = std::chrono::duration<float>(now - lastTime).count();
        lastTime  = now;

        window.pollEvents();

        float spd = MOVE * dt;
        if (window.keyHeld(GLFW_KEY_W)) cam.moveForward( spd);
        if (window.keyHeld(GLFW_KEY_S)) cam.moveForward(-spd);
        if (window.keyHeld(GLFW_KEY_A)) cam.moveRight  (-spd);
        if (window.keyHeld(GLFW_KEY_D)) cam.moveRight  ( spd);
        if (window.keyHeld(GLFW_KEY_Q)) cam.moveUp     (-spd);
        if (window.keyHeld(GLFW_KEY_E)) cam.moveUp     ( spd);

        cam.rotateYaw  ( window.mouseDX() * LOOK);
        cam.rotatePitch(-window.mouseDY() * LOOK);

        if (window.keyPressed(GLFW_KEY_ESCAPE)) break;

        // IBL toggle — only useful if sky.hdr was found at startup
        if (window.keyPressed(GLFW_KEY_F1) && hdrFound) {
            cfg.ibl.enabled = !cfg.ibl.enabled;
            std::cout << "IBL: " << (cfg.ibl.enabled ? "ON" : "OFF") << "\n";
            renderer.applyConfig(cfg);
        }

        if (window.keyPressed(GLFW_KEY_F2)) {
            cfg.sun.enabled = !cfg.sun.enabled;
            sun->setEnabled(cfg.sun.enabled);
            std::cout << "Sun: " << (cfg.sun.enabled ? "ON" : "OFF") << "\n";
        }

        static const std::pair<int, GBufferDebugView> debugKeys[] = {
            {GLFW_KEY_F3, GBufferDebugView::Albedo},
            {GLFW_KEY_F4, GBufferDebugView::Normal},
            {GLFW_KEY_F5, GBufferDebugView::Roughness},
            {GLFW_KEY_F6, GBufferDebugView::Metallic},
            {GLFW_KEY_F7, GBufferDebugView::Depth},
            {GLFW_KEY_F8, GBufferDebugView::AO},
            {GLFW_KEY_F9, GBufferDebugView::None},
        };
        for (auto& [key, view] : debugKeys) {
            if (window.keyPressed(key)) {
                cfg.gbufferDebug = view;
                renderer.applyConfig(cfg);
            }
        }

        renderer.render(scene);
    }

    // Free GPU buffers for all meshes before renderer shutdown.
    // Renderer must still be alive (device valid) when destroy() is called.
    for (auto& s : spheres)
        s->destroy(renderer.context());
    spheres.clear();

    renderer.shutdown();
    std::cout << "[example] clean exit\n";
    return 0;
}
