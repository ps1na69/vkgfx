// examples/main.cpp
// Deferred PBR + IBL + PCF shadows demo.
//
// Scene layout:
//   - Ground plane (large flat sphere cluster forming a floor)
//   - 5x5 sphere grid: roughness increases right (0→1), metallic increases up (0→1)
//   - 4 colored point lights orbiting the scene
//   - 1 large showcase sphere in the center
//   - Directional sun with shadows
//
// sky.hdr is optional — IBL gracefully disabled if absent.

#include <vkgfx/vkgfx.h>
#include <iostream>
#include <filesystem>
#include <chrono>
#include <vector>
#include <cmath>
#include <sstream>
#include <iomanip>

using namespace vkgfx;

// ── Path resolution helpers ───────────────────────────────────────────────────
static std::string resolveFirst(std::vector<std::string> candidates) {
    for (auto& c : candidates)
        if (std::filesystem::exists(c)) return c;
    return candidates[0];
}

static void printHelp(bool hdrFound) {
    std::cout
        << "\n=== vkgfx demo ===\n"
        << "  WASD / QE   Move\n"
        << "  Mouse       Look\n"
        << "  F1          Toggle IBL" << (hdrFound ? "" : " [no sky.hdr]") << "\n"
        << "  F2          Toggle Sun\n"
        << "  F3-F9       G-buffer debug views\n"
        << "  Esc         Quit\n\n";
}

int main(int argc, char** argv) {
    // ── Config ────────────────────────────────────────────────────────────────
    RendererConfig cfg;
    cfg.vsync              = false;
    cfg.ibl.envMapSize     = 512;
    cfg.ibl.irradianceSize = 32;
    cfg.ibl.intensity      = 1.0f;
    cfg.sun.enabled        = true;
    cfg.sun.direction[0]   = -0.4f;
    cfg.sun.direction[1]   = -1.0f;
    cfg.sun.direction[2]   = -0.3f;
    cfg.sun.intensity      = 4.0f;
    cfg.sun.color[0] = cfg.sun.color[1] = cfg.sun.color[2] = 1.f;
    cfg.gbufferDebug       = GBufferDebugView::None;

    if (argc > 1) cfg = RendererConfig::fromFile(argv[1]);

    // Resolve shader dir
    cfg.shaderDir = resolveFirst({
        cfg.shaderDir, "shaders", "../shaders",
        "build/shaders", "build/Debug/shaders", "build/Release/shaders",
        "../build/shaders", "../build/Debug/shaders"
    });

    // Resolve sky.hdr
    bool hdrFound = false;
    {
        auto path = resolveFirst({
            cfg.ibl.hdrPath, "assets/sky.hdr", "../assets/sky.hdr",
            "../../assets/sky.hdr"
        });
        if (std::filesystem::exists(path)) {
            cfg.ibl.hdrPath = path;
            cfg.ibl.enabled = true;
            hdrFound = true;
            std::cout << "[example] HDR: " << path << "\n";
        } else {
            cfg.ibl.enabled = false;
            std::cout << "[example] sky.hdr not found — IBL disabled\n"
                      << "          Download from https://polyhaven.com/hdris\n\n";
        }
    }
    printHelp(hdrFound);

    // ── Window + Renderer ─────────────────────────────────────────────────────
    Window   window("vkgfx — deferred PBR + IBL", 1920, 1080);
    Renderer renderer(window, cfg);
    Context& ctx = renderer.context();
    window.setFullscreen(false);

    // ── Camera ────────────────────────────────────────────────────────────────
    Camera cam;
    cam.setPosition({0.f, 2.5f, -9.f}).setFov(60.f);

    // ── Scene ─────────────────────────────────────────────────────────────────
    Scene scene;
    scene.setCamera(&cam);

    std::vector<std::shared_ptr<Mesh>> meshes;

    auto addSphere = [&](glm::vec3 pos, float r, float g, float b,
                         float roughness, float metallic) {
        auto s = Mesh::createSphere(0.42f, 32, 32, ctx);
        s->setPosition(pos);
        auto m = std::make_shared<PBRMaterial>();
        m->setAlbedo(r, g, b).setRoughness(roughness).setMetallic(metallic);
        s->setMaterial(m);
        scene.add(s);
        meshes.push_back(s);
    };

    // 5×5 PBR grid: col = roughness (0→1 left→right), row = metallic (0→1 bottom→top)
    const int   GRID = 5;
    const float STEP = 1.1f;
    const float OX   = -(GRID - 1) * STEP * 0.5f;
    const float OY   =  1.0f;
    for (int row = 0; row < GRID; ++row) {
        float met = static_cast<float>(row) / (GRID - 1);
        for (int col = 0; col < GRID; ++col) {
            float rou = 0.05f + static_cast<float>(col) / (GRID - 1) * 0.95f;
            // Color gradient: copper for metallic, warm white for dielectric
            float colR = glm::mix(0.95f, 0.72f, met);
            float colG = glm::mix(0.85f, 0.45f, met);
            float colB = glm::mix(0.80f, 0.20f, met);
            addSphere({OX + col * STEP, OY + row * STEP, 0.f},
                       colR, colG, colB, rou, met);
        }
    }

    // Large showcase sphere dead centre, gold metallic
    addSphere({0.f, OY + (GRID - 1) * STEP * 0.5f, -2.8f},
               1.0f, 0.78f, 0.34f, 0.1f, 1.0f);

    // Ground — a row of flat spheres as a floor plane
    for (int i = -6; i <= 6; ++i) {
        for (int j = -6; j <= 6; ++j) {
            auto s = Mesh::createSphere(0.5f, 16, 16, ctx);
            s->setPosition({i * 1.0f, -0.35f, j * 1.0f});
            s->setScale({1.f, 0.12f, 1.f}); // squash to make a flat tile
            auto m = std::make_shared<PBRMaterial>();
            float checker = ((i + j) % 2 == 0) ? 0.8f : 0.3f;
            m->setAlbedo(checker, checker, checker).setRoughness(0.9f).setMetallic(0.0f);
            s->setMaterial(m);
            scene.add(s);
            meshes.push_back(s);
        }
    }

    // ── Collision world ───────────────────────────────────────────────────────
    CollisionWorld physics(4.f);

    // Register all spheres with sphere colliders
    for (auto& m : meshes) {
        auto& obj  = physics.add(m.get(), Collider::fitSphere(*m), false);
        obj.tag    = "sphere";
    }

    // Collision callback — print on contact (throttled)
    int contactCount = 0;
    physics.setOnContact([&](const CollisionEvent& ev) {
        ++contactCount; // counted per frame, not printed every frame (too spammy)
    });

    // ── Sun ───────────────────────────────────────────────────────────────────
    auto sun = std::make_shared<DirectionalLight>();
    sun->setDirection(cfg.sun.direction[0], cfg.sun.direction[1], cfg.sun.direction[2])
        .setIntensity(cfg.sun.intensity)
        .setEnabled(cfg.sun.enabled);
    scene.add(sun);

    // ── 4 colored point lights ────────────────────────────────────────────────
    const glm::vec3 ptColors[4] = {
        {1.0f, 0.2f, 0.1f},  // red
        {0.1f, 0.4f, 1.0f},  // blue
        {0.2f, 1.0f, 0.3f},  // green
        {1.0f, 0.8f, 0.1f},  // yellow
    };
    std::vector<std::shared_ptr<PointLight>> ptLights;
    for (int i = 0; i < 4; ++i) {
        auto l = std::make_shared<PointLight>();
        l->setColor(ptColors[i].r, ptColors[i].g, ptColors[i].b)
         .setIntensity(60.f)
         .setRadius(8.f);
        scene.add(l);
        ptLights.push_back(l);
    }

    // ── Main loop ─────────────────────────────────────────────────────────────
    const float MOVE  = 5.0f;
    const float LOOK  = 0.10f;
    auto lastTime     = std::chrono::high_resolution_clock::now();
    float totalTime   = 0.f;

    // FPS counter state
    float  fpsTimer   = 0.f;
    int    fpsFrames  = 0;
    float  fpsDisplay = 0.f;

    while (!window.shouldClose()) {
        auto now = std::chrono::high_resolution_clock::now();
        float dt = std::chrono::duration<float>(now - lastTime).count();
        lastTime = now;
        totalTime += dt;

        // FPS counter — update title every 0.5 s
        ++fpsFrames;
        fpsTimer += dt;
        if (fpsTimer >= 0.5f) {
            fpsDisplay = fpsFrames / fpsTimer;
            fpsTimer   = 0.f;
            fpsFrames  = 0;
            std::ostringstream oss;
            oss << "vkgfx — deferred PBR + IBL  |  "
                << std::fixed << std::setprecision(1) << fpsDisplay << " fps  |  "
                << std::setprecision(2) << (1000.f / fpsDisplay) << " ms";
            window.setTitle(oss.str());
        }

        window.pollEvents();

        // Camera movement
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

        // Fullscreen toggle (F11)
        if (window.keyPressed(GLFW_KEY_F11))
            window.toggleFullscreen();

        // Cursor lock toggle (Tab — useful for GUI interaction)
        if (window.keyPressed(GLFW_KEY_TAB))
            window.toggleCursorLock();

        // IBL toggle
        if (window.keyPressed(GLFW_KEY_F1) && hdrFound) {
            cfg.ibl.enabled = !cfg.ibl.enabled;
            std::cout << "IBL: " << (cfg.ibl.enabled ? "ON" : "OFF") << "\n";
            renderer.applyConfig(cfg);
        }
        // Sun toggle
        if (window.keyPressed(GLFW_KEY_F2)) {
            cfg.sun.enabled = !cfg.sun.enabled;
            sun->setEnabled(cfg.sun.enabled);
            std::cout << "Sun: " << (cfg.sun.enabled ? "ON" : "OFF") << "\n";
        }
        // G-buffer debug views
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

        // Orbit point lights around the grid
        float orbitR = 4.5f;
        for (int i = 0; i < 4; ++i) {
            float angle = totalTime * 0.6f + i * (glm::pi<float>() * 0.5f);
            float height = 2.0f + std::sin(totalTime * 0.4f + i) * 1.5f;
            ptLights[i]->setPosition({
                std::cos(angle) * orbitR,
                OY + height,
                std::sin(angle) * orbitR
            });
        }

        // ── Collision update (separates overlapping objects) ─────────────────
        contactCount = 0;
        physics.update(true);  // applyResponse=true pushes objects apart

        // ── Left-click to ray cast from camera ────────────────────────────────
        // Tab unlocks cursor; while unlocked, click fires a ray
        if (!window.isCursorLocked() &&
            glfwGetMouseButton(window.handle(), GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
            // Cast ray along camera's forward direction
            glm::mat4 view  = cam.view();
            glm::vec3 fwd   = glm::normalize(-glm::vec3(view[0][2], view[1][2], view[2][2]));
            auto hit = physics.castRay(cam.position(), fwd, 100.f);
            if (hit) {
                std::cout << "[ray] hit mesh at distance "
                          << hit.t << "  point " << hit.point.x << ","
                          << hit.point.y << "," << hit.point.z << "\n";
            }
        }

        renderer.render(scene);
    }

    // Cleanup
    vkDeviceWaitIdle(ctx.device());
    for (auto& m : meshes) m->destroy(ctx);
    meshes.clear();

    renderer.shutdown();
    std::cout << "[example] clean exit\n";
    return 0;
}
