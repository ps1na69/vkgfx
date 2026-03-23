// examples/06_textures/main.cpp
// Demonstrates: auto-scanning assets/textures/ for albedo/normal/rma files.
//
// USAGE:
//   1. Put your textures in:  assets/textures/
//   2. Name them with keywords in the filename:
//        albedo  (or diffuse, basecolor, color)
//        normal  (or nrm, norm)
//        rma     (or roughness, metallic, orm, arm)
//      Examples: rock_albedo.png, rock_normal.jpg, rock_rma.png
//   3. Run the example — textures are found and applied automatically.
//
// If no textures are found, fallback solid colours are used (no crash).

#include <vkgfx/vkgfx.h>
#include <filesystem>
#include <iostream>
#include <algorithm>
#include <chrono>

namespace fs = std::filesystem;

// ── Texture auto-discovery ────────────────────────────────────────────────────
// Scans a directory for image files whose names contain a keyword.
// Returns the first match, or "" if nothing found.
static std::string findTexture(const fs::path& dir,
                                std::initializer_list<const char*> keywords) {
    if (!fs::exists(dir)) return "";

    static const std::vector<std::string> IMAGE_EXT =
        {".png", ".jpg", ".jpeg", ".tga", ".bmp", ".tif", ".tiff"};

    for (auto& entry : fs::directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        std::string ext  = entry.path().extension().string();
        std::string name = entry.path().stem().string();

        // Convert both to lowercase for case-insensitive matching
        std::transform(ext.begin(),  ext.end(),  ext.begin(),  ::tolower);
        std::transform(name.begin(), name.end(), name.begin(), ::tolower);

        // Check extension is a known image format
        bool isImage = std::find(IMAGE_EXT.begin(), IMAGE_EXT.end(), ext)
                       != IMAGE_EXT.end();
        if (!isImage) continue;

        // Check name contains one of the keywords
        for (auto* kw : keywords) {
            if (name.find(kw) != std::string::npos) {
                std::cout << "  [textures] found " << kw << ": "
                          << entry.path().filename() << "\n";
                return entry.path().string();
            }
        }
    }
    return "";
}

int main() {
    using namespace vkgfx;

    Window window("06 – Textures", 1280, 720);
    window.setCursorLocked(true);

    RendererConfig cfg;
    cfg.ibl.enabled   = false;
    cfg.sun.enabled   = true;
    cfg.sun.intensity = 4.f;
    cfg.sun.direction[0] = -0.4f;
    cfg.sun.direction[1] = -1.0f;
    cfg.sun.direction[2] = -0.3f;


    Renderer   renderer(window, cfg);
    Context&   ctx = renderer.context();
    TextureCache textures(ctx);

    // ── Auto-scan assets/textures/ ────────────────────────────────────────────
    // Search several candidate locations (CWD, relative paths from build dir)
    fs::path texDir;
    // Search from the exe location upward — exe is inside build/examples/06.../Debug/
    // so project root (where assets/ lives) is 4 levels up.
    for (auto& candidate : {
            "assets/textures",        // next to exe (POST_BUILD copied here)
            "../assets/textures",
            "../../assets/textures",
            "../../../assets/textures",
            "../../../../assets/textures",  // build/examples/06.../Debug/ → project root
            "../../../../../assets/textures",
            "assets",                 // flat layout fallback
            "../../assets",
            "../../../../assets",
        }) {
        if (fs::exists(candidate)) { texDir = candidate; break; }
    }

    if (texDir.empty()) {
        std::cout << "[06_textures] No assets/textures/ folder found.\n"
                  << "  Create it and place your textures there:\n"
                  << "    assets/textures/albedo.png\n"
                  << "    assets/textures/normal.png\n"
                  << "    assets/textures/rma.png\n\n";
    } else {
        std::cout << "[06_textures] Scanning: " << fs::absolute(texDir) << "\n";
    }

    // Find each texture slot by keyword in the filename
    std::string albedoPath = findTexture(texDir,
        {"albedo", "diffuse", "basecolor", "base_color", "color", "col"});
    std::string normalPath = findTexture(texDir,
        {"normal", "nrm", "norm", "nmap"});
    std::string rmaPath    = findTexture(texDir,
        {"rma", "orm", "arm", "roughness", "metallic", "pbr"});

    // Load found textures — missing ones become 1×1 fallbacks automatically
    auto albedoTex = albedoPath.empty() ? textures.solid(180, 120, 80)
                                        : textures.load(albedoPath);
    auto normalTex = normalPath.empty() ? textures.solid(128, 128, 255)   // flat normal
                                        : textures.load(normalPath);
    auto rmaTex    = rmaPath.empty()    ? textures.solid(128, 0, 255)     // rough=0.5, met=0
                                        : textures.load(rmaPath);

    bool hasAlbedo = !albedoPath.empty();
    bool hasNormal = !normalPath.empty();
    bool hasRMA    = !rmaPath.empty();

    std::cout << "\nLoaded: albedo=" << (hasAlbedo ? "YES" : "fallback")
              << "  normal=" << (hasNormal ? "YES" : "fallback")
              << "  rma=" << (hasRMA ? "YES" : "fallback") << "\n\n";

    // ── Scene ─────────────────────────────────────────────────────────────────
    Scene  scene;
    Camera cam;
    cam.setPosition({0.f, 0.f, -5.f}).setFov(60.f);
    scene.setCamera(&cam);

    std::vector<std::shared_ptr<Mesh>> meshes;

    // Left sphere: full texture set (albedo + normal + rma)
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

    // Centre sphere: albedo only, manual PBR values
    {
        auto s = Mesh::createSphere(0.8f, 32, 32, ctx);
        s->setPosition({0.f, 0.f, 0.f});
        auto m = std::make_shared<PBRMaterial>();
        m->setAlbedoTexture(albedoTex)
         .setRoughness(0.2f)
         .setMetallic(0.8f);
        s->setMaterial(m);
        scene.add(s); meshes.push_back(s);
    }

    // Right sphere: no textures, solid colour with emissive glow
    {
        auto s = Mesh::createSphere(0.8f, 32, 32, ctx);
        s->setPosition({2.f, 0.f, 0.f});
        auto m = std::make_shared<PBRMaterial>();
        m->setAlbedo(1.f, 0.4f, 0.1f)
         .setRoughness(0.5f)
         .setMetallic(0.f)
         .setEmissive(1.f, 0.3f, 0.f, 1.5f);
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
    textures.clear();
    renderer.shutdown();
    return 0;
}
