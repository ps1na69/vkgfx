# VKGFX — Usage Examples

Every example is a single `main.cpp`.  
Compile with `#include <vkgfx/vkgfx.h>` and link against the library.
---

## Example 1 — Minimal "Hello Cube"

The smallest possible program: one spinning cube, one directional light.

```cpp
#include <vkgfx/vkgfx.h>
using namespace vkgfx;

int main() {
    // ── Window ──────────────────────────────────────────────────────────────
    Window window("Hello Cube", 1280, 720);

    // ── Renderer ────────────────────────────────────────────────────────────
    RendererSettings rs;
    rs.msaa       = MSAASamples::x4;
    rs.vsync      = true;
    rs.clearColor = {0.05f, 0.08f, 0.15f, 1.f}; // dark navy
    rs.shaderDir  = "shaders";
    Renderer renderer(window, rs);

    // ── Camera ──────────────────────────────────────────────────────────────
    // Position camera behind the cube on the -Z axis.
    // yaw=90 makes the camera face +Z (toward the cube at origin).
    Camera camera;
    camera.setPosition({0.f, 1.5f, -5.f})
          .setYaw(90.f)
          .setPitch(-15.f)
          .setFov(70.f)
          .setAspect(window.getAspectRatio());

    // ── Scene ───────────────────────────────────────────────────────────────
    Scene scene(&camera);
    scene.setAmbient({0.5f, 0.6f, 1.f}, 0.05f);

    auto sun = std::make_shared<DirectionalLight>();
    sun->setDirection({-1.f, -2.f, 1.f})
        .setColor({1.f, 0.95f, 0.85f})
        .setIntensity(4.f);
    scene.add(sun);

    auto cube = Mesh::createCube(1.f);
    auto mat  = std::make_shared<PBRMaterial>();

    // ── Cat texture ─────────────────────────────────────────────────────────
    // Load the cat image as the albedo (diffuse colour) of the cube.
    // Texture::fromFile() uploads to GPU and generates mipmaps automatically.
    // VKGFX_ASSET_DIR is injected by CMake (absolute path to source assets/).
    // Add #ifndef VKGFX_ASSET_DIR / #define VKGFX_ASSET_DIR "assets" / #endif
    // at the top of main.cpp as a fallback for non-CMake builds.
    auto catTex = Texture::fromFile(renderer.context(), VKGFX_ASSET_DIR "/cat.png");
    mat->setTexture(PBRMaterial::ALBEDO, catTex);
    mat->setAlbedo({1.f, 1.f, 1.f, 1.f}) // white tint — texture colour is shown as-is
        .setRoughness(0.6f)
        .setMetallic(0.0f);
    cube->setMaterial(mat);
    scene.add(cube);

    // ── Post-processing ──────────────────────────────────────────────────────
    // Must be set up AFTER the Renderer is constructed and BEFORE the loop.
    // Calling setPostProcess with enabled=true rebuilds scene pipelines for
    // the HDR offscreen pass and creates the tone-mapping/bloom pipeline.
    PostProcessSettings pp;
    pp.enabled        = true;

    pp.exposure       = 1.3f;   // slightly brighter overall
    pp.toneMapping    = true;
    pp.toneMapMode    = PostProcessSettings::ToneMapMode::ACES; // filmic

    pp.brightness     =  0.05f; // subtle lift
    pp.contrast       =  1.15f; // a touch more punch
    pp.saturation     =  1.20f; // vivid colours
    pp.colorBalance   = {1.00f, 0.96f, 0.90f}; // warm tint

    pp.bloom          = true;
    pp.bloomThreshold = 0.80f;  // only the brightest highlights bleed
    pp.bloomStrength  = 0.35f;
    pp.bloomRadius    = 1.5f;

    renderer.setPostProcess(pp);

    // ── Loop ────────────────────────────────────────────────────────────────
    float lastTime = window.getTime();

    while (!window.shouldClose()) {
        window.pollEvents();

        float now = window.getTime();
        float dt  = now - lastTime;
        lastTime  = now;

        cube->setRotation({now * 20.f, now * 30.f, 0.f});

        renderer.render(scene);
        window.setTitle("Hello Cube | FPS: "
            + std::to_string(int(renderer.stats().fps)));
    }

    renderer.shutdown(&scene);
}
```

---

## Example 2 — Solar System

Animated orbits, emissive sun, point light, background stars, black sky.

```cpp
#include <vkgfx/vkgfx.h>
#include <cmath>
#include <cstdlib>
using namespace vkgfx;

static std::shared_ptr<Mesh> makePlanet(float radius, Vec4 color,
                                         float roughness, float metallic = 0.f)
{
    auto m   = Mesh::createSphere(radius, 48, 24);
    auto mat = std::make_shared<PBRMaterial>();
    mat->setAlbedo(color).setRoughness(roughness).setMetallic(metallic);
    m->setMaterial(mat);
    return m;
}

int main() {
    Window window("Solar System", 1280, 720);

    RendererSettings rs;
    rs.msaa           = MSAASamples::x4;
    rs.frustumCulling = true;
    rs.vsync          = false;
    rs.clearColor     = {0.f, 0.f, 0.f, 1.f}; // black space
    rs.shaderDir      = "shaders";
    Renderer renderer(window, rs);

    // Camera sits above and behind the solar system, aimed at origin.
    // yaw=90 -> faces +Z. Planets orbit in the XZ plane.
    Camera camera;
    camera.setPosition({0.f, 12.f, -22.f})
          .setYaw(90.f)
          .setPitch(-29.f)
          .setFov(65.f)
          .setFar(1000.f)
          .setAspect(window.getAspectRatio());

    Scene scene(&camera);
    scene.setAmbient({0.f, 0.f, 0.05f}, 0.02f);

    // ── Sun ─────────────────────────────────────────────────────────────────
    auto sun = makePlanet(2.f, {1.f, 0.9f, 0.3f, 1.f}, 1.f, 0.f);
    {
        auto mat = std::dynamic_pointer_cast<PBRMaterial>(sun->getMaterial());
        mat->setEmissive({1.f, 0.8f, 0.2f, 1.f}, 6.f);
    }
    scene.add(sun);

    auto sunLight = std::make_shared<PointLight>();
    sunLight->setPosition({0.f, 0.f, 0.f})
             .setColor({1.f, 0.9f, 0.7f})
             .setIntensity(20.f);
    sunLight->setRange(200.f);
    scene.add(sunLight);

    // ── Planets ─────────────────────────────────────────────────────────────
    // Animation state lives inside the struct — no static maps, no UB.
    struct PlanetDef {
        float  orbitRadius;
        float  orbitSpeed;   // degrees/sec
        float  orbitPhase;   // starting angle (degrees)
        float  selfRotSpeed; // degrees/sec
        float  orbitAngle = 0.f;
        float  selfAngle  = 0.f;
        std::shared_ptr<Mesh> mesh;
    };

    std::vector<PlanetDef> planets = {
        { 4.5f, 47.f,   0.f, 180.f, 0.f, 0.f, makePlanet(0.25f, {0.6f,0.6f,0.6f,1.f}, 0.9f) },
        { 6.5f, 35.f,  40.f,  -5.f, 0.f, 0.f, makePlanet(0.55f, {0.9f,0.8f,0.5f,1.f}, 0.7f) },
        { 9.0f, 29.f,  80.f, 360.f, 0.f, 0.f, makePlanet(0.60f, {0.2f,0.5f,0.9f,1.f}, 0.6f) },
        {12.0f, 24.f, 130.f, 350.f, 0.f, 0.f, makePlanet(0.40f, {0.8f,0.3f,0.2f,1.f}, 0.8f) },
        {17.0f, 13.f, 200.f, 870.f, 0.f, 0.f, makePlanet(1.40f, {0.8f,0.65f,0.45f,1.f}, 0.5f) },
    };
    for (auto& p : planets) scene.add(p.mesh);

    // ── Stars ────────────────────────────────────────────────────────────────
    auto starMat = std::make_shared<PBRMaterial>();
    starMat->setAlbedo({1.f,1.f,1.f,1.f}).setEmissive({1.f,1.f,1.f,1.f}, 3.f);
    std::srand(42);
    for (int i = 0; i < 200; ++i) {
        auto star = Mesh::createSphere(0.05f, 4, 4);
        star->setMaterial(starMat);
        float theta = (std::rand() / float(RAND_MAX)) * 360.f;
        float phi   = (std::rand() / float(RAND_MAX)) * 180.f - 90.f;
        float r     = 150.f + (std::rand() / float(RAND_MAX)) * 50.f;
        float tp = glm::radians(phi), tt = glm::radians(theta);
        star->setPosition({ r * std::cos(tp) * std::cos(tt),
                            r * std::sin(tp),
                            r * std::cos(tp) * std::sin(tt) });
        scene.add(star);
    }

    // ── Loop ────────────────────────────────────────────────────────────────
    float lastTime = window.getTime();

    while (!window.shouldClose()) {
        window.pollEvents();

        float now = window.getTime();
        float dt  = now - lastTime;
        lastTime  = now;

        for (auto& p : planets) {
            p.orbitAngle += p.orbitSpeed * dt;
            float rad = glm::radians(p.orbitPhase + p.orbitAngle);
            p.mesh->setPosition({
                std::cos(rad) * p.orbitRadius,
                0.f,
                std::sin(rad) * p.orbitRadius
            });
            p.selfAngle += p.selfRotSpeed * dt;
            p.mesh->setRotation({0.f, p.selfAngle, 0.f});
        }

        renderer.render(scene);
        window.setTitle("Solar System | FPS: "
            + std::to_string(int(renderer.stats().fps)));
    }

    renderer.shutdown(&scene);
}
```

---

## Example 3 — First-Person Explorer (PBR vs Phong)

Interactive fly-camera, mixed materials, animated colour-cycling spotlight.

```cpp
#include <vkgfx/vkgfx.h>
#include <cmath>
using namespace vkgfx;

class FlyCamera {
public:
    explicit FlyCamera(Camera& cam) : m_cam(cam) {}

    void update(const Window& win, float dt) {
        float speed = (win.isKeyPressed(GLFW_KEY_LEFT_SHIFT) ? 20.f : 6.f) * dt;
        if (win.isKeyPressed(GLFW_KEY_W)) m_cam.translate( m_cam.forward() * speed);
        if (win.isKeyPressed(GLFW_KEY_S)) m_cam.translate(-m_cam.forward() * speed);
        if (win.isKeyPressed(GLFW_KEY_A)) m_cam.translate(-m_cam.right()   * speed);
        if (win.isKeyPressed(GLFW_KEY_D)) m_cam.translate( m_cam.right()   * speed);
        if (win.isKeyPressed(GLFW_KEY_E)) m_cam.translate({0.f,  speed, 0.f});
        if (win.isKeyPressed(GLFW_KEY_Q)) m_cam.translate({0.f, -speed, 0.f});

        Vec2 cursor = win.getCursorPos();
        if (m_first) { m_last = cursor; m_first = false; }
        Vec2 delta = (cursor - m_last) * m_sensitivity;
        m_last = cursor;
        m_cam.rotate(delta.x, -delta.y);
    }

private:
    Camera& m_cam;
    Vec2    m_last{};
    bool    m_first       = true;
    float   m_sensitivity = 0.12f;
};

int main() {
    Window window("PBR vs Phong — WASD/QE move, Shift=sprint, ESC=quit", 1280, 720);
    window.setCursorVisible(false);

    RendererSettings rs;
    rs.msaa           = MSAASamples::x4;
    rs.frustumCulling = true;
    rs.vsync          = false;
    rs.clearColor     = {0.02f, 0.02f, 0.04f, 1.f}; // near-black night sky
    rs.shaderDir      = "shaders";
    Renderer renderer(window, rs);

    Camera camera;
    camera.setPosition({0.f, 2.f, -10.f})
          .setYaw(90.f)
          .setPitch(-10.f)
          .setFov(75.f)
          .setNear(0.05f)
          .setFar(500.f)
          .setAspect(window.getAspectRatio());

    FlyCamera fly(camera);

    Scene scene(&camera);
    scene.setAmbient({0.1f, 0.12f, 0.15f}, 0.06f);

    // ── Directional key light ────────────────────────────────────────────────
    auto sun = std::make_shared<DirectionalLight>();
    sun->setDirection({-0.6f, -1.f, 0.4f})
        .setColor({1.f, 0.95f, 0.85f})
        .setIntensity(3.f);
    scene.add(sun);

    // ── Animated colour spotlight ────────────────────────────────────────────
    // CORRECT: base-class chain first, then SpotLight-specific chain.
    auto spot = std::make_shared<SpotLight>();
    spot->setPosition({0.f, 6.f, 0.f})
         .setColor({1.f, 1.f, 1.f})
         .setIntensity(12.f);
    spot->setDirection({0.f, -1.f, 0.f})
         .setInnerCone(15.f)
         .setOuterCone(35.f)
         .setRange(30.f);
    scene.add(spot);

    // ── Ground ───────────────────────────────────────────────────────────────
    auto ground = Mesh::createPlane(40.f, 8);
    auto groundMat = std::make_shared<PBRMaterial>();
    groundMat->setAlbedo({0.18f, 0.18f, 0.18f, 1.f}).setRoughness(0.85f);
    ground->setMaterial(groundMat);
    scene.add(ground);

    // ── PBR spheres (metallic gradient) ─────────────────────────────────────
    const int N = 7;
    for (int i = 0; i < N; ++i) {
        auto s   = Mesh::createSphere(0.5f, 32, 16);
        auto mat = std::make_shared<PBRMaterial>();
        mat->setAlbedo({0.9f, 0.1f, 0.05f, 1.f})
            .setRoughness(0.15f + 0.12f * i)
            .setMetallic(float(i) / (N - 1));
        s->setMaterial(mat);
        s->setPosition({float(i - N/2) * 1.5f, 0.55f, 2.f});
        scene.add(s);
    }

    // ── Phong spheres (colour palette) ───────────────────────────────────────
    std::vector<Vec4> phongColors = {
        {1.f,0.2f,0.2f,1.f}, {0.2f,1.f,0.2f,1.f}, {0.2f,0.4f,1.f,1.f},
        {1.f,0.9f,0.1f,1.f}, {0.8f,0.2f,1.f,1.f}, {0.1f,0.9f,0.9f,1.f},
        {1.f,0.5f,0.1f,1.f},
    };
    for (int i = 0; i < N; ++i) {
        auto s   = Mesh::createSphere(0.5f, 32, 16);
        auto mat = std::make_shared<PhongMaterial>();
        mat->setDiffuse(phongColors[i])
            .setSpecular({1.f, 1.f, 1.f, 1.f})
            .setShininess(8.f + 24.f * i);
        s->setMaterial(mat);
        s->setPosition({float(i - N/2) * 1.5f, 0.55f, -2.f});
        scene.add(s);
    }

    // ── Loop ────────────────────────────────────────────────────────────────
    float lastTime = window.getTime();

    while (!window.shouldClose()) {
        window.pollEvents();
        if (window.isKeyPressed(GLFW_KEY_ESCAPE)) window.close();

        float now = window.getTime();
        float dt  = now - lastTime;
        lastTime  = now;

        fly.update(window, dt);

        // Cycle spotlight colour through the rainbow
        float hue = std::fmod(now * 0.4f, 1.f);
        float h6  = hue * 6.f;
        float f   = h6 - std::floor(h6);
        Vec3 rgb;
        switch (int(h6) % 6) {
            case 0: rgb = {1.f,   f,    0.f  }; break;
            case 1: rgb = {1.f-f, 1.f,  0.f  }; break;
            case 2: rgb = {0.f,   1.f,  f    }; break;
            case 3: rgb = {0.f,   1.f-f,1.f  }; break;
            case 4: rgb = {f,     0.f,  1.f  }; break;
            default:rgb = {1.f,   0.f,  1.f-f}; break;
        }
        spot->setColor(rgb);

        float sx = std::cos(now * 0.3f) * 6.f;
        float sz = std::sin(now * 0.3f) * 6.f;
        spot->setPosition({sx, 6.f, sz});
        spot->setDirection(glm::normalize(Vec3{-sx, -6.f, -sz}));

        renderer.render(scene);
        window.setTitle("PBR vs Phong | FPS: "
            + std::to_string(int(renderer.stats().fps))
            + " | Culled: " + std::to_string(renderer.stats().culledObjects));
    }

    renderer.shutdown(&scene);
}
```

---

## Example 4 — Procedural Terrain

Custom vertex/index buffers, height-map normals, sky-blue background, F1 wireframe.

```cpp
#include <vkgfx/vkgfx.h>
#include <cmath>
#include <cstdlib>
using namespace vkgfx;

static std::shared_ptr<Mesh> createHeightmap(int nx, int ny,
                                              float size, float waveAmp)
{
    std::vector<Vertex>   verts;
    std::vector<uint32_t> indices;
    verts.reserve((nx + 1) * (ny + 1));
    indices.reserve(nx * ny * 6);

    float dx = size / nx;
    float dz = size / ny;

    for (int row = 0; row <= ny; ++row) {
        for (int col = 0; col <= nx; ++col) {
            float x = col * dx - size * 0.5f;
            float z = row * dz - size * 0.5f;
            float y = waveAmp * (std::sin(x * 0.8f) * std::cos(z * 0.8f));
            Vertex v;
            v.position = {x, y, z};
            v.uv       = {float(col) / nx, float(row) / ny};
            v.normal   = {0.f, 1.f, 0.f};
            verts.push_back(v);
        }
    }

    for (int row = 0; row < ny; ++row) {
        for (int col = 0; col < nx; ++col) {
            uint32_t tl = row       * (nx + 1) + col;
            uint32_t tr = row       * (nx + 1) + col + 1;
            uint32_t bl = (row + 1) * (nx + 1) + col;
            uint32_t br = (row + 1) * (nx + 1) + col + 1;
            indices.insert(indices.end(), {tl, bl, tr, tr, bl, br});
        }
    }

    // Smooth normals via finite differences
    auto sampleY = [&](int r, int c) {
        r = std::clamp(r, 0, ny);
        c = std::clamp(c, 0, nx);
        return verts[r * (nx + 1) + c].position.y;
    };
    for (int row = 0; row <= ny; ++row) {
        for (int col = 0; col <= nx; ++col) {
            float hL = sampleY(row, col-1), hR = sampleY(row, col+1);
            float hD = sampleY(row-1, col), hU = sampleY(row+1, col);
            verts[row * (nx + 1) + col].normal =
                glm::normalize(Vec3{hL - hR, 2.f * dx, hD - hU});
        }
    }

    return std::make_shared<Mesh>(std::move(verts), std::move(indices));
}

int main() {
    Window window("Procedural Terrain — F1: wireframe, ESC: quit", 1280, 720);
    window.setCursorVisible(false);

    RendererSettings rs;
    rs.msaa       = MSAASamples::x4;
    rs.vsync      = false;
    rs.clearColor = {0.53f, 0.75f, 0.92f, 1.f}; // sky blue
    rs.shaderDir  = "shaders";
    Renderer renderer(window, rs);

    Camera camera;
    camera.setPosition({0.f, 5.f, -15.f})
          .setYaw(90.f)
          .setPitch(-18.f)
          .setFov(70.f)
          .setFar(500.f)
          .setAspect(window.getAspectRatio());

    Scene scene(&camera);
    scene.setAmbient({0.6f, 0.75f, 0.9f}, 0.2f); // blue-tinted sky ambient

    auto sun = std::make_shared<DirectionalLight>();
    sun->setDirection({-0.5f, -1.f, 0.3f})
        .setColor({1.f, 0.95f, 0.8f})
        .setIntensity(4.f);
    scene.add(sun);

    // Terrain
    auto terrain = createHeightmap(80, 80, 30.f, 1.2f);
    auto terrainMat = std::make_shared<PBRMaterial>();
    terrainMat->setAlbedo({0.35f, 0.58f, 0.28f, 1.f})
               .setRoughness(0.9f)
               .setMetallic(0.f);
    terrain->setMaterial(terrainMat);
    scene.add(terrain);

    // Rocks
    auto rockMat = std::make_shared<PBRMaterial>();
    rockMat->setAlbedo({0.35f, 0.32f, 0.30f, 1.f})
            .setRoughness(0.8f)
            .setMetallic(0.05f);
    std::srand(1337);
    for (int i = 0; i < 30; ++i) {
        auto rock = Mesh::createSphere(
            0.2f + (std::rand() / float(RAND_MAX)) * 0.5f, 8, 6);
        rock->setMaterial(rockMat);
        float rx = (std::rand() / float(RAND_MAX) - 0.5f) * 24.f;
        float rz = (std::rand() / float(RAND_MAX) - 0.5f) * 24.f;
        float ry = 1.2f * std::sin(rx * 0.8f) * std::cos(rz * 0.8f);
        rock->setPosition({rx, ry, rz});
        rock->setRotation({
            float(std::rand() % 360),
            float(std::rand() % 360),
            float(std::rand() % 360)
        });
        scene.add(rock);
    }

    bool wireframe = false;
    bool f1WasDown = false;
    float lastTime = window.getTime();

    while (!window.shouldClose()) {
        window.pollEvents();
        if (window.isKeyPressed(GLFW_KEY_ESCAPE)) window.close();

        bool f1 = window.isKeyPressed(GLFW_KEY_F1);
        if (f1 && !f1WasDown) {
            wireframe = !wireframe;
            renderer.setWireframe(wireframe);
        }
        f1WasDown = f1;

        float now = window.getTime();
        float dt  = now - lastTime;
        lastTime  = now;

        // Orbit camera automatically
        float angle = now * 0.2f;
        camera.setPosition({std::sin(angle) * 15.f, 5.f, std::cos(angle) * 15.f - 5.f});
        camera.setYaw(glm::degrees(angle) + 90.f);
        camera.setPitch(-18.f);

        renderer.render(scene);
        window.setTitle(std::string("Terrain | FPS: ")
            + std::to_string(int(renderer.stats().fps))
            + (wireframe ? "  [WIREFRAME — F1 to toggle]" : "  [F1 = wireframe]"));
    }

    renderer.shutdown(&scene);
}
```

---

## Example 5 — Runtime Scene Management

Dynamically spawn and remove meshes each frame, toggle visibility with Space.

```cpp
#include <vkgfx/vkgfx.h>
#include <vector>
using namespace vkgfx;

int main() {
    Window window("Runtime Scene Management — Space: toggle visibility", 1280, 720);

    RendererSettings rs;
    rs.msaa       = MSAASamples::x4;
    rs.vsync      = false;
    rs.clearColor = {0.04f, 0.04f, 0.06f, 1.f};
    rs.shaderDir  = "shaders";
    Renderer renderer(window, rs);

    Camera camera;
    camera.setPosition({0.f, 5.f, -12.f})
          .setYaw(90.f)
          .setPitch(-22.f)
          .setFov(70.f)
          .setAspect(window.getAspectRatio());

    Scene scene(&camera);
    scene.setAmbient({0.1f, 0.1f, 0.15f}, 0.08f);

    auto sun = std::make_shared<DirectionalLight>();
    sun->setDirection({-1.f, -1.5f, 0.5f})
        .setColor({1.f, 0.95f, 0.85f})
        .setIntensity(4.f);
    scene.add(sun);

    // Ground
    auto ground = Mesh::createPlane(20.f, 4);
    auto gMat   = std::make_shared<PBRMaterial>();
    gMat->setAlbedo({0.15f, 0.15f, 0.15f, 1.f}).setRoughness(0.8f);
    ground->setMaterial(gMat);
    scene.add(ground);

    // Shared material palette
    std::vector<std::shared_ptr<PBRMaterial>> palette;
    for (Vec4 c : std::initializer_list<Vec4>{
            {1.f,0.2f,0.2f,1.f}, {0.2f,0.9f,0.3f,1.f}, {0.2f,0.4f,1.f,1.f},
            {1.f,0.8f,0.1f,1.f}, {0.8f,0.2f,1.f,1.f},  {0.1f,0.9f,0.9f,1.f}}) {
        auto m = std::make_shared<PBRMaterial>();
        m->setAlbedo(c).setRoughness(0.3f).setMetallic(0.5f);
        palette.push_back(m);
    }

    // Pool of live meshes — oldest removed when full
    std::vector<std::shared_ptr<Mesh>> pool;
    const int MAX_ALIVE = 20;

    std::srand(42);
    float spawnTimer   = 0.f;
    float spawnRate    = 0.4f;
    bool  hideOdd      = false;
    bool  spaceWasDown = false;
    float lastTime     = window.getTime();

    while (!window.shouldClose()) {
        window.pollEvents();
        if (window.isKeyPressed(GLFW_KEY_ESCAPE)) window.close();

        float now = window.getTime();
        float dt  = now - lastTime;
        lastTime  = now;

        // Space — toggle odd-indexed meshes visible/hidden
        bool spaceDown = window.isKeyPressed(GLFW_KEY_SPACE);
        if (spaceDown && !spaceWasDown) {
            hideOdd = !hideOdd;
            for (int i = 0; i < int(pool.size()); ++i)
                pool[i]->setVisible(!(hideOdd && (i % 2 == 1)));
        }
        spaceWasDown = spaceDown;

        // Spawn
        spawnTimer += dt;
        if (spawnTimer >= spawnRate) {
            spawnTimer = 0.f;

            if (int(pool.size()) >= MAX_ALIVE) {
                scene.remove(pool.front());
                pool.erase(pool.begin()); // erase-from-front replaces deque::pop_front
            }

            std::shared_ptr<Mesh> m;
            switch (std::rand() % 3) {
                case 0: m = Mesh::createSphere(0.3f + (std::rand()%10)/20.f, 20, 10); break;
                case 1: m = Mesh::createCube(0.4f + (std::rand()%10)/15.f);           break;
                default:m = Mesh::createSphere(0.25f, 6, 4);                           break;
            }
            m->setMaterial(palette[std::rand() % int(palette.size())]);
            m->setPosition({
                (std::rand() % 100 / 100.f - 0.5f) * 14.f,
                0.5f + std::rand() % 30 / 10.f,
                (std::rand() % 100 / 100.f - 0.5f) * 10.f
            });
            m->setRotation({
                float(std::rand() % 360),
                float(std::rand() % 360),
                float(std::rand() % 360)
            });
            pool.push_back(m);
            scene.add(m);
        }

        // Rotate live meshes
        for (int i = 0; i < int(pool.size()); ++i) {
            Vec3 r = pool[i]->rotation();
            r.y += 45.f * dt * (1.f + i % 3);
            pool[i]->setRotation(r);
        }

        renderer.render(scene);
        window.setTitle("Scene Mgmt | Alive: " + std::to_string(pool.size())
            + " | SPACE=" + (hideOdd ? "half hidden" : "all visible")
            + " | FPS: " + std::to_string(int(renderer.stats().fps)));
    }

    renderer.shutdown(&scene);
}
```

---

---

## Post-Processing Reference

The `PostProcessSettings` struct exposes seven independent effects that can be
composed freely.  Call `renderer.setPostProcess(pp)` once before the loop
(or at any time during runtime — settings are hot-reloadable).

### Effects overview

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | `bool` | `false` | Master switch.  Disabling restores direct-to-swapchain rendering. |
| `exposure` | `float` | `1.0` | HDR exposure multiplier. `>1` = brighter, `<1` = darker. |
| `toneMapping` | `bool` | `true` | Compress HDR values into display range. |
| `toneMapMode` | `ToneMapMode` | `ACES` | `Linear` · `Reinhard` · `ACES` (filmic). |
| `brightness` | `float` | `0.0` | Additive offset after tone-mapping.  Range `−1 … +1`. |
| `contrast` | `float` | `1.0` | Contrast multiplier, pivoted at 0.5. `1` = neutral. |
| `saturation` | `float` | `1.0` | Colour saturation. `0` = greyscale, `2` = vivid. |
| `colorBalance` | `Vec3` | `{1,1,1}` | Per-channel RGB tint multipliers. |
| `bloom` | `bool` | `false` | Additive glow on bright highlights. |
| `bloomThreshold` | `float` | `0.8` | Luminance level above which bloom kicks in. |
| `bloomStrength` | `float` | `0.4` | Bloom blend intensity. |
| `bloomRadius` | `float` | `1.5` | Tap spread radius in texels. |

### Render architecture when PP is enabled

```
  ┌────────────────────────────────────────────────────────────────────────┐
  │  Scene render pass  →  HDR offscreen target (R16G16B16A16_SFLOAT)      │
  │  (existing PBR / Phong / unlit materials — unchanged)                  │
  └─────────────────────────────┬──────────────────────────────────────────┘
                                 │ sampled by
  ┌──────────────────────────────▼─────────────────────────────────────────┐
  │  Post-process pass  →  Swapchain image (B8G8R8A8_SRGB, sRGB gamma)     │
  │  postprocess.frag:  exposure → bloom → tone-map → brightness/contrast  │
  │                     → saturation → color balance → present             │
  └────────────────────────────────────────────────────────────────────────┘
```

Switching `enabled` on/off triggers a device-wait and pipeline rebuild
automatically — no manual teardown required.

### Compile the postprocess shaders

```sh
glslc shaders/postprocess.vert -o shaders/postprocess.vert.spv
glslc shaders/postprocess.frag -o shaders/postprocess.frag.spv
```

### Quick example

```cpp
#include <vkgfx/vkgfx.h>
using namespace vkgfx;

int main() {
    Window   window("PP Demo", 1280, 720);
    Renderer renderer(window);

    // ── Camera, scene, meshes … (as usual) ──────────────────────────────────

    // ── Post-process preset: "cinematic night" ───────────────────────────────
    PostProcessSettings pp;
    pp.enabled      = true;
    pp.exposure     = 0.9f;
    pp.toneMapping  = true;
    pp.toneMapMode  = PostProcessSettings::ToneMapMode::ACES;
    pp.brightness   = -0.05f;
    pp.contrast     = 1.25f;
    pp.saturation   = 0.80f;
    pp.colorBalance = {0.85f, 0.90f, 1.10f}; // cool blue tint
    pp.bloom        = true;
    pp.bloomThreshold = 0.75f;
    pp.bloomStrength  = 0.50f;
    renderer.setPostProcess(pp);

    // ── Loop ────────────────────────────────────────────────────────────────
    while (!window.shouldClose()) {
        window.pollEvents();
        // Tweak PP at runtime — changes take effect next frame:
        // pp.saturation = someSliderValue;
        // renderer.setPostProcess(pp);
        renderer.render(scene);
    }
    renderer.shutdown(&scene);
}
```

## Quick Reference

```cpp
// ── Window ────────────────────────────────────────────────────────────────────
Window window("Title", 1280, 720);
window.setCursorVisible(false);
window.isKeyPressed(GLFW_KEY_W);    // bool
window.getCursorPos();              // Vec2
window.getAspectRatio();            // float
window.setTitle("new title");
window.close();

// ── Renderer ──────────────────────────────────────────────────────────────────
RendererSettings rs;
rs.msaa           = MSAASamples::x4; // x1 / x2 / x4 / x8
rs.vsync          = false;
rs.frustumCulling = true;
rs.clearColor     = {r, g, b, 1.f}; // background / "sky" colour — SET THIS!
rs.shaderDir      = "shaders";
Renderer renderer(window, rs);
renderer.setWireframe(true);         // toggle at runtime
renderer.setClearColor({r,g,b,a});   // also changeable at runtime
renderer.render(scene);
renderer.stats();                    // .fps .drawCalls .culledObjects .frameTimeMs
renderer.shutdown(&scene);

// ── Camera ────────────────────────────────────────────────────────────────────
// yaw=90 -> camera faces +Z. Put the scene in front at positive Z.
Camera camera;
camera.setPosition({0.f, 2.f, -5.f})
      .setYaw(90.f)      // 90 = look toward +Z
      .setPitch(-15.f)   // negative = look slightly down
      .setFov(70.f)
      .setAspect(window.getAspectRatio())
      .setNear(0.05f).setFar(1000.f);
camera.translate(delta);
camera.rotate(dYaw, dPitch);

// ── Light chaining rules ─────────────────────────────────────────────────────
// Base Light methods (setPosition/setColor/setIntensity) return Light&.
// Subclass methods return their own type. Never mix them in one chain.

// CORRECT:
auto spot = std::make_shared<SpotLight>();
spot->setPosition({x,y,z}).setColor({r,g,b}).setIntensity(f); // Light& chain
spot->setDirection({x,y,z}).setInnerCone(d).setOuterCone(d).setRange(f); // SpotLight& chain

// WRONG — compile error (Light& has no setDirection):
// spot->setPosition({x,y,z}).setDirection({x,y,z});

// ── Materials ─────────────────────────────────────────────────────────────────
auto pbr = std::make_shared<PBRMaterial>();
pbr->setAlbedo({r,g,b,a}).setRoughness(0–1).setMetallic(0–1);
pbr->setEmissive({r,g,b,a}, scale);
// Texture slots: ALBEDO=0, NORMAL=1, METALROUGH=2, AO=3, EMISSIVE=4

auto phong = std::make_shared<PhongMaterial>();
phong->setDiffuse({r,g,b,a}).setSpecular({r,g,b,a}).setShininess(32.f);
// Texture slots: DIFFUSE=0, NORMAL=1

mat->pipelineSettings.cullMode   = CullMode::None;
mat->pipelineSettings.alphaBlend = true;
mat->pipelineSettings.depthWrite = false;

// ── Mesh ──────────────────────────────────────────────────────────────────────
auto m = Mesh::createSphere(radius, sectors=32, stacks=16);
auto m = Mesh::createCube(size);
auto m = Mesh::createPlane(size, subdivisions=1);
auto m = Mesh::fromFile("model.obj");
auto m = std::make_shared<Mesh>(std::move(vertices), std::move(indices));

m->setPosition({x,y,z});
m->setRotation({pitchDeg, yawDeg, rollDeg});
m->setScale({x,y,z});
m->setMaterial(mat);
m->setVisible(false);

// ── Lights ────────────────────────────────────────────────────────────────────
auto d = std::make_shared<DirectionalLight>();
d->setDirection({x,y,z}).setColor({r,g,b}).setIntensity(f);

auto p = std::make_shared<PointLight>();
p->setPosition({x,y,z}).setColor({r,g,b}).setIntensity(f);
p->setRange(f); // PointLight-only — separate statement

auto s = std::make_shared<SpotLight>();
s->setPosition({x,y,z}).setColor({r,g,b}).setIntensity(f);
s->setDirection({x,y,z}).setInnerCone(deg).setOuterCone(deg).setRange(f);

// ── Scene ─────────────────────────────────────────────────────────────────────
Scene scene(&camera);
scene.setAmbient({r,g,b}, intensity);
scene.add(mesh);    scene.remove(mesh);
scene.add(light);   scene.remove(light);
scene.clear();
```
