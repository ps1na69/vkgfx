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