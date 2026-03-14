# vkgfx Examples

All examples assume the working directory contains a `shaders/` folder with
compiled `.spv` files (built automatically by CMake via `vkgfx_shaders`).

---

## Minimal — built-in cube, no assets needed

```cpp
#include <vkgfx/vkgfx.h>

int main() {
    vkgfx::WindowSettings ws;
    ws.title = "vkgfx"; ws.width = 1280; ws.height = 720;
    vkgfx::Window window(ws);

    vkgfx::RendererSettings rs;
    rs.shaderDir = "shaders";          // path to compiled .spv directory
    rs.vsync     = true;
    vkgfx::Renderer renderer(window, rs);

    vkgfx::Camera camera;
    camera.setPosition({0.f, 1.5f, -4.f})
          .setYaw(-90.f)
          .setAspect(1280.f / 720.f);

    vkgfx::Scene scene(&camera);

    // Directional light (required for shadows)
    auto sun = std::make_shared<vkgfx::DirectionalLight>();
    sun->setDirection({-0.5f, -1.f, -0.5f})
        .setColor({1.f, 0.95f, 0.85f})
        .setIntensity(3.f);
    scene.add(sun);
    scene.setAmbient({0.1f, 0.1f, 0.15f}, 0.05f);

    // A cube + a ground plane
    auto cube = vkgfx::Mesh::createCube(1.f);
    auto mat  = std::make_shared<vkgfx::PBRMaterial>();
    mat->setAlbedo({0.8f, 0.6f, 0.4f, 1.f}).setMetallic(0.f).setRoughness(0.6f);
    cube->setMaterial(mat);
    scene.add(cube);

    auto plane = vkgfx::Mesh::createPlane(8.f, 4);
    auto pmat  = std::make_shared<vkgfx::PBRMaterial>();
    pmat->setAlbedo({0.5f, 0.5f, 0.5f, 1.f}).setRoughness(0.9f);
    plane->setMaterial(pmat);
    plane->setPosition({0.f, -0.5f, 0.f});
    scene.add(plane);

    while (!window.shouldClose()) {
        glfwPollEvents();
        renderer.render(scene);
    }
    renderer.shutdown(&scene);
}
```

---

## WASD + mouse-look camera

```cpp
double lastX = 0, lastY = 0;
bool   firstMouse = true, mouseDown = false;

glfwSetWindowUserPointer(window.handle(), &mouseDown);
glfwSetMouseButtonCallback(window.handle(), [](GLFWwindow* w, int btn, int action, int) {
    if (btn == GLFW_MOUSE_BUTTON_LEFT)
        *static_cast<bool*>(glfwGetWindowUserPointer(w)) = (action == GLFW_PRESS);
});

float lastTime = (float)glfwGetTime();
while (!window.shouldClose()) {
    glfwPollEvents();
    float dt = (float)glfwGetTime() - lastTime;
    lastTime += dt;

    float spd = 3.f * dt;
    if (glfwGetKey(window.handle(), GLFW_KEY_W) == GLFW_PRESS) camera.translate( camera.forward() * spd);
    if (glfwGetKey(window.handle(), GLFW_KEY_S) == GLFW_PRESS) camera.translate(-camera.forward() * spd);
    if (glfwGetKey(window.handle(), GLFW_KEY_D) == GLFW_PRESS) camera.translate( camera.right()   * spd);
    if (glfwGetKey(window.handle(), GLFW_KEY_A) == GLFW_PRESS) camera.translate(-camera.right()   * spd);
    if (glfwGetKey(window.handle(), GLFW_KEY_E) == GLFW_PRESS) camera.translate( camera.up()      * spd);
    if (glfwGetKey(window.handle(), GLFW_KEY_Q) == GLFW_PRESS) camera.translate(-camera.up()      * spd);
    if (glfwGetKey(window.handle(), GLFW_KEY_ESCAPE) == GLFW_PRESS) break;

    double cx, cy;
    glfwGetCursorPos(window.handle(), &cx, &cy);
    if (mouseDown) {
        if (firstMouse) { lastX = cx; lastY = cy; firstMouse = false; }
        camera.rotate((float)(cx - lastX) * 0.15f, -(float)(cy - lastY) * 0.15f);
    } else { firstMouse = true; }
    lastX = cx; lastY = cy;

    // Keep aspect ratio in sync with the window
    auto [w, h] = window.getFramebufferSize();
    if (w > 0 && h > 0)
        camera.setAspect((float)w / (float)h);

    renderer.render(scene);
}
```

---

## Loading a glTF model

```cpp
// Pass a .gltf or .glb path; meshes are uploaded to the GPU on load.
auto meshes = vkgfx::Mesh::fromGltf(renderer.contextPtr(), "assets/DamagedHelmet.glb");
for (auto& m : meshes) scene.add(m);
```

---

## Texture-mapped PBR material

```cpp
auto ctx = renderer.contextPtr();

// srgb=true  for albedo / emissive maps
// srgb=false for normal / metalRough / AO maps
auto albedo    = vkgfx::Texture::fromFile(ctx, "assets/albedo.png",      true);
auto normal    = vkgfx::Texture::fromFile(ctx, "assets/normal.png",      false);
auto metalRough = vkgfx::Texture::fromFile(ctx, "assets/metalRough.png", false);

auto mat = std::make_shared<vkgfx::PBRMaterial>();
mat->setTexture(vkgfx::PBRMaterial::ALBEDO,    albedo)
   .setTexture(vkgfx::PBRMaterial::NORMAL,     normal)
   .setTexture(vkgfx::PBRMaterial::METALROUGH, metalRough);
```

---

## Multiple lights

```cpp
// Warm sun
auto sun = std::make_shared<vkgfx::DirectionalLight>();
sun->setDirection({-0.5f, -1.f, -0.5f}).setColor({1.f, 0.95f, 0.8f}).setIntensity(3.f);
scene.add(sun);

// Cool fill point light
auto fill = std::make_shared<vkgfx::PointLight>();
fill->setPosition({-4.f, 3.f, 2.f}).setColor({0.4f, 0.6f, 1.f}).setIntensity(2.f);
static_cast<vkgfx::PointLight*>(fill.get())->setRange(20.f);
scene.add(fill);

// Cone spot light
auto spot = std::make_shared<vkgfx::SpotLight>();
spot->setPosition({0.f, 4.f, 0.f})
    .setDirection({0.f, -1.f, 0.f})
    .setColor({1.f, 1.f, 1.f}).setIntensity(5.f);
static_cast<vkgfx::SpotLight*>(spot.get())
    ->setInnerCone(20.f).setOuterCone(35.f).setRange(15.f);
scene.add(spot);
```

---

## IBL environment lighting

Place an equirectangular `.hdr` panorama at `assets/sky.hdr` **before** launching —
it is picked up automatically. The renderer bakes irradiance, prefiltered specular,
and a BRDF LUT at startup (one-time CPU cost, roughly 0.5 s for a 2K HDR).

To specify a custom path via settings:

```cpp
vkgfx::RendererSettings rs;
rs.shaderDir = "shaders";
rs.iblPath   = "assets/studio_small.hdr";   // set before constructing Renderer
```

Without an HDR file the renderer prints:
```
[VKGFX] IBL: "assets/sky.hdr" not found — using flat ambient
```
and falls back to the ambient colour set with `scene.setAmbient()`.

---

## Renderer settings reference

| Field           | Type     | Default    | Description                                   |
|-----------------|----------|------------|-----------------------------------------------|
| `vsync`         | `bool`   | `true`     | `VK_PRESENT_MODE_FIFO` (tearing-free)         |
| `validation`    | `bool`   | `false`    | Enable Vulkan validation layers               |
| `wireframe`     | `bool`   | `false`    | Render geometry as lines                      |
| `ssaoRadius`    | `float`  | `0.5`      | SSAO hemisphere radius (world units)          |
| `ssaoBias`      | `float`  | `0.025`    | SSAO depth-test bias (reduces acne)           |
| `exposure`      | `float`  | `0.0`      | EV offset applied before tone mapping         |
| `tonemapOp`     | `int`    | `0`        | `0`=ACES, `1`=Reinhard, `2`=Hable            |
| `shaderDir`     | `path`   | `"shaders"`| Directory containing compiled `.spv` files    |
| `iblPath`       | `path`   | `""`       | Equirectangular HDR for IBL; auto `assets/sky.hdr` |
| `workerThreads` | `int`    | `0`        | CPU worker threads; `0` = hardware_concurrency−1 |

---

## PBR material quick reference

```cpp
auto mat = std::make_shared<vkgfx::PBRMaterial>();

// Factors (multiplied with textures when both are set)
mat->setAlbedo({r, g, b, a})   // base colour + alpha
   .setMetallic(0.f)            // 0 = dielectric, 1 = metal
   .setRoughness(0.5f)          // 0 = mirror, 1 = fully rough
   .setEmissiveStrength(0.f);   // > 0 for glowing surfaces

// Slot constants for setTexture()
// PBRMaterial::ALBEDO      — base colour map (sRGB)
// PBRMaterial::NORMAL      — tangent-space normal map (linear)
// PBRMaterial::METALROUGH  — metallic (B) + roughness (G) (linear)
// PBRMaterial::EMISSIVE    — emissive colour map (sRGB)
// PBRMaterial::AO          — ambient occlusion (R channel) (linear)
```
