# VKGFX — High-Performance Vulkan 3D Graphics Library

A clean C++20 Vulkan rendering library designed for ergonomic use, inspired by
[AEspinosaDev/Vulkan-Engine](https://github.com/AEspinosaDev/Vulkan-Engine).

---

## Features

| Feature | Details |
|---|---|
| **API** | C++20, RAII, fluent builder syntax |
| **Rendering** | Deffered renderer, PBR + Phong materials |
| **BRDF** | Cook-Torrance (GGX NDF, Smith-G, Schlick-F) |
| **Lighting** | Point, Directional, Spot (up to 8 dynamic) |
| **MSAA** | x1 / x2 / x4 / x8  (auto-clamped to device max) |
| **Textures** | stb_image loading, auto mipmap generation, anisotropy |
| **Meshes** | tinyobjloader, vertex deduplication, tangent generation |
| **Cameras** | Perspective / Ortho, frustum extraction |
| **Culling** | Frustum culling with AABB per mesh |
| **Shaders** | GLSL 4.50, compiled to SPIR-V via `glslc` |
| **Sync** | Double-buffered frames, timeline semaphores |

---

## Quick Start

```cpp
#include <vkgfx/vkgfx.h>
using namespace vkgfx;

int main() {
    Window window("My App", 1280, 720);

    RendererSettings rs;
    rs.msaa      = MSAASamples::x4;
    rs.shaderDir = "shaders";
    Renderer renderer(window, rs);

    Camera camera;
    camera.setPosition({0, 1, -5}).setFov(70.f);

    Scene scene(&camera);

    // Geometry
    auto sphere = Mesh::createSphere(1.f);
    auto mat    = std::make_shared<PBRMaterial>();
    mat->setAlbedo({0.8f, 0.2f, 0.1f, 1.f})
        .setRoughness(0.3f)
        .setMetallic(0.7f);
    sphere->setMaterial(mat);
    scene.add(sphere);

    // Light
    auto light = std::make_shared<PointLight>();
    light->setPosition({3, 4, -2}).setIntensity(5.f);
    scene.add(light);

    while (!window.shouldClose()) {
        window.pollEvents();
        renderer.render(scene);
    }
    renderer.shutdown(&scene);
}
```

---

## Directory Structure

```
vkgfx/
├── include/vkgfx/
│   ├── vkgfx.h          ← master include
│   ├── types.h          ← Vertex, UBO structs, macros
│   ├── window.h         ← GLFW window wrapper
│   ├── context.h        ← Vulkan instance + device + memory
│   ├── swapchain.h      ← Swapchain + renderpass + framebuffers
│   ├── texture.h        ← stb_image texture + cache
│   ├── material.h       ← PBRMaterial / PhongMaterial
│   ├── mesh.h           ← tinyobjloader mesh + procedural shapes
│   ├── camera.h         ← Camera + Frustum
│   ├── scene.h          ← Scene graph + lights
│   └── renderer.h       ← Main orchestrator
├── src/
│   ├── window.cpp
│   ├── context.cpp
│   ├── swapchain.cpp
│   ├── texture.cpp
│   ├── mesh.cpp
│   ├── camera.cpp
│   ├── scene.cpp
│   └── renderer.cpp
├── shaders/
│   ├── pbr.vert / pbr.frag       ← Full Cook-Torrance PBR
│   ├── phong.vert / phong.frag   ← Classic Blinn-Phong
│   └── unlit.vert / unlit.frag   ← Unlit / debug
├── example/
│   └── main.cpp                  ← Full demo application
└── compile_shaders.sh
└── compile_shaders.bat
```

---

## Dependencies

| Library | Version | Purpose |
|---|---|---|
| Vulkan SDK | 1.3.x | Core API |
| GLFW | 3.3+ | Windowing + input |
| GLM | 0.9.9+ | Math |
| stb_image | single-header | Texture loading |
| tinyobjloader | single-header | OBJ mesh loading |

Install (Ubuntu):
```bash
sudo apt install libglfw3-dev libglm-dev vulkan-sdk
# stb_image and tinyobjloader are header-only — just drop them in your include path
```

---

## Building

```bash
# 1. Compile shaders first
chmod +x compile_shaders.sh
./compile_shaders.sh

# 2. Build (example using g++ directly)
g++ -std=c++20 -O2 \
  example/main.cpp src/*.cpp \
  -I include \
  -lglfw -lvulkan -ldl -lpthread \
  -o vkgfx_demo

# 3. Run
./vkgfx_demo
```

Or with CMake:
```cmake
cmake_minimum_required(VERSION 3.20)
project(VKGFXApp)
set(CMAKE_CXX_STANDARD 20)

find_package(Vulkan REQUIRED)
find_package(glfw3  REQUIRED)

file(GLOB SRC src/*.cpp)
add_executable(vkgfx_demo example/main.cpp ${SRC})
target_include_directories(vkgfx_demo PRIVATE include)
target_link_libraries(vkgfx_demo PRIVATE Vulkan::Vulkan glfw)
```

---

## Performance Optimisations

The following techniques are baked into the library:

### 1. Push Constants for Model Matrices
Per-object model+normal matrices are uploaded via push constants rather than
uniform buffers. This eliminates descriptor overhead and is much faster for
hundreds of objects.

```glsl
layout(push_constant) uniform PushConstants {
    mat4 model;
    mat4 normalMatrix;
} push;
```

### 2. Device-Local GPU Buffers via Staging
Vertex and index data are uploaded once to `DEVICE_LOCAL` memory through a
staging buffer. This is the fastest memory accessible by the GPU, with no CPU
contention.

```cpp
AllocatedBuffer Context::uploadBuffer(const void* data, VkDeviceSize size,
                                       VkBufferUsageFlags usage) const { ... }
```

### 3. Frustum Culling
The camera's frustum is extracted every frame using the Gribb-Hartmann method.
Each mesh's AABB is tested before submitting any draw calls.

```cpp
std::vector<Mesh*> visible = scene.visibleMeshes(); // frustum-culled
```

### 4. Pipeline Caching
Vulkan pipelines are cached by `(shader × cullMode × polygonMode × alphaBlend)`
key. A `VkPipelineCache` is also maintained for faster PSO creation across runs.

### 5. Automatic Mipmap Generation
All textures get full mip chains generated on the GPU via `vkCmdBlitImage`.
This improves texture cache efficiency dramatically for distant surfaces.

### 6. Vertex Deduplication on Load
OBJ loading uses an `unordered_map<Vertex, index>` to deduplicate vertices,
significantly reducing GPU memory and bandwidth.

### 7. Dynamic Viewport / Scissor
Pipeline creation uses `VK_DYNAMIC_STATE_VIEWPORT / SCISSOR`, so the swapchain
can resize without pipeline recreation.

### 8. Descriptor Set Double-Buffering
Per-frame descriptors avoid GPU-CPU synchronisation stalls. The renderer cycles
through `MAX_FRAMES_IN_FLIGHT` sets so a new frame can begin while the previous
is still rendering.

### 9. Texture Cache
`TextureCache` prevents the same file from being loaded/uploaded twice, reducing
both CPU and GPU memory waste.

### 10. Index Buffers Everywhere
All procedural and loaded meshes always use index buffers, enabling vertex
reuse and hardware post-transform cache hits.

---

## Controls (Example App)

| Key | Action |
|---|---|
| W / S / A / D | Move forward / back / left / right |
| Q / E | Move down / up |
| Mouse | Look around |
| Escape | Quit |

---

## Extending the Library

### Adding a new material type

1. Add a properties struct in `material.h` (must be `alignas(16)`)
2. Subclass `Material`, implement `propertiesData()`, `shaderName()` etc.
3. Write `.vert` + `.frag` GLSL shaders, compile to `.spv`
4. The renderer's pipeline cache and descriptor system handle everything else

### Adding a render pass (shadows, post-FX)

Extend `Renderer` to create additional render passes before the main forward
pass. `Swapchain` already exposes its render pass handle for reference.

---
