# VKGFX — High-Performance Vulkan 3D Graphics Engine

[![CI](https://github.com/ps1na69/vkgfx/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/ps1na69/vkgfx/actions/workflows/ci.yml)
[![CI (advanced-rendering)](https://github.com/ps1na69/vkgfx/actions/workflows/ci.yml/badge.svg?branch=feature%2Fadvanced-rendering)](https://github.com/ps1na69/vkgfx/actions/workflows/ci.yml)

A C++20 Vulkan deferred renderer: PBR, IBL, CSM directional shadows, point-light cubemap shadows, and a GPU timestamp profiling overlay.

---

## Features

| Feature | Details |
|---|---|
| **Rendering** | Deferred G-buffer → PBR lighting → HDR tonemap |
| **Materials** | Cook-Torrance BRDF (GGX NDF, Smith-G, Schlick-F) |
| **IBL** | Equirectangular HDR → irradiance + prefiltered env cubes |
| **Shadows** | Directional PCF 2048² + Point-light cubemap PCF 512² |
| **Lighting** | Directional + up to 8 point lights, fully dynamic |
| **Meshes** | glTF 2.0 (fastgltf), OBJ (tinyobjloader), procedurals |
| **Culling** | Frustum culling with per-mesh AABB |
| **Frame graph** | DAG-based, Vulkan 1.3 synchronization2 barriers |
| **GPU Profiler** | Per-pass timestamps, VRAM budget, ImGui overlay |
| **CI** | Ubuntu 24.04 + Windows 2022, Debug + Release |

---

## Quick Start

```cpp
#include <vkgfx/vkgfx.h>
using namespace vkgfx;

int main() {
    Window window("My App", 1280, 720);

    RendererConfig cfg;
    cfg.ibl.hdrPath           = "assets/sky.hdr";
    cfg.profiling.enabled     = true;  // needs -DVKGFX_ENABLE_PROFILING=ON
    cfg.profiling.showOverlay = true;
    Renderer renderer(window, cfg);

    Camera camera;
    scene.setCamera(&camera);

    auto light = std::make_shared<PointLight>();
    light->setPosition({3, 4, -2}).setIntensity(5.f).setCastsShadow(true);
    scene.add(light);

    while (!window.shouldClose()) {
        window.pollEvents();
        renderer.render(scene);
        renderer.renderProfilerOverlay(); // between ImGui::NewFrame / Render
    }
    renderer.shutdown();
}
```

---

## GPU Profiler

### Enable

```cmake
cmake -DVKGFX_ENABLE_PROFILING=ON ..
```

```cpp
cfg.profiling.enabled = true;
```

### Read stats

```cpp
const FrameStats& s = renderer.profilerStats();
for (auto& p : s.passes)
    printf("%s: %.3f ms\n", p.name.c_str(), p.gpuMs);
printf("VRAM: %.0f / %.0f MB\n", s.vramUsedMB, s.vramBudgetMB);
```

The overlay auto-positions top-right and colour-codes pass timings (green < 0.5 ms, yellow < 2 ms, red ≥ 2 ms).  Without `-DVKGFX_ENABLE_PROFILING=ON` all calls are no-ops — no `#ifdef` needed in user code.

---

## Building

```bash
cmake -S . -B build -DVKGFX_ENABLE_PROFILING=ON
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

---

## CI

Both Ubuntu 24.04 and Windows 2022 build Debug + Release with `-DVKGFX_ENABLE_PROFILING=ON`, run `ctest`, install, and verify the install layout on every push and pull request.
