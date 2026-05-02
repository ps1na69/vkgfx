# vkgfx Release Build Guide

This document explains how to build and distribute vkgfx as a pre-compiled library that users can easily integrate into their projects without needing CMake or Git.

## Overview

The goal is to create release packages that contain:
- `include/` - Public header files
- `lib/` - Compiled static library
- `bin/shaders/` - Compiled SPIR-V shaders
- `examples/` - Example source code
- `README.md` - Usage instructions

Users can download the ZIP file from GitHub Releases and link it to their IDE.

## Prerequisites for Building Releases

### Linux
```bash
# Install Vulkan SDK
sudo apt install libvulkan-dev vulkan-validationlayers-dev glslc

# Install build tools
sudo apt install cmake g++ zip
```

### Windows
```powershell
# Install Vulkan SDK from https://vulkan.lunarg.com/
# Install Visual Studio 2019+ with C++ workload
# Install CMake from https://cmake.org/
# Install 7-Zip or ensure zip is in PATH
```

## Building Release Packages

### Linux Release
```bash
./build_release.sh linux release
```

### Linux Debug
```bash
./build_release.sh linux debug
```

### Windows Release (run on Windows)
```powershell
.\build_release.bat windows release
```

Or if using WSL/bash on Windows:
```bash
./build_release.sh windows release
```

## Output

The script creates:
- `release/vkgfx-linux-x86_64-release/` - Folder with all files
- `release/vkgfx-linux-x86_64-release.zip` - ZIP archive for distribution

## Directory Structure After Build

```
vkgfx-linux-x86_64-release/
├── include/
│   └── vkgfx/
│       ├── vkgfx.h         # Master include
│       ├── camera.h
│       ├── collision.h
│       ├── config.h
│       ├── context.h
│       ├── frame_graph.h
│       ├── ibl.h
│       ├── material.h
│       ├── mesh.h
│       ├── profiler.h
│       ├── renderer.h
│       ├── scene.h
│       ├── swapchain.h
│       ├── texture.h
│       ├── types.h
│       ├── vk_raii.h
│       └── window.h
├── lib/
│   └── libvkgfx.a          # Static library
├── bin/
│   └── shaders/            # Compiled .spv files
├── examples/
│   ├── CMakeLists.txt
│   └── *.cpp               # Example sources
└── README.md               # User instructions
```

## Using the Release Package

### Method 1: With CMake (Recommended)

Create a `CMakeLists.txt` in your project:

```cmake
cmake_minimum_required(VERSION 3.16)
project(my_app)

set(CMAKE_CXX_STANDARD 20)

# Path to vkgfx release
set(VKGFX_ROOT "${CMAKE_SOURCE_DIR}/thirdparty/vkgfx-linux-x86_64-release")

# Import vkgfx library
add_library(vkgfx STATIC IMPORTED)
set_target_properties(vkgfx PROPERTIES
    IMPORTED_LOCATION "${VKGFX_ROOT}/lib/libvkgfx.a"
    INTERFACE_INCLUDE_DIRECTORIES "${VKGFX_ROOT}/include"
)

# Find Vulkan
find_package(Vulkan REQUIRED)

# Your executable
add_executable(my_app src/main.cpp)
target_link_libraries(my_app PRIVATE vkgfx Vulkan::Vulkan)

# Copy shaders to output
add_custom_command(TARGET my_app POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${VKGFX_ROOT}/bin/shaders"
        "$<TARGET_FILE_DIR:my_app>/shaders"
)
```

### Method 2: Visual Studio (Windows)

1. Right-click project → Properties
2. **C/C++** → **General** → **Additional Include Directories**:
   - Add: `path\to\vkgfx\include`
3. **Linker** → **Input** → **Additional Dependencies**:
   - Add: `path\to\vkgfx\lib\vkgfx.lib`
   - Add: `vulkan-1.lib`
4. Copy `bin\shaders` folder to your executable directory

### Method 3: CLion / Linux Makefile

**CLion:**
- Use CMake method above

**Manual Makefile:**
```makefile
CXX = g++
CXXFLAGS = -std=c++20 -I/path/to/vkgfx/include
LDFLAGS = -L/path/to/vkgfx/lib -lvkgfx -lvulkan

my_app: main.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)
```

## Minimal Example

```cpp
#include <vkgfx/vkgfx.h>
#include <iostream>

int main() {
    // Create window
    vkgfx::WindowConfig winCfg{};
    winCfg.width = 800;
    winCfg.height = 600;
    winCfg.title = "My App";
    
    vkgfx::Window window(winCfg);
    
    // Create Vulkan context
    vkgfx::ContextConfig ctxCfg{};
    ctxCfg.appName = "MyApp";
    ctxCfg.validation = true;  // Enable validation in debug
    
    vkgfx::Context ctx(ctxCfg, window.surface());
    
    std::cout << "vkgfx initialized!" << std::endl;
    return 0;
}
```

## Distributing on GitHub Releases

1. Go to your repository on GitHub
2. Click "Releases" → "Draft a new release"
3. Create a new tag (e.g., `v0.1.0`)
4. Upload the ZIP files:
   - `vkgfx-linux-x86_64-release.zip`
   - `vkgfx-windows-x86_64-release.zip`
5. Write release notes
6. Publish

## For Users: Download and Use

1. Go to GitHub Releases page
2. Download the ZIP for your OS
3. Extract to your project folder
4. Follow the README.md instructions
5. Start coding!

## Troubleshooting

### Build fails with "Cannot find Vulkan"
- Install Vulkan SDK
- Ensure `VULKAN_SDK` environment variable is set

### Build fails with "Cannot find glslc"
- Install Vulkan SDK (includes glslc shader compiler)
- Or install separately: `sudo apt install glslc`

### Runtime error: "Cannot find shaders"
- Copy `bin/shaders/` folder to your executable's working directory
- Or run executable from the correct directory

### Validation layer errors
- Install Vulkan SDK with validation layers
- Or set `ctxCfg.validation = false` (not recommended for development)

## Notes

- All dependencies (GLFW, GLM, stb, tinyobjloader, fastgltf, VMA) are statically linked
- Users only need Vulkan SDK installed
- The library is compiled with C++20 standard
- Shaders must be copied to the runtime directory
