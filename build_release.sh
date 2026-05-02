#!/bin/bash
# build_release.sh — Build vkgfx release package for distribution
# Usage: ./build_release.sh [windows|linux] [debug|release]
# 
# This script creates a standalone release package that users can download
# and link to their IDE without needing CMake or Git!

set -e

PLATFORM="${1:-linux}"
CONFIG="${2:-release}"
BUILD_TYPE="$([[ "$CONFIG" == "debug" ]] && echo "Debug" || echo "Release")"
ARCH="$(uname -m)"

if [[ "$PLATFORM" == "windows" ]]; then
    OS_NAME="windows"
    CMAKE_GENERATOR="Visual Studio 17 2022"
    CMAKE_ARCH="x64"
elif [[ "$PLATFORM" == "linux" ]]; then
    OS_NAME="linux"
    CMAKE_GENERATOR="Unix Makefiles"
    CMAKE_ARCH=""
else
    echo "Unknown platform: $PLATFORM (use 'windows' or 'linux')"
    exit 1
fi

RELEASE_NAME="vkgfx-${OS_NAME}-${ARCH}-${CONFIG}"
RELEASE_DIR="release/${RELEASE_NAME}"
BUILD_DIR="build_${PLATFORM}_${CONFIG}"

echo "============================================"
echo "Building vkgfx release package"
echo "Platform: $OS_NAME ($ARCH)"
echo "Config:   $BUILD_TYPE"
echo "Output:   $RELEASE_DIR"
echo "============================================"

# Clean previous build
rm -rf "$BUILD_DIR" "$RELEASE_DIR"
mkdir -p "$BUILD_DIR" "$RELEASE_DIR"

cd "$BUILD_DIR"

# Configure with CMake - build everything statically
echo "[1/4] Configuring with CMake..."
if [[ "$PLATFORM" == "windows" ]]; then
    cmake .. \
        -G "$CMAKE_GENERATOR" \
        -A x64 \
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
        -DVKGFX_BUILD_EXAMPLES=OFF \
        -DVKGFX_BUILD_TESTS=OFF \
        -DVKGFX_ENABLE_VALIDATION=OFF \
        -DCMAKE_INSTALL_PREFIX="../${RELEASE_DIR}" \
        -DBUILD_SHARED_LIBS=OFF
else
    cmake .. \
        -G "$CMAKE_GENERATOR" \
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
        -DVKGFX_BUILD_EXAMPLES=OFF \
        -DVKGFX_BUILD_TESTS=OFF \
        -DVKGFX_ENABLE_VALIDATION=OFF \
        -DCMAKE_INSTALL_PREFIX="../${RELEASE_DIR}" \
        -DBUILD_SHARED_LIBS=OFF
fi

# Build
echo "[2/4] Building library..."
cmake --build . --config "$BUILD_TYPE" --target vkgfx -j"$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"

# Install
echo "[3/4] Installing to release directory..."
cmake --install . --config "$BUILD_TYPE"

cd ..

# Create final directory structure for easy IDE integration
echo "[4/4] Preparing standalone package..."
mkdir -p "${RELEASE_DIR}/include/vkgfx"
mkdir -p "${RELEASE_DIR}/lib"
mkdir -p "${RELEASE_DIR}/shaders"
mkdir -p "${RELEASE_DIR}/examples"
mkdir -p "${RELEASE_DIR}/docs"

# Copy public headers
cp -r include/vkgfx/*.h "${RELEASE_DIR}/include/vkgfx/"

# Copy compiled library with standard name
if [[ "$PLATFORM" == "windows" ]]; then
    # Find and copy the library file (could be in different locations)
    find "${BUILD_DIR}" -name "*.lib" -path "*/Release/*" -o -name "*.lib" -path "*/Debug/*" | head -1 | while read libfile; do
        cp "$libfile" "${RELEASE_DIR}/lib/vkgfx.lib"
    done
    # If not found in subdirs, check root
    if [[ ! -f "${RELEASE_DIR}/lib/vkgfx.lib" ]]; then
        find "${BUILD_DIR}" -name "*.lib" | head -1 | while read libfile; do
            cp "$libfile" "${RELEASE_DIR}/lib/vkgfx.lib"
        done
    fi
else
    # Linux
    find "${BUILD_DIR}" -name "libvkgfx.a" | head -1 | while read libfile; do
        cp "$libfile" "${RELEASE_DIR}/lib/libvkgfx.a"
    done
fi

# Copy compiled shaders
if [[ -d "${BUILD_DIR}/shaders" ]]; then
    cp "${BUILD_DIR}/shaders/"*.spv "${RELEASE_DIR}/shaders/" 2>/dev/null || true
fi

# Copy example source files
cp examples/*.cpp "${RELEASE_DIR}/examples/" 2>/dev/null || true

# Copy README for users
cat > "${RELEASE_DIR}/README.md" << EOF
# vkgfx — Simple Vulkan Graphics Engine

Pre-built binary release for **${OS_NAME}** (**${ARCH}**, **${CONFIG}**)

## Quick Start

### Prerequisites

Before using vkgfx, ensure you have:

1. **Vulkan SDK** installed
   - Windows: Download from https://vulkan.lunarg.com/
   - Linux: \`sudo apt install vulkan-sdk\` or equivalent

2. **C++20 compatible compiler**
   - Windows: MSVC 2019+ or Clang 12+
   - Linux: GCC 10+ or Clang 12+

### Directory Structure

\`\`\`
vkgfx-${OS_NAME}-${ARCH}-${CONFIG}/
├── include/          # Public header files
│   └── vkgfx/
│       ├── vkgfx.h   # Master include (include this one)
│       └── *.h       # Individual headers
├── lib/              # Compiled library files
│   ├── libvkgfx.a    # Linux static library
│   └── vkgfx.lib     # Windows static library
├── bin/              # Runtime binaries
│   └── shaders/      # Compiled SPIR-V shaders (*.spv)
├── examples/         # Example source code
└── README.md         # This file
\`\`\`

### Using in Your Project

#### Option 1: CMake (Recommended)

1. Copy the entire release folder to your project
2. In your CMakeLists.txt:

\`\`\`cmake
cmake_minimum_required(VERSION 3.16)
project(my_app)

set(CMAKE_CXX_STANDARD 20)

# Point to vkgfx location
set(VKGFX_ROOT "\${CMAKE_SOURCE_DIR}/thirdparty/vkgfx-${OS_NAME}-${ARCH}-${CONFIG}")

# Add vkgfx library
add_library(vkgfx STATIC IMPORTED)
set_target_properties(vkgfx PROPERTIES
    IMPORTED_LOCATION "\${VKGFX_ROOT}/lib/libvkgfx.a"
    INTERFACE_INCLUDE_DIRECTORIES "\${VKGFX_ROOT}/include"
)

# Find Vulkan
find_package(Vulkan REQUIRED)

# Your executable
add_executable(my_app src/main.cpp)
target_link_libraries(my_app PRIVATE vkgfx Vulkan::Vulkan)
\`\`\`

#### Option 2: Manual IDE Configuration

**Visual Studio (Windows):**

1. Right-click project → Properties
2. C/C++ → General → Additional Include Directories:
   - Add: \`path\\to\\vkgfx\\include\`
3. Linker → Input → Additional Dependencies:
   - Add: \`path\\to\\vkgfx\\lib\\vkgfx.lib\`
   - Add: \`vulkan-1.lib\`
4. Copy \`bin\\shaders\` folder to your executable directory

**CLion / Linux:**

1. In CMakeLists.txt (see Option 1 above)
2. Or manually add include/library paths in IDE settings

### Minimal Example

\`\`\`cpp
#include <vkgfx/vkgfx.h>
#include <iostream>

int main() {
    // Create window and context
    vkgfx::WindowConfig winCfg{};
    winCfg.width = 800;
    winCfg.height = 600;
    winCfg.title = "My App";
    
    vkgfx::Window window(winCfg);
    
    vkgfx::ContextConfig ctxCfg{};
    ctxCfg.appName = "MyApp";
    ctxCfg.validation = true;  // Enable validation layers in debug
    
    vkgfx::Context ctx(ctxCfg, window.surface());
    
    std::cout << "vkgfx initialized successfully!" << std::endl;
    return 0;
}
\`\`\`

### Important Notes

1. **Shaders**: The \`bin/shaders/\` folder must be copied to your executable's working directory
2. **Validation Layers**: This release build has validation layers **DISABLED** for better performance and smaller size. For development, build from source with \`VKGFX_ENABLE_VALIDATION=ON\`.
3. **GLFW**: vkgfx uses GLFW internally - it's statically linked in this release
4. **Other dependencies**: GLM, stb, tinyobjloader, fastgltf are all statically linked

### Troubleshooting

**Error: Cannot find Vulkan**
- Install Vulkan SDK from https://vulkan.lunarg.com/
- Ensure \`VULKAN_SDK\` environment variable is set

**Error: Missing shaders**
- Copy \`bin/shaders/\` folder to your executable directory
- Or set working directory to include shaders path

**Error: Validation layer not found**
- Install Vulkan SDK with validation layers
- Set \`ctxCfg.validation = false\` to disable (not recommended for development)

### API Reference

For full API documentation, see headers in \`include/vkgfx/\`:

- \`vkgfx.h\` — Master include, includes all public APIs
- \`window.h\` — Window creation and management
- \`context.h\` — Vulkan context and device
- \`renderer.h\` — Main rendering interface
- \`scene.h\` — Scene graph and entities
- \`mesh.h\` — 3D model loading
- \`texture.h\` — Texture loading and management
- \`camera.h\` — Camera utilities
- \`collision.h\` — Collision detection
- \`config.h\` — Configuration and settings

### License

[Your license here]

### Support

GitHub: https://github.com/[your-username]/vkgfx
Issues: https://github.com/[your-username]/vkgfx/issues
EOF

# Create ZIP archive
echo ""
echo "Creating ZIP archive..."
cd release
zip -r "${RELEASE_NAME}.zip" "${RELEASE_NAME}"
cd ..

echo ""
echo "============================================"
echo "✓ Release package created successfully!"
echo ""
echo "Archive: release/${RELEASE_NAME}.zip"
echo "Folder:  ${RELEASE_DIR}/"
echo ""
echo "To use:"
echo "  1. Extract the ZIP to your project"
echo "  2. Follow instructions in README.md"
echo "============================================"
