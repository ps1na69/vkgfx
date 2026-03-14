cmake_minimum_required(VERSION 3.22)
project(vkgfx VERSION 0.4.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Suppress deprecation warnings from third-party CMakeLists files
set(CMAKE_WARN_DEPRECATED OFF CACHE BOOL "" FORCE)

# ─── Vulkan (installed with Vulkan SDK) ───────────────────────────────────────
find_package(Vulkan REQUIRED)

# ─── All other dependencies fetched automatically ────────────────────────────
include(FetchContent)

# GLFW
FetchContent_Declare(glfw
    GIT_REPOSITORY https://github.com/glfw/glfw
    GIT_TAG        3.4
    GIT_SHALLOW    TRUE)
set(GLFW_BUILD_DOCS     OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL        OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glfw)

# GLM
FetchContent_Declare(glm
    GIT_REPOSITORY https://github.com/g-truc/glm
    GIT_TAG        1.0.1
    GIT_SHALLOW    TRUE)
FetchContent_MakeAvailable(glm)

# VulkanMemoryAllocator
FetchContent_Declare(vma
    GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator
    GIT_TAG        v3.1.0
    GIT_SHALLOW    TRUE)
FetchContent_MakeAvailable(vma)

# tinygltf — header-only glTF 2.0 loader (bundles its own stb_image)
FetchContent_Declare(tinygltf
    GIT_REPOSITORY https://github.com/syoyo/tinygltf
    GIT_TAG        v2.9.3
    GIT_SHALLOW    TRUE)
set(TINYGLTF_BUILD_LOADER_EXAMPLE OFF CACHE BOOL "" FORCE)
set(TINYGLTF_HEADER_ONLY           ON CACHE BOOL "" FORCE)
set(TINYGLTF_INSTALL               OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(tinygltf)

# stb — stb_image for PNG/JPG/HDR loading (texture.cpp + ibl.cpp)
FetchContent_Declare(stb
    GIT_REPOSITORY https://github.com/nothings/stb
    GIT_TAG        master
    GIT_SHALLOW    TRUE)
FetchContent_MakeAvailable(stb)

# ─── Library sources ──────────────────────────────────────────────────────────
set(VKGFX_SOURCES
    src/context.cpp
    src/window.cpp
    src/swapchain.cpp
    src/camera.cpp
    src/scene.cpp
    src/mesh.cpp
    src/texture.cpp
    src/GBuffer.cpp
    src/renderer.cpp
    src/render_graph.cpp
    src/ibl.cpp
    src/shadow.cpp
    src/gpu_culling.cpp
    src/stb_impl.cpp
)

add_library(vkgfx STATIC ${VKGFX_SOURCES})

target_include_directories(vkgfx PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${tinygltf_SOURCE_DIR}
    ${vma_SOURCE_DIR}/include
    ${stb_SOURCE_DIR}
)

target_link_libraries(vkgfx PUBLIC
    Vulkan::Vulkan
    glfw
    glm::glm
    GPUOpen::VulkanMemoryAllocator
)

target_compile_definitions(vkgfx PUBLIC
    GLM_FORCE_RADIANS
    GLM_FORCE_DEPTH_ZERO_TO_ONE
    GLM_ENABLE_EXPERIMENTAL
    GLM_FORCE_CXX17
    $<$<PLATFORM_ID:Windows>:NOMINMAX>
    $<$<PLATFORM_ID:Windows>:WIN32_LEAN_AND_MEAN>
    $<$<PLATFORM_ID:Windows>:VC_EXTRA_LEAN>
)

if(MSVC)
    target_compile_options(vkgfx PUBLIC
        /W3
        /wd4201   # nameless struct/union
        /wd4100   # unreferenced formal parameter
        /utf-8
        /MP
    )
    # MSVC debug STL's _Is_pointer_address_convertible constexpr check fires on
    # non-trivially-copyable types (Vulkan opaque handles, std::function, Mesh …).
    # Disabling iterator debugging (_IDL=0) eliminates all those false-positive
    # Clangd errors while keeping full debug info and address sanitiser support.
    target_compile_definitions(vkgfx PUBLIC
        # Disable MSVC debug iterator checking globally — it triggers a chain of
        # constexpr errors in _Is_pointer_address_convertible for non-trivially-
        # copyable types (Vulkan handles, std::function, etc.) that confuses both
        # the compiler in some edge cases and clangd's IntelliSense permanently.
        _ITERATOR_DEBUG_LEVEL=0
        _HAS_ITERATOR_DEBUGGING=0
    )
else()
    target_compile_options(vkgfx PUBLIC -Wall -Wextra -Wno-unused-parameter)
endif()

# Precompiled header
target_precompile_headers(vkgfx PRIVATE
    <vulkan/vulkan.h>
    <GLFW/glfw3.h>
    <glm/glm.hpp>
    <glm/gtc/matrix_transform.hpp>
    <vector>
    <string>
    <memory>
    <functional>
    <unordered_map>
)

# ─── Shader compilation ───────────────────────────────────────────────────────
find_program(GLSLC glslc
    HINTS
        $ENV{VULKAN_SDK}/bin
        $ENV{VULKAN_SDK}/Bin
        ${Vulkan_GLSLC_EXECUTABLE})

if(GLSLC)
    set(SHADER_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
    set(SHADER_BIN_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)
    file(MAKE_DIRECTORY ${SHADER_BIN_DIR})

    set(SHADER_SOURCES
        ${SHADER_SRC_DIR}/geometry.vert
        ${SHADER_SRC_DIR}/geometry.frag
        ${SHADER_SRC_DIR}/fullscreen_quad.vert
        ${SHADER_SRC_DIR}/ssao.frag
        ${SHADER_SRC_DIR}/ssao_blur.frag
        ${SHADER_SRC_DIR}/lighting.frag
        ${SHADER_SRC_DIR}/tonemap.frag
        ${SHADER_SRC_DIR}/shadow_depth.vert
        ${SHADER_SRC_DIR}/gpu_cull.comp
    )

    set(SHADER_SPVS)
    foreach(SHADER ${SHADER_SOURCES})
        get_filename_component(FNAME ${SHADER} NAME)
        set(SPV ${SHADER_BIN_DIR}/${FNAME}.spv)
        add_custom_command(
            OUTPUT  ${SPV}
            COMMAND ${GLSLC} ${SHADER} -o ${SPV}
            DEPENDS ${SHADER}
            COMMENT "Compiling shader: ${FNAME}"
        )
        list(APPEND SHADER_SPVS ${SPV})
    endforeach()

    add_custom_target(vkgfx_shaders ALL DEPENDS ${SHADER_SPVS})
    add_dependencies(vkgfx vkgfx_shaders)
else()
    message(WARNING "glslc not found — shaders will not be compiled automatically.")
endif()

# ─── Example executable ───────────────────────────────────────────────────────
add_executable(vkgfx_example example/main.cpp)
target_link_libraries(vkgfx_example PRIVATE vkgfx)
add_dependencies(vkgfx_example vkgfx)

if(GLSLC)
    add_custom_command(TARGET vkgfx_example POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${SHADER_BIN_DIR} $<TARGET_FILE_DIR:vkgfx_example>/shaders
        COMMENT "Copying shaders to output directory"
    )
endif()

# Copy assets (sky.hdr, textures, etc.) to the output directory so the executable
# can find them with relative paths regardless of the IDE working directory.
if(EXISTS ${CMAKE_SOURCE_DIR}/assets)
    add_custom_command(TARGET vkgfx_example POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_SOURCE_DIR}/assets $<TARGET_FILE_DIR:vkgfx_example>/assets
        COMMENT "Copying assets to output directory"
    )
endif()
