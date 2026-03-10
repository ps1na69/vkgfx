cmake_minimum_required(VERSION 4.1)

# ─────────────────────────────────────────────────────────────────────────────
#  VKGFX — High-Performance Vulkan 3D Graphics Library
#  C++20 | Vulkan 1.3 | GLFW | GLM | stb_image | tinyobjloader
# ─────────────────────────────────────────────────────────────────────────────
project(vkgfx
    VERSION      1.0.0
    DESCRIPTION  "High-Performance Vulkan 3D Graphics Library"
    LANGUAGES    CXX
)

# ─── C++ Standard ─────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD          20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS        OFF)   # no GNU extensions — strict standard

# ─── Build type defaults ──────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif()
message(STATUS "[VKGFX] Build type : ${CMAKE_BUILD_TYPE}")
message(STATUS "[VKGFX] Compiler   : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")

# ─── Options ──────────────────────────────────────────────────────────────────
option(VKGFX_BUILD_EXAMPLE        "Build the example application"          ON)
option(VKGFX_BUILD_SHARED         "Build vkgfx as a shared library"        OFF)
option(VKGFX_ENABLE_VALIDATION    "Force-enable Vulkan validation layers"  ON)
option(VKGFX_COMPILE_SHADERS      "Compile GLSL shaders to SPIR-V"        ON)
option(VKGFX_WERROR               "Treat warnings as errors"               OFF)

# ─── Module path ──────────────────────────────────────────────────────────────
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# ═══════════════════════════════════════════════════════════════════════════════
#  Dependencies
# ═══════════════════════════════════════════════════════════════════════════════

# ── Vulkan ────────────────────────────────────────────────────────────────────
find_package(Vulkan REQUIRED)
if(NOT Vulkan_FOUND)
    message(FATAL_ERROR "Vulkan SDK not found. Install from https://vulkan.lunarg.com/")
endif()
message(STATUS "[VKGFX] Vulkan     : ${Vulkan_VERSION} @ ${Vulkan_INCLUDE_DIRS}")

# Detect glslc for shader compilation
find_program(GLSLC_EXECUTABLE
    NAMES glslc
    HINTS
        "$ENV{VULKAN_SDK}/bin"
        "$ENV{VK_SDK_PATH}/bin"
        "/usr/bin"
        "/usr/local/bin"
)
if(GLSLC_EXECUTABLE)
    message(STATUS "[VKGFX] glslc      : ${GLSLC_EXECUTABLE}")
else()
    message(WARNING "[VKGFX] glslc not found — shader compilation disabled")
    set(VKGFX_COMPILE_SHADERS OFF)
endif()

# ── GLFW ──────────────────────────────────────────────────────────────────────
set(VKGFX_GLFW_FETCHED FALSE)
find_package(glfw3 3.3 QUIET)
if(NOT glfw3_FOUND)
    message(STATUS "[VKGFX] glfw3 not found via find_package, trying FetchContent...")
    include(FetchContent)
    FetchContent_Declare(
        glfw
        GIT_REPOSITORY https://github.com/glfw/glfw.git
        GIT_TAG        3.3.9
        GIT_SHALLOW    TRUE
    )
    set(GLFW_BUILD_DOCS     OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(GLFW_INSTALL        OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(glfw)
    set(VKGFX_GLFW_FETCHED TRUE)
    message(STATUS "[VKGFX] GLFW       : fetched from GitHub")
else()
    message(STATUS "[VKGFX] GLFW       : ${glfw3_VERSION}")
endif()

# ── GLM ───────────────────────────────────────────────────────────────────────
find_package(glm QUIET)
if(NOT glm_FOUND)
    message(STATUS "[VKGFX] GLM not found via find_package, trying FetchContent...")
    include(FetchContent)
    FetchContent_Declare(
        glm
        GIT_REPOSITORY https://github.com/g-truc/glm.git
        GIT_TAG        1.0.1
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(glm)
    message(STATUS "[VKGFX] GLM        : fetched from GitHub")
else()
    message(STATUS "[VKGFX] GLM        : ${glm_VERSION}")
endif()

# ── Vulkan Memory Allocator (VMA) ─────────────────────────────────────────────
# VMA dramatically reduces vkAllocateMemory calls by pooling GPU allocations.
# It is a single-header C++ library, so it only adds compile time, not link time.
include(FetchContent)
FetchContent_Declare(
    vma
    GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
    GIT_TAG        v3.1.0
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(vma)
message(STATUS "[VKGFX] VMA        : fetched from GitHub (v3.1.0)")

# ── Header-only dependencies (stb_image + tinyobjloader) ─────────────────────
#   Look in thirdparty/ first, then fall back to FetchContent
set(VKGFX_THIRDPARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty")

# stb_image
find_path(STB_INCLUDE_DIR stb_image.h
    HINTS "${VKGFX_THIRDPARTY_DIR}/stb" "${VKGFX_THIRDPARTY_DIR}" /usr/include /usr/local/include
)
if(NOT STB_INCLUDE_DIR)
    message(STATUS "[VKGFX] stb not found locally, fetching...")
    include(FetchContent)
    FetchContent_Declare(
        stb
        GIT_REPOSITORY https://github.com/nothings/stb.git
        GIT_TAG        master
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(stb)
    set(STB_INCLUDE_DIR "${stb_SOURCE_DIR}")
endif()
message(STATUS "[VKGFX] stb_image  : ${STB_INCLUDE_DIR}")

# tinyobjloader
find_path(TINYOBJ_INCLUDE_DIR tiny_obj_loader.h
    HINTS "${VKGFX_THIRDPARTY_DIR}/tinyobjloader" "${VKGFX_THIRDPARTY_DIR}" /usr/include /usr/local/include
)
if(NOT TINYOBJ_INCLUDE_DIR)
    message(STATUS "[VKGFX] tinyobjloader not found locally, fetching...")
    include(FetchContent)
    FetchContent_Declare(
        tinyobjloader
        GIT_REPOSITORY https://github.com/tinyobjloader/tinyobjloader.git
        GIT_TAG        v2.0.0rc13
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(tinyobjloader)
    set(TINYOBJ_INCLUDE_DIR "${tinyobjloader_SOURCE_DIR}")
endif()
message(STATUS "[VKGFX] tinyobj    : ${TINYOBJ_INCLUDE_DIR}")

# ═══════════════════════════════════════════════════════════════════════════════
#  Shader compilation (GLSL → SPIR-V)
# ═══════════════════════════════════════════════════════════════════════════════
set(VKGFX_SHADER_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shaders")
set(VKGFX_SHADER_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")
file(MAKE_DIRECTORY "${VKGFX_SHADER_OUTPUT_DIR}")

if(VKGFX_COMPILE_SHADERS)
    file(GLOB_RECURSE GLSL_SOURCES
        "${VKGFX_SHADER_SOURCE_DIR}/*.vert"
        "${VKGFX_SHADER_SOURCE_DIR}/*.frag"
        "${VKGFX_SHADER_SOURCE_DIR}/*.comp"
        "${VKGFX_SHADER_SOURCE_DIR}/*.geom"
    )

    set(SPIRV_FILES "")
    foreach(GLSL_FILE ${GLSL_SOURCES})
        get_filename_component(SHADER_NAME "${GLSL_FILE}" NAME)
        set(SPV_FILE "${VKGFX_SHADER_OUTPUT_DIR}/${SHADER_NAME}.spv")

        add_custom_command(
            OUTPUT  "${SPV_FILE}"
            COMMAND "${GLSLC_EXECUTABLE}"
                    "${GLSL_FILE}"
                    -o "${SPV_FILE}"
                    --target-env=vulkan1.3
                    -O                          # optimise
                    $<$<CONFIG:Debug>:-g>        # debug info in Debug
            DEPENDS "${GLSL_FILE}"
            COMMENT "Compiling shader: ${SHADER_NAME} → ${SHADER_NAME}.spv"
            VERBATIM
        )
        list(APPEND SPIRV_FILES "${SPV_FILE}")
    endforeach()

    add_custom_target(vkgfx_shaders ALL
        DEPENDS ${SPIRV_FILES}
        COMMENT "Building VKGFX shaders"
    )
    message(STATUS "[VKGFX] Shaders    : ${VKGFX_SHADER_OUTPUT_DIR}")
endif()

# ═══════════════════════════════════════════════════════════════════════════════
#  Library target
# ═══════════════════════════════════════════════════════════════════════════════
set(VKGFX_SOURCES
    src/window.cpp
    src/context.cpp
    src/swapchain.cpp
    src/texture.cpp
    src/mesh.cpp
    src/camera.cpp
    src/scene.cpp
    src/renderer.cpp
)

set(VKGFX_HEADERS
    include/vkgfx/vkgfx.h
    include/vkgfx/types.h
    include/vkgfx/window.h
    include/vkgfx/context.h
    include/vkgfx/swapchain.h
    include/vkgfx/texture.h
    include/vkgfx/material.h
    include/vkgfx/mesh.h
    include/vkgfx/camera.h
    include/vkgfx/scene.h
    include/vkgfx/postprocess.h
    include/vkgfx/renderer.h
    include/vkgfx/thread_pool.h
)

if(VKGFX_BUILD_SHARED)
    add_library(vkgfx SHARED ${VKGFX_SOURCES} ${VKGFX_HEADERS})
    message(STATUS "[VKGFX] Library    : SHARED")
else()
    add_library(vkgfx STATIC ${VKGFX_SOURCES} ${VKGFX_HEADERS})
    message(STATUS "[VKGFX] Library    : STATIC")
endif()

# Alias so consumers can do target_link_libraries(foo vkgfx::vkgfx)
add_library(vkgfx::vkgfx ALIAS vkgfx)

# ─── Include directories ──────────────────────────────────────────────────────
# PUBLIC  → consumers of the library also get these paths
# PRIVATE → only needed when compiling vkgfx itself
#
# stb_image and tinyobjloader are #included only in .cpp files (texture.cpp,
# mesh.cpp), never in any public header → PRIVATE.
# Vulkan headers ARE referenced in public headers (types.h) → PUBLIC.
target_include_directories(vkgfx
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        ${Vulkan_INCLUDE_DIRS}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${STB_INCLUDE_DIR}       # only used in src/texture.cpp
        ${TINYOBJ_INCLUDE_DIR}   # only used in src/mesh.cpp
)

# ─── Link libraries ───────────────────────────────────────────────────────────
# GLFW is linked PRIVATELY:
#   - window.h includes <GLFW/glfw3.h> via its own include path, but consumers
#     of vkgfx don't call GLFW directly — they go through Window.
#   - Keeping it PRIVATE avoids the FetchContent export-set problem entirely:
#     a fetched 'glfw' target cannot appear in vkgfxTargets.cmake.
# Vulkan is PUBLIC because Vulkan types appear in public vkgfx headers.
target_link_libraries(vkgfx
    PUBLIC
        Vulkan::Vulkan
        $<TARGET_NAME_IF_EXISTS:glm::glm>
    PRIVATE
        glfw
        # Do NOT link GPUOpen::VulkanMemoryAllocator here.
        # VMA is a header-only library — there is nothing to link.
        # Linking the imported target (even PRIVATE) causes CMake to record it
        # in the static library's link interface, triggering:
        #   "install(EXPORT) requires target VulkanMemoryAllocator not in any export set"
        # Instead we propagate only the include path below.
)

# VMA headers only — no link target, no export problem.
# Must be PUBLIC because context.h (a public header) includes <vk_mem_alloc.h>.
# We use the raw directory variable instead of GPUOpen::VulkanMemoryAllocator
# so no imported target appears in the export set.
target_include_directories(vkgfx PUBLIC
    $<BUILD_INTERFACE:${vma_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# ── Unity build (reduces TU overhead during development rebuilds) ─────────────
set_target_properties(vkgfx PROPERTIES UNITY_BUILD OFF)  # set ON to combine TUs

# ── Parallel compilation ──────────────────────────────────────────────────────
cmake_host_system_information(RESULT CPU_CORES QUERY NUMBER_OF_LOGICAL_CORES)
set_property(GLOBAL PROPERTY JOB_POOLS compile_pool=${CPU_CORES})
if(NOT DEFINED CMAKE_BUILD_PARALLEL_LEVEL)
    set(CMAKE_BUILD_PARALLEL_LEVEL ${CPU_CORES} CACHE STRING "Parallel build jobs" FORCE)
endif()

# Propagate GLFW include dirs to consumers (needed because window.h includes
# <GLFW/glfw3.h>) without propagating the link target itself.
if(VKGFX_GLFW_FETCHED)
    # FetchContent: glfw target exists but is not installed — use include dir only
    target_include_directories(vkgfx PUBLIC
        $<BUILD_INTERFACE:${glfw_SOURCE_DIR}/include>
    )
else()
    # System install: glfw3::glfw3 carries its own include dirs via INTERFACE
    target_link_libraries(vkgfx INTERFACE glfw)
endif()

# ─── Compile definitions ──────────────────────────────────────────────────────
target_compile_definitions(vkgfx
    PUBLIC
        GLM_FORCE_RADIANS
        GLM_FORCE_DEPTH_ZERO_TO_ONE
        GLM_ENABLE_EXPERIMENTAL
        $<$<BOOL:${VKGFX_ENABLE_VALIDATION}>:VKGFX_FORCE_VALIDATION>
        # VMA configuration
        VMA_STATIC_VULKAN_FUNCTIONS=0
        VMA_DYNAMIC_VULKAN_FUNCTIONS=1
    PRIVATE
        VKGFX_SHADER_DIR="${VKGFX_SHADER_OUTPUT_DIR}"
        $<$<BOOL:${VKGFX_BUILD_SHARED}>:VKGFX_BUILD_SHARED>
)

# ─── Platform-specific ────────────────────────────────────────────────────────
if(WIN32)
    target_compile_definitions(vkgfx PUBLIC
        VK_USE_PLATFORM_WIN32_KHR
        NOMINMAX
        WIN32_LEAN_AND_MEAN
    )
elseif(APPLE)
    target_compile_definitions(vkgfx PUBLIC VK_USE_PLATFORM_METAL_EXT)
    target_link_libraries(vkgfx PUBLIC "-framework Cocoa" "-framework IOKit" "-framework CoreFoundation")
elseif(UNIX)
    target_compile_definitions(vkgfx PUBLIC VK_USE_PLATFORM_XCB_KHR)
    target_link_libraries(vkgfx PUBLIC dl pthread)
    find_package(Threads REQUIRED)
    target_link_libraries(vkgfx PUBLIC Threads::Threads)
endif()

# ─── Compiler warnings ────────────────────────────────────────────────────────
if(MSVC)
    target_compile_options(vkgfx PRIVATE
        /W4
        /wd4201  # nameless struct/union
        /wd4100  # unreferenced formal parameter
        /Zc:__cplusplus  # MSVC by default reports __cplusplus=199711 regardless of /std:c++XX
                         # glm/gtx/hash.hpp checks __cplusplus >= 201103L and fails without this
        $<$<BOOL:${VKGFX_WERROR}>:/WX>
    )
else()
    target_compile_options(vkgfx PRIVATE
        -Wall -Wextra -Wpedantic
        -Wno-unused-parameter
        -Wno-missing-field-initializers
        $<$<BOOL:${VKGFX_WERROR}>:-Werror>
        # Release optimisations
        $<$<CONFIG:Release>:-O3 -march=native -ffast-math>
        # Debug sanitisers
        $<$<CONFIG:Debug>:-fsanitize=address,undefined>
    )
    target_link_options(vkgfx PRIVATE
        $<$<CONFIG:Debug>:-fsanitize=address,undefined>
    )
endif()

# ─── Precompiled header (speeds up rebuilds significantly) ────────────────────
target_precompile_headers(vkgfx PRIVATE
    <vector>
    <string>
    <memory>
    <unordered_map>
    <array>
    <functional>
    <vulkan/vulkan.h>
    <GLFW/glfw3.h>
    <glm/glm.hpp>
)

# Shader dependency
if(VKGFX_COMPILE_SHADERS)
    add_dependencies(vkgfx vkgfx_shaders)
endif()

# ═══════════════════════════════════════════════════════════════════════════════
#  Example application
# ═══════════════════════════════════════════════════════════════════════════════
if(VKGFX_BUILD_EXAMPLE)
    add_executable(vkgfx_example example/main.cpp
    )

    target_link_libraries(vkgfx_example PRIVATE vkgfx::vkgfx)

    target_compile_definitions(vkgfx_example PRIVATE
        VKGFX_SHADER_DIR="${VKGFX_SHADER_OUTPUT_DIR}"
        VKGFX_ASSET_DIR="${CMAKE_CURRENT_SOURCE_DIR}/assets"
    )

    # Copy shader output dir next to the binary for convenience
    add_custom_command(TARGET vkgfx_example POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E copy_directory
            "${VKGFX_SHADER_OUTPUT_DIR}"
            "$<TARGET_FILE_DIR:vkgfx_example>/shaders"
        COMMENT "Copying compiled shaders to output directory"
    )

    # Copy assets folder next to the binary so relative paths like
    # "assets/cat.png" work regardless of the working directory.
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/assets")
        add_custom_command(TARGET vkgfx_example POST_BUILD
            COMMAND "${CMAKE_COMMAND}" -E copy_directory
                "${CMAKE_CURRENT_SOURCE_DIR}/assets"
                "$<TARGET_FILE_DIR:vkgfx_example>/assets"
            COMMENT "Copying assets to output directory"
        )
    endif()

    # Set the VS / CLion debugger working directory to the binary output dir
    # so that relative paths ("shaders/", "assets/") resolve correctly when
    # launching from the IDE.
    set_target_properties(vkgfx_example PROPERTIES
        VS_DEBUGGER_WORKING_DIRECTORY "$<TARGET_FILE_DIR:vkgfx_example>"
    )

    if(VKGFX_COMPILE_SHADERS)
        add_dependencies(vkgfx_example vkgfx_shaders)
    endif()

    message(STATUS "[VKGFX] Example    : ON  (vkgfx_example)")
else()
    message(STATUS "[VKGFX] Example    : OFF")
endif()

# ═══════════════════════════════════════════════════════════════════════════════
#  Install rules
# ═══════════════════════════════════════════════════════════════════════════════
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# vkgfx itself
install(TARGETS vkgfx
    EXPORT   vkgfxTargets
    ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# If GLFW was fetched (not a system install) we must also install + export it,
# otherwise vkgfxTargets.cmake would reference a non-existent imported target.
if(VKGFX_GLFW_FETCHED)
    install(TARGETS glfw
        EXPORT   vkgfxTargets
        ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
endif()

install(DIRECTORY include/vkgfx
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT vkgfxTargets
    FILE      vkgfxTargets.cmake
    NAMESPACE vkgfx::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/vkgfx
)

# Package config
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/vkgfxConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/vkgfxConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/vkgfx
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/vkgfxConfigVersion.cmake"
    VERSION       ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/vkgfxConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/vkgfxConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/vkgfx
)

# ─── Install compiled shaders ─────────────────────────────────────────────────
if(VKGFX_COMPILE_SHADERS)
    install(DIRECTORY "${VKGFX_SHADER_OUTPUT_DIR}/"
        DESTINATION ${CMAKE_INSTALL_DATADIR}/vkgfx/shaders
        FILES_MATCHING PATTERN "*.spv"
    )
endif()

# ═══════════════════════════════════════════════════════════════════════════════
#  Summary
# ═══════════════════════════════════════════════════════════════════════════════
message(STATUS "")
message(STATUS "┌────────────────────────────────────────┐")
message(STATUS "│          VKGFX Configuration           │")
message(STATUS "├────────────────────────────────────────┤")
message(STATUS "│ Version    : ${PROJECT_VERSION}                      │")
message(STATUS "│ Build type : ${CMAKE_BUILD_TYPE}                   │")
message(STATUS "│ Shared lib : ${VKGFX_BUILD_SHARED}                        │")
message(STATUS "│ Example    : ${VKGFX_BUILD_EXAMPLE}                         │")
message(STATUS "│ Shaders    : ${VKGFX_COMPILE_SHADERS}                         │")
message(STATUS "│ Validation : ${VKGFX_ENABLE_VALIDATION}                        │")
message(STATUS "│ Werror     : ${VKGFX_WERROR}                        │")
message(STATUS "└────────────────────────────────────────┘")
message(STATUS "")
