cmake_minimum_required(VERSION 3.24)

# FetchContent_Populate() is deprecated in CMake 3.28+ (CMP0169).
# We use the manual populate+add_subdirectory(EXCLUDE_FROM_ALL) pattern
# intentionally to suppress third-party install rules, so silence the warning.
if(POLICY CMP0169)
    cmake_policy(SET CMP0169 OLD)
endif()
project(vkgfx VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(VKGFX_BUILD_EXAMPLES    "Build example applications"               ON)
option(VKGFX_BUILD_TESTS       "Build unit tests"                         ON)
option(VKGFX_ENABLE_VALIDATION "Enable Vulkan validation layers in Debug"  ON)

include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

# ── Vulkan Headers ────────────────────────────────────────────────────────────
# Always fetch via FetchContent — no Vulkan SDK required on any machine.
message(STATUS "vkgfx: fetching Vulkan-Headers")
FetchContent_Declare(vulkan_headers
    GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Headers.git
    GIT_TAG        v1.3.281
    GIT_SHALLOW    TRUE)
FetchContent_GetProperties(vulkan_headers)
if(NOT vulkan_headers_POPULATED)
    FetchContent_Populate(vulkan_headers)
    # Headers only — no add_subdirectory, we reference the path directly.
endif()
set(VULKAN_HEADERS_INCLUDE_DIR "${vulkan_headers_SOURCE_DIR}/include")

# ── volk (Vulkan meta-loader) ─────────────────────────────────────────────────
# volk opens vulkan-1.dll at runtime via LoadLibrary(). That DLL ships with
# every GPU driver (NVIDIA / AMD / Intel) so end-users do NOT need the Vulkan
# SDK installed at all — only the GPU driver, which they already have.
#
# How it works:
#   1. VK_NO_PROTOTYPES (defined below) tells vulkan.h to skip the extern
#      function declarations.
#   2. volk.h re-declares every Vulkan function as a function-pointer global.
#   3. volk's static library provides the definitions of those globals.
#   4. volkInitialize() (called inside vkgfx::Context) does the LoadLibrary.
#
# End result: vkgfx.lib already contains the volk object code, so users link
# only vkgfx.lib — no vulkan-1.lib, no SDK.
message(STATUS "vkgfx: fetching volk")
FetchContent_Declare(volk
    GIT_REPOSITORY https://github.com/zeux/volk.git
    GIT_TAG        vulkan-sdk-1.3.275.0
    GIT_SHALLOW    TRUE)
FetchContent_GetProperties(volk)
if(NOT volk_POPULATED)
    FetchContent_Populate(volk)
    # Spoof Vulkan as found so volk's CMakeLists never calls find_package(Vulkan)
    # and never links against vulkan-1.lib from the SDK.
    # We set all three variables volk checks before its find_package call.
    set(Vulkan_FOUND          TRUE  CACHE BOOL   "" FORCE)
    set(Vulkan_INCLUDE_DIRS   "${VULKAN_HEADERS_INCLUDE_DIR}" CACHE PATH "" FORCE)
    set(VOLK_VULKAN_HEADERS_PATH "${VULKAN_HEADERS_INCLUDE_DIR}" CACHE STRING "" FORCE)
    add_subdirectory(${volk_SOURCE_DIR} ${volk_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

# ── GLFW ──────────────────────────────────────────────────────────────────────
find_package(glfw3 3.3 QUIET)
if(NOT glfw3_FOUND)
    message(STATUS "vkgfx: fetching GLFW")
    FetchContent_Declare(glfw
        GIT_REPOSITORY https://github.com/glfw/glfw.git
        GIT_TAG        3.4
        GIT_SHALLOW    TRUE)
    set(GLFW_BUILD_DOCS     OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(GLFW_INSTALL        OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(glfw)
endif()

if(TARGET glfw3::glfw3)
    set(VKGFX_GLFW_TARGET glfw3::glfw3)
elseif(TARGET glfw3)
    set(VKGFX_GLFW_TARGET glfw3)
else()
    set(VKGFX_GLFW_TARGET glfw)
endif()

# ── GLM ───────────────────────────────────────────────────────────────────────
find_package(glm QUIET)
if(NOT glm_FOUND)
    message(STATUS "vkgfx: fetching GLM")
    FetchContent_Declare(glm
        GIT_REPOSITORY https://github.com/g-truc/glm.git
        GIT_TAG        1.0.1
        GIT_SHALLOW    TRUE)
    set(GLM_ENABLE_CXX_20 ON CACHE BOOL "" FORCE)
    FetchContent_GetProperties(glm)
    if(NOT glm_POPULATED)
        FetchContent_Populate(glm)
        add_subdirectory(${glm_SOURCE_DIR} ${glm_BINARY_DIR} EXCLUDE_FROM_ALL)
    endif()
endif()

if(TARGET glm::glm)
    set(VKGFX_GLM_TARGET glm::glm)
elseif(TARGET glm)
    set(VKGFX_GLM_TARGET glm)
else()
    set(VKGFX_GLM_TARGET "")
endif()

if(VKGFX_GLM_TARGET)
    get_target_property(_GLM_INC ${VKGFX_GLM_TARGET} INTERFACE_INCLUDE_DIRECTORIES)
    if(NOT _GLM_INC AND glm_SOURCE_DIR)
        set(_GLM_INC "${glm_SOURCE_DIR}")
    endif()
endif()

# ── VulkanMemoryAllocator ─────────────────────────────────────────────────────
message(STATUS "vkgfx: fetching VulkanMemoryAllocator")
FetchContent_Declare(vma
    GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
    GIT_TAG        v3.1.0
    GIT_SHALLOW    TRUE)
set(VMA_BUILD_SAMPLES         OFF CACHE BOOL "" FORCE)
set(VMA_BUILD_DOCUMENTATION   OFF CACHE BOOL "" FORCE)
FetchContent_GetProperties(vma)
if(NOT vma_POPULATED)
    FetchContent_Populate(vma)
    add_subdirectory(${vma_SOURCE_DIR} ${vma_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

# ── stb ───────────────────────────────────────────────────────────────────────
message(STATUS "vkgfx: fetching stb")
FetchContent_Declare(stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG        master
    GIT_SHALLOW    TRUE)
FetchContent_MakeAvailable(stb)

# ── nlohmann/json ─────────────────────────────────────────────────────────────
message(STATUS "vkgfx: fetching nlohmann/json")
FetchContent_Declare(json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
    GIT_SHALLOW    TRUE)
set(JSON_BuildTests   OFF CACHE INTERNAL "")
set(JSON_Install      OFF CACHE INTERNAL "")
FetchContent_MakeAvailable(json)

# ── tinyobjloader ─────────────────────────────────────────────────────────────
message(STATUS "vkgfx: fetching tinyobjloader")
FetchContent_Declare(tinyobjloader
    GIT_REPOSITORY https://github.com/tinyobjloader/tinyobjloader.git
    GIT_TAG        v2.0.0rc13
    GIT_SHALLOW    TRUE)
set(TINYOBJLOADER_BUILD_TEST_LOADER OFF CACHE BOOL "" FORCE)
FetchContent_GetProperties(tinyobjloader)
if(NOT tinyobjloader_POPULATED)
    FetchContent_Populate(tinyobjloader)
    add_subdirectory(${tinyobjloader_SOURCE_DIR} ${tinyobjloader_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

# ── fastgltf ──────────────────────────────────────────────────────────────────
message(STATUS "vkgfx: fetching fastgltf")
FetchContent_Declare(fastgltf
    GIT_REPOSITORY https://github.com/spnda/fastgltf.git
    GIT_TAG        v0.9.0
    GIT_SHALLOW    TRUE)
set(FASTGLTF_COMPILE_AS_CPP20 ON  CACHE BOOL "" FORCE)
set(FASTGLTF_ENABLE_TESTS     OFF CACHE BOOL "" FORCE)
set(FASTGLTF_ENABLE_EXAMPLES  OFF CACHE BOOL "" FORCE)
FetchContent_GetProperties(fastgltf)
if(NOT fastgltf_POPULATED)
    FetchContent_Populate(fastgltf)
    add_subdirectory(${fastgltf_SOURCE_DIR} ${fastgltf_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

# ── Dear ImGui (GPU profiler overlay) ─────────────────────────────────────────
# ON by default so examples show the profiler out of the box.
# Set -DVKGFX_ENABLE_PROFILING=OFF to build without ImGui for size-sensitive targets.
option(VKGFX_ENABLE_PROFILING "Enable GPU timestamp profiler + ImGui overlay" ON)

if(VKGFX_ENABLE_PROFILING)
    message(STATUS "vkgfx: fetching Dear ImGui (profiling overlay)")
    FetchContent_Declare(imgui
        GIT_REPOSITORY https://github.com/ocornut/imgui.git
        GIT_TAG        v1.91.9
        GIT_SHALLOW    TRUE)
    FetchContent_MakeAvailable(imgui)

    if(NOT TARGET imgui_vkgfx)
        add_library(imgui_vkgfx STATIC
            ${imgui_SOURCE_DIR}/imgui.cpp
            ${imgui_SOURCE_DIR}/imgui_draw.cpp
            ${imgui_SOURCE_DIR}/imgui_tables.cpp
            ${imgui_SOURCE_DIR}/imgui_widgets.cpp
            ${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp
            ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp)
        target_include_directories(imgui_vkgfx
            PUBLIC
                ${imgui_SOURCE_DIR}
                ${imgui_SOURCE_DIR}/backends)
        # volk_headers gives imgui_impl_vulkan.cpp the volk function-pointer
        # declarations (via volk.h) so it can call Vulkan without vulkan-1.lib.
        # VK_NO_PROTOTYPES must match the rest of the build.
        target_link_libraries(imgui_vkgfx
            PUBLIC volk_headers ${VKGFX_GLFW_TARGET})
        target_compile_definitions(imgui_vkgfx
            PUBLIC VK_NO_PROTOTYPES IMGUI_IMPL_VULKAN_USE_VOLK)
        target_include_directories(imgui_vkgfx
            PUBLIC
                ${imgui_SOURCE_DIR}
                ${imgui_SOURCE_DIR}/backends
                ${volk_SOURCE_DIR}
                ${VULKAN_HEADERS_INCLUDE_DIR})
        target_compile_options(imgui_vkgfx
            PRIVATE $<IF:$<CXX_COMPILER_ID:MSVC>,/W0,-w>)
    endif()
endif()

# ── Shader compilation ─────────────────────────────────────────────────────────
# glslc is a BUILD-TIME tool (developer only). End-users of the release package
# receive pre-compiled .spv files and never need glslc or the Vulkan SDK.
find_program(GLSLC glslc
    HINTS
        "$ENV{VULKAN_SDK}/bin"
        "$ENV{VULKAN_SDK}/Bin"
        "$ENV{PATH}")

if(NOT GLSLC)
    message(FATAL_ERROR
        "glslc not found. Install the Vulkan SDK (https://vulkan.lunarg.com/) "
        "or ensure glslc is on your PATH. "
        "Note: this is only required when BUILDING vkgfx — end-users of the "
        "pre-built release package do not need it.")
endif()

file(GLOB_RECURSE GLSL_SOURCES
    "${CMAKE_SOURCE_DIR}/shaders/*.vert"
    "${CMAKE_SOURCE_DIR}/shaders/*.frag"
    "${CMAKE_SOURCE_DIR}/shaders/*.comp")

set(SPV_DIR "${CMAKE_BINARY_DIR}/shaders")
file(MAKE_DIRECTORY "${SPV_DIR}")

set(SPV_OUTPUTS "")
foreach(GLSL ${GLSL_SOURCES})
    get_filename_component(GLSL_NAME "${GLSL}" NAME)
    set(SPV "${SPV_DIR}/${GLSL_NAME}.spv")
    add_custom_command(
        OUTPUT  "${SPV}"
        COMMAND "${GLSLC}" -o "${SPV}" "${GLSL}"
        DEPENDS "${GLSL}"
        COMMENT "Compiling shader ${GLSL_NAME}"
        VERBATIM)
    list(APPEND SPV_OUTPUTS "${SPV}")
endforeach()

add_custom_target(vkgfx_shaders ALL DEPENDS ${SPV_OUTPUTS})

# ── Library ────────────────────────────────────────────────────────────────────
file(GLOB_RECURSE VKGFX_SOURCES src/*.cpp)

# Embed volk's object code directly into vkgfx.lib via $<TARGET_OBJECTS:volk>.
# On Windows, MSVC static libs do NOT embed their static-lib dependencies --
# the consumer would have to link volk.lib separately. Pulling the object
# files in at library-creation time makes vkgfx.lib truly self-contained:
# users link ONE library and get all Vulkan dispatch for free.
# On Windows, MSVC static libs do NOT absorb their static-lib dependencies.
# Use $<TARGET_OBJECTS:tgt> to pull every dependency's .obj files directly
# into vkgfx.lib so the consumer links ONE file and everything resolves.
# $<TARGET_OBJECTS:> works on STATIC targets in CMake 3.21+ (we require 3.24).
set(VKGFX_DEP_OBJECTS $<TARGET_OBJECTS:volk>)

# GLFW: actual FetchContent target is "glfw", not alias glfw3::glfw3
foreach(_t IN ITEMS glfw glfw3)
    if(TARGET ${_t})
        list(APPEND VKGFX_DEP_OBJECTS $<TARGET_OBJECTS:${_t}>)
        message(STATUS "vkgfx: embedding ${_t} objects")
        break()
    endif()
endforeach()

# GLM: has a small compiled component
foreach(_t IN ITEMS glm glm_static)
    if(TARGET ${_t})
        list(APPEND VKGFX_DEP_OBJECTS $<TARGET_OBJECTS:${_t}>)
        message(STATUS "vkgfx: embedding ${_t} objects")
        break()
    endif()
endforeach()

# fastgltf: also absorbs simdjson.cpp compiled directly into it
if(TARGET fastgltf)
    list(APPEND VKGFX_DEP_OBJECTS $<TARGET_OBJECTS:fastgltf>)
    message(STATUS "vkgfx: embedding fastgltf objects")
endif()

# tinyobjloader
foreach(_t IN ITEMS tinyobjloader tinyobjloader_rosetta)
    if(TARGET ${_t})
        list(APPEND VKGFX_DEP_OBJECTS $<TARGET_OBJECTS:${_t}>)
        message(STATUS "vkgfx: embedding ${_t} objects")
        break()
    endif()
endforeach()

# imgui_vkgfx (profiling overlay, defined earlier in this file)
if(VKGFX_ENABLE_PROFILING AND TARGET imgui_vkgfx)
    list(APPEND VKGFX_DEP_OBJECTS $<TARGET_OBJECTS:imgui_vkgfx>)
    message(STATUS "vkgfx: embedding imgui_vkgfx objects")
endif()

add_library(vkgfx STATIC ${VKGFX_SOURCES} ${VKGFX_DEP_OBJECTS})
add_library(vkgfx::vkgfx ALIAS vkgfx)

target_include_directories(vkgfx
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)

# GLM include dirs
if(_GLM_INC)
    target_include_directories(vkgfx PUBLIC $<BUILD_INTERFACE:${_GLM_INC}>)
endif()

# VMA include dir
if(TARGET GPUOpen::VulkanMemoryAllocator)
    target_include_directories(vkgfx PRIVATE
        $<BUILD_INTERFACE:$<TARGET_PROPERTY:GPUOpen::VulkanMemoryAllocator,INTERFACE_INCLUDE_DIRECTORIES>>)
elseif(vma_SOURCE_DIR)
    target_include_directories(vkgfx PRIVATE $<BUILD_INTERFACE:${vma_SOURCE_DIR}/include>)
endif()

# stb include dir
if(stb_SOURCE_DIR)
    target_include_directories(vkgfx PRIVATE $<BUILD_INTERFACE:${stb_SOURCE_DIR}>)
endif()

# nlohmann/json — header-only library, include dirs only (never link_libraries).
# Linking a FetchContent target that has no install rule causes:
#   "install(EXPORT) includes target vkgfx which requires target nlohmann_json
#    that is not in any export set."
if(TARGET nlohmann_json::nlohmann_json)
    get_target_property(_JSON_INC nlohmann_json::nlohmann_json INTERFACE_INCLUDE_DIRECTORIES)
    if(_JSON_INC)
        target_include_directories(vkgfx PRIVATE $<BUILD_INTERFACE:${_JSON_INC}>)
    endif()
elseif(json_SOURCE_DIR)
    target_include_directories(vkgfx PRIVATE $<BUILD_INTERFACE:${json_SOURCE_DIR}/include>)
endif()

# tinyobjloader
if(tinyobjloader_SOURCE_DIR)
    target_include_directories(vkgfx PRIVATE $<BUILD_INTERFACE:${tinyobjloader_SOURCE_DIR}>)
endif()

# fastgltf — compiled FetchContent target with no install rule.
# BUILD_INTERFACE suppresses it from the install export set, preventing:
#   "install(EXPORT) includes target vkgfx which requires target fastgltf
#    that is not in any export set."
target_link_libraries(vkgfx PRIVATE $<BUILD_INTERFACE:fastgltf>)

# volk objects are already embedded via $<TARGET_OBJECTS:volk> in add_library.
# Here we only need to propagate the include paths and the VK_NO_PROTOTYPES
# define so that consumers of vkgfx see the volk/vulkan headers correctly.
# INSTALL_INTERFACE paths point to where build_release.bat copies the headers.
target_include_directories(vkgfx
    PUBLIC
        $<BUILD_INTERFACE:${VULKAN_HEADERS_INCLUDE_DIR}>
        $<BUILD_INTERFACE:${volk_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>)
target_compile_definitions(vkgfx PUBLIC VK_NO_PROTOTYPES)

target_link_libraries(vkgfx
    PUBLIC
        $<BUILD_INTERFACE:${VKGFX_GLFW_TARGET}>
    PRIVATE
        $<BUILD_INTERFACE:${VKGFX_GLM_TARGET}>)

target_compile_definitions(vkgfx
    PUBLIC
        GLM_FORCE_RADIANS
        GLM_FORCE_DEPTH_ZERO_TO_ONE
        GLM_ENABLE_EXPERIMENTAL
    PRIVATE
        $<$<AND:$<CONFIG:Debug>,$<BOOL:${VKGFX_ENABLE_VALIDATION}>>:VKGFX_ENABLE_VALIDATION>)

# Profiling: define the macro and link ImGui when the option is enabled.
# imgui_vkgfx is scoped to BUILD_INTERFACE so it doesn't leak into the
# install export set (downstream targets use their own ImGui).
if(VKGFX_ENABLE_PROFILING)
    target_compile_definitions(vkgfx PUBLIC VKGFX_ENABLE_PROFILING)
    target_link_libraries(vkgfx PRIVATE $<BUILD_INTERFACE:imgui_vkgfx>)
    target_include_directories(vkgfx
        PUBLIC
            $<BUILD_INTERFACE:${imgui_SOURCE_DIR}>
            $<BUILD_INTERFACE:${imgui_SOURCE_DIR}/backends>)
endif()

if(MSVC)
    target_compile_options(vkgfx PRIVATE /W3 /wd4201 /wd4100 /wd4996 /wd4244 /wd4267)
    target_compile_options(vkgfx PUBLIC  /Zc:__cplusplus)
    # /FI (Force Include) injects volk.h as the implicit first #include in every
    # .cpp translation unit. With VK_NO_PROTOTYPES defined, vulkan.h omits the
    # extern function declarations; volk.h re-declares them as function-pointer
    # globals so all vkXxx calls compile without touching a single source file.
    target_compile_options(vkgfx PRIVATE "/FI${volk_SOURCE_DIR}/volk.h")
else()
    # GCC / Clang equivalent of /FI
    target_compile_options(vkgfx PRIVATE "-include${volk_SOURCE_DIR}/volk.h")
endif()

add_dependencies(vkgfx vkgfx_shaders)

add_custom_command(TARGET vkgfx POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
        "$<TARGET_FILE_DIR:vkgfx>/shaders"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${SPV_DIR}"
        "$<TARGET_FILE_DIR:vkgfx>/shaders"
    COMMENT "Copying SPV shaders to $<TARGET_FILE_DIR:vkgfx>/shaders"
    VERBATIM
)

# ── Example ────────────────────────────────────────────────────────────────────
if(VKGFX_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# ── Tests ──────────────────────────────────────────────────────────────────────
if(VKGFX_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ── Install ────────────────────────────────────────────────────────────────────
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS vkgfx
    EXPORT  vkgfxTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

install(DIRECTORY include/vkgfx DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Bundle Vulkan headers and volk.h so users need NO Vulkan SDK installed.
# The release package becomes self-contained: include/ has everything.
install(DIRECTORY "${VULKAN_HEADERS_INCLUDE_DIR}/vulkan"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES "${volk_SOURCE_DIR}/volk.h" "${volk_SOURCE_DIR}/volk.c"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/volk)

install(DIRECTORY "${SPV_DIR}/"
    DESTINATION ${CMAKE_INSTALL_DATADIR}/vkgfx/shaders
    FILES_MATCHING PATTERN "*.spv")

install(EXPORT vkgfxTargets
    FILE      vkgfxTargets.cmake
    NAMESPACE vkgfx::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/vkgfx)

configure_package_config_file(
    "${CMAKE_SOURCE_DIR}/cmake/vkgfxConfig.cmake.in"
    "${CMAKE_BINARY_DIR}/vkgfxConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/vkgfx)

write_basic_package_version_file(
    "${CMAKE_BINARY_DIR}/vkgfxConfigVersion.cmake"
    VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)

install(FILES
    "${CMAKE_BINARY_DIR}/vkgfxConfig.cmake"
    "${CMAKE_BINARY_DIR}/vkgfxConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/vkgfx)
