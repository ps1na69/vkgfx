cmake_minimum_required(VERSION 3.24)
project(vkgfx VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(VKGFX_BUILD_EXAMPLES    "Build example applications"               ON)
option(VKGFX_BUILD_TESTS       "Build unit tests"                         ON)
option(VKGFX_ENABLE_VALIDATION "Enable Vulkan validation layers in Debug"  ON)

# ── Vulkan ─────────────────────────────────────────────────────────────────────
find_package(Vulkan REQUIRED)

include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

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
    FetchContent_MakeAvailable(glm)
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
FetchContent_MakeAvailable(vma)

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
FetchContent_MakeAvailable(tinyobjloader)

# ── fastgltf ──────────────────────────────────────────────────────────────────
message(STATUS "vkgfx: fetching fastgltf")
FetchContent_Declare(fastgltf
    GIT_REPOSITORY https://github.com/spnda/fastgltf.git
    GIT_TAG        v0.7.2
    GIT_SHALLOW    TRUE)
set(FASTGLTF_COMPILE_AS_CPP20 ON  CACHE BOOL "" FORCE)
set(FASTGLTF_ENABLE_TESTS     OFF CACHE BOOL "" FORCE)
set(FASTGLTF_ENABLE_EXAMPLES  OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(fastgltf)

# ── Shader compilation ─────────────────────────────────────────────────────────
if(Vulkan_GLSLC_EXECUTABLE)
    set(GLSLC "${Vulkan_GLSLC_EXECUTABLE}")
else()
    find_program(GLSLC glslc
        HINTS "$ENV{VULKAN_SDK}/bin" "$ENV{VULKAN_SDK}/Bin"
        REQUIRED)
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

add_library(vkgfx STATIC ${VKGFX_SOURCES})
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

# nlohmann/json
if(TARGET nlohmann_json::nlohmann_json)
    target_link_libraries(vkgfx PRIVATE $<BUILD_INTERFACE:nlohmann_json::nlohmann_json>)
elseif(json_SOURCE_DIR)
    target_include_directories(vkgfx PRIVATE $<BUILD_INTERFACE:${json_SOURCE_DIR}/include>)
endif()

# tinyobjloader
if(tinyobjloader_SOURCE_DIR)
    target_include_directories(vkgfx PRIVATE $<BUILD_INTERFACE:${tinyobjloader_SOURCE_DIR}>)
endif()

# fastgltf — link to the imported target; it brings its own include dirs
target_link_libraries(vkgfx PRIVATE $<BUILD_INTERFACE:fastgltf>)

target_link_libraries(vkgfx
    PUBLIC
        Vulkan::Vulkan
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

if(MSVC)
    target_compile_options(vkgfx PRIVATE /W3 /wd4201 /wd4100 /wd4996 /wd4244 /wd4267)
    target_compile_options(vkgfx PUBLIC  /Zc:__cplusplus)
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
