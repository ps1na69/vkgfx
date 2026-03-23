# examples/CMakeLists.txt
# Builds the full demo and all focused example applications.
# Each example links against vkgfx::vkgfx — no engine source duplication.

# ── Helper macro ─────────────────────────────────────────────────────────────
# vkgfx_add_example(name)
#   Creates an executable target named 'name', compiles main.cpp in the
#   current source dir, links the engine, copies shaders and assets.
macro(vkgfx_add_example name)
    add_executable(${name} main.cpp)

    target_link_libraries(${name}
        PRIVATE
            vkgfx::vkgfx
            ${VKGFX_GLFW_TARGET})

    if(MSVC)
        target_compile_options(${name} PRIVATE /W3 /wd4201 /wd4100)
        target_compile_options(${name} PUBLIC  /Zc:__cplusplus)
    endif()

    # Copy compiled shaders next to the executable
    add_custom_command(TARGET ${name} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory
            "$<TARGET_FILE_DIR:${name}>/shaders"
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_BINARY_DIR}/shaders"
            "$<TARGET_FILE_DIR:${name}>/shaders"
        COMMENT "Copying shaders → ${name}"
        VERBATIM)

    # Copy assets (optional — sky.hdr lives here)
    if(EXISTS "${CMAKE_SOURCE_DIR}/assets")
        add_custom_command(TARGET ${name} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory
                "${CMAKE_SOURCE_DIR}/assets"
                "$<TARGET_FILE_DIR:${name}>/assets"
            COMMENT "Copying assets → ${name}"
            VERBATIM)
    endif()
endmacro()

# ── Full demo (existing main.cpp) ─────────────────────────────────────────────
vkgfx_add_example(vkgfx_example)

install(TARGETS vkgfx_example DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY "${CMAKE_SOURCE_DIR}/assets"
    DESTINATION ${CMAKE_INSTALL_DATADIR}/vkgfx
    OPTIONAL)

# ── Focused examples ──────────────────────────────────────────────────────────
add_subdirectory(01_window)
add_subdirectory(02_triangle)
add_subdirectory(03_model_loading)
add_subdirectory(04_camera)
add_subdirectory(05_lighting)
add_subdirectory(06_textures)
add_subdirectory(07_input)
add_subdirectory(08_collision)
add_subdirectory(09_scene)
