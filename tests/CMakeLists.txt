include(FetchContent)
FetchContent_Declare(Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.5.3
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(Catch2)

add_executable(vkgfx_tests
    test_render_pass.cpp
    test_ibl_load.cpp
    test_vertex_layout.cpp
    test_headless_smoke.cpp
)

target_link_libraries(vkgfx_tests
    PRIVATE
        vkgfx::vkgfx
        ${VKGFX_GLFW_TARGET}
        Catch2::Catch2WithMain
)

if(MSVC)
    target_compile_options(vkgfx_tests PRIVATE /W3 /wd4201 /wd4100)
    target_compile_options(vkgfx_tests PUBLIC  /Zc:__cplusplus)
endif()

# Copy compiled shaders next to the test binary.
# Use make_directory + copy_directory so missing source dirs don't hard-fail.
add_custom_command(TARGET vkgfx_tests POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
        "$<TARGET_FILE_DIR:vkgfx_tests>/shaders"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_BINARY_DIR}/shaders"
        "$<TARGET_FILE_DIR:vkgfx_tests>/shaders"
    COMMENT "Copying shaders to test output dir"
    VERBATIM
)

# Copy assets only if the directory exists (sky.hdr is optional for most tests)
if(EXISTS "${CMAKE_SOURCE_DIR}/assets")
    add_custom_command(TARGET vkgfx_tests POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_SOURCE_DIR}/assets"
            "$<TARGET_FILE_DIR:vkgfx_tests>/assets"
        COMMENT "Copying assets to test output dir"
        VERBATIM
    )
endif()

include(Catch)
catch_discover_tests(vkgfx_tests)
