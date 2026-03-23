vkgfx_add_example(ex_06_textures)

# Copy assets/textures/ next to the executable so the auto-scan finds textures.
# This command is in the same directory as the target — required by CMake.
if(EXISTS "${CMAKE_SOURCE_DIR}/assets/textures")
    add_custom_command(TARGET ex_06_textures POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory
            "$<TARGET_FILE_DIR:ex_06_textures>/assets/textures"
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_SOURCE_DIR}/assets/textures"
            "$<TARGET_FILE_DIR:ex_06_textures>/assets/textures"
        COMMENT "Copying assets/textures → ex_06_textures"
        VERBATIM)
endif()
