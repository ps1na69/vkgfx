vkgfx_add_example(ex_03_model_loading)

# Always create assets/models/ and copy its contents next to the executable.
# The make_directory step ensures this never fails even on first run.
add_custom_command(TARGET ex_03_model_loading POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
        "${CMAKE_SOURCE_DIR}/assets/models"
    COMMAND ${CMAKE_COMMAND} -E make_directory
        "$<TARGET_FILE_DIR:ex_03_model_loading>/assets/models"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/assets/models"
        "$<TARGET_FILE_DIR:ex_03_model_loading>/assets/models"
    COMMENT "Copying assets/models → ex_03_model_loading"
    VERBATIM)
