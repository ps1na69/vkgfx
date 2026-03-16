add_executable(vkgfx_example main.cpp)

target_link_libraries(vkgfx_example
    PRIVATE
        vkgfx::vkgfx
        ${VKGFX_GLFW_TARGET}
)

if(MSVC)
    target_compile_options(vkgfx_example PRIVATE /W3 /wd4201 /wd4100)
    target_compile_options(vkgfx_example PUBLIC  /Zc:__cplusplus)
endif()

# Copy shaders next to the executable.
add_custom_command(TARGET vkgfx_example POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
        "$<TARGET_FILE_DIR:vkgfx_example>/shaders"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_BINARY_DIR}/shaders"
        "$<TARGET_FILE_DIR:vkgfx_example>/shaders"
    COMMENT "Copying shaders to example output dir"
    VERBATIM
)

# Copy assets only if the directory exists.
# Place sky.hdr under assets/ before running to enable IBL.
if(EXISTS "${CMAKE_SOURCE_DIR}/assets")
    add_custom_command(TARGET vkgfx_example POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_SOURCE_DIR}/assets"
            "$<TARGET_FILE_DIR:vkgfx_example>/assets"
        COMMENT "Copying assets to example output dir"
        VERBATIM
    )
endif()

install(TARGETS vkgfx_example DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY "${CMAKE_SOURCE_DIR}/assets"
    DESTINATION ${CMAKE_INSTALL_DATADIR}/vkgfx
    OPTIONAL   # don't fail install if assets/ is empty
)
