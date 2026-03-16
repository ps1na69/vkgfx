#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "vkgfx::vkgfx" for configuration "Debug"
set_property(TARGET vkgfx::vkgfx APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(vkgfx::vkgfx PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "CXX"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/vkgfx.lib"
  )

list(APPEND _cmake_import_check_targets vkgfx::vkgfx )
list(APPEND _cmake_import_check_files_for_vkgfx::vkgfx "${_IMPORT_PREFIX}/lib/vkgfx.lib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
