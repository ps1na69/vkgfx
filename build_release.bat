@echo off
REM build_release.bat — Build vkgfx release package for Windows
REM Usage: build_release.bat [release|debug]

setlocal enabledelayedexpansion

set PLATFORM=windows
set CONFIG=%1
if "%CONFIG%"=="" set CONFIG=release

if /i "%CONFIG%"=="debug" (
    set BUILD_TYPE=Debug
) else (
    set BUILD_TYPE=Release
)

set OS_NAME=windows
set ARCH=x86_64
set RELEASE_NAME=vkgfx-%OS_NAME%-%ARCH%-%CONFIG%
set RELEASE_DIR=release\%RELEASE_NAME%
set BUILD_DIR=build_%PLATFORM%_%CONFIG%

echo ============================================
echo Building vkgfx release package
echo Platform: %OS_NAME% (%ARCH%)
echo Config:   %BUILD_TYPE%
echo Output:   %RELEASE_DIR%
echo ============================================

REM Clean previous build
if exist "%BUILD_DIR%" rmdir /s /q "%BUILD_DIR%"
if exist "%RELEASE_DIR%" rmdir /s /q "%RELEASE_DIR%"
mkdir "%BUILD_DIR%"
mkdir "%RELEASE_DIR%"

cd "%BUILD_DIR%"

REM Configure with CMake
echo [1/3] Configuring with CMake...
cmake .. ^
    -G "Visual Studio 17 2022" ^
    -A x64 ^
    -DCMAKE_BUILD_TYPE="%BUILD_TYPE%" ^
    -DVKGFX_BUILD_EXAMPLES=OFF ^
    -DVKGFX_BUILD_TESTS=OFF ^
    -DVKGFX_ENABLE_VALIDATION=OFF ^
    -DCMAKE_INSTALL_PREFIX="../%RELEASE_DIR%"
if errorlevel 1 (
    echo CMake configuration failed!
    cd ..
    exit /b 1
)

REM Build
echo [2/3] Building library...
cmake --build . --config "%BUILD_TYPE%" --target vkgfx
if errorlevel 1 (
    echo Build failed!
    cd ..
    exit /b 1
)

REM Install
echo [3/3] Installing to release directory...
cmake --install . --config "%BUILD_TYPE%"
if errorlevel 1 (
    echo Install failed!
    cd ..
    exit /b 1
)

cd ..

REM Create additional directories structure
mkdir "%RELEASE_DIR%\bin\shaders"
mkdir "%RELEASE_DIR%\examples"
mkdir "%RELEASE_DIR%\docs"

REM Copy compiled shaders
if exist "%BUILD_DIR%\shaders\*.spv" (
    copy "%BUILD_DIR%\shaders\*.spv" "%RELEASE_DIR%\bin\shaders\" >nul
)

REM Copy example source files
if exist "examples\*.cpp" (
    copy "examples\*.cpp" "%RELEASE_DIR%\examples\" >nul
)
if exist "examples\CMakeLists.txt" (
    copy "examples\CMakeLists.txt" "%RELEASE_DIR%\examples\" >nul
)

REM Create simple example CMakeLists for users
(
echo cmake_minimum_required^(VERSION 3.16^)
echo project^(my_vkgfx_app^)
echo.
echo set^(CMAKE_CXX_STANDARD 20^)
echo set^(CMAKE_CXX_STANDARD_REQUIRED ON^)
echo.
echo # Find Vulkan
echo find_package^(Vulkan REQUIRED^)
echo.
echo # Add your executable
echo add_executable^(my_app main.cpp^)
echo.
echo # Link vkgfx library
echo target_link_libraries^(my_app PRIVATE^)
echo     vkgfx::vkgfx^)
echo     Vulkan::Vulkan^)
echo.
echo # Include directories
echo target_include_directories^(my_app PRIVATE^)
echo     ${CMAKE_CURRENT_SOURCE_DIR}/../include^)
echo.
echo # Copy shaders to output directory
echo add_custom_command^(TARGET my_app POST_BUILD^)
echo     COMMAND ${CMAKE_COMMAND} -E copy_directory^)
echo         "${CMAKE_CURRENT_SOURCE_DIR}/../bin/shaders"^)
echo         "$^<TARGET_FILE_DIR:my_app^>/shaders"^)
echo ^)
) > "%RELEASE_DIR%\examples\CMakeLists.txt"

REM Create README for users
echo Creating README.md...
(
echo # vkgfx - Simple Vulkan Graphics Engine
echo.
echo Pre-built binary release for **%OS_NAME%** (**%ARCH%**, **%CONFIG**^)
echo.
echo ## Quick Start
echo.
echo ### Prerequisites
echo.
echo Before using vkgfx, ensure you have:
echo.
echo 1. **Vulkan SDK** installed
echo    - Download from https://vulkan.lunarg.com/
echo.
echo 2. **C++20 compatible compiler**
echo    - Visual Studio 2019+ or Clang 12+
echo.
echo ### Directory Structure
echo.
echo ```
echo %RELEASE_NAME%\
echo ├── include/          # Public header files
echo │   └── vkgfx/
echo │       ├── vkgfx.h   # Master include ^(include this one^)
echo │   └── *.h       # Individual headers
echo ├── lib/              # Compiled library files
echo │   └── vkgfx.lib     # Windows static library
echo ├── bin/              # Runtime binaries
echo │   └── shaders/      # Compiled SPIR-V shaders ^(*.spv^)
echo ├── examples/         # Example source code
echo └── README.md         # This file
echo ```
echo.
echo ### Using in Your Project
echo.
echo #### Option 1: CMake ^(Recommended^)
echo.
echo 1. Copy the entire release folder to your project
echo 2. In your CMakeLists.txt:
echo.
echo ```cmake
echo cmake_minimum_required^(VERSION 3.16^)
echo project^(my_app^)
echo.
echo set^(CMAKE_CXX_STANDARD 20^)
echo.
echo # Point to vkgfx location
echo set^(VKGFX_ROOT "${CMAKE_SOURCE_DIR}/thirdparty/%RELEASE_NAME%^")
echo.
echo # Add vkgfx library
echo add_library^(vkgfx STATIC IMPORTED^)
echo set_target_properties^(vkgfx PROPERTIES^)
echo     IMPORTED_LOCATION "${VKGFX_ROOT}/lib/vkgfx.lib"^)
echo     INTERFACE_INCLUDE_DIRECTORIES "${VKGFX_ROOT}/include"^)
echo ^)
echo.
echo # Find Vulkan
echo find_package^(Vulkan REQUIRED^)
echo.
echo # Your executable
echo add_executable^(my_app src/main.cpp^)
echo target_link_libraries^(my_app PRIVATE vkgfx Vulkan::Vulkan^)
echo ```
echo.
echo #### Option 2: Visual Studio Configuration
echo.
echo 1. Right-click project → Properties
echo 2. C/C++ → General → Additional Include Directories:
echo    - Add: `path\\to\\vkgfx\\include`
echo 3. Linker → Input → Additional Dependencies:
echo    - Add: `path\\to\\vkgfx\\lib\\vkgfx.lib`
echo    - Add: `vulkan-1.lib`
echo 4. Copy `bin\\shaders` folder to your executable directory
echo.
echo ### Minimal Example
echo.
echo ```cpp
echo #include ^<vkgfx/vkgfx.h^>
echo #include ^<iostream^>
echo.
echo int main^(^) {
echo     // Create window and context
echo     vkgfx::WindowConfig winCfg{};
echo     winCfg.width = 800;
echo     winCfg.height = 600;
echo     winCfg.title = "My App";
echo.
echo     vkgfx::Window window^(winCfg^);
echo.
echo     vkgfx::ContextConfig ctxCfg{};
echo     ctxCfg.appName = "MyApp";
echo     ctxCfg.validation = true;  // Enable validation layers in debug
echo.
echo     vkgfx::Context ctx^(ctxCfg, window.surface^(^)^);
echo.
echo     std::cout ^<^< "vkgfx initialized successfully!" ^<^< std::endl;
echo     return 0;
echo }
echo ```
echo.
echo ### Important Notes
echo.
echo 1. **Shaders**: The `bin\\shaders\\` folder must be copied to your executable's working directory
echo 2. **Validation Layers**: This release build has validation layers **DISABLED** for better performance and smaller size. For development, build from source with `VKGFX_ENABLE_VALIDATION=ON`.
echo 3. **GLFW**: vkgfx uses GLFW internally - it's statically linked in this release
echo 4. **Other dependencies**: GLM, stb, tinyobjloader, fastgltf are all statically linked
echo.
echo ### Troubleshooting
echo.
echo **Error: Cannot find Vulkan**
echo - Install Vulkan SDK from https://vulkan.lunarg.com/
echo - Ensure `VULKAN_SDK` environment variable is set
echo.
echo **Error: Missing shaders**
echo - Copy `bin\\shaders\\` folder to your executable directory
echo - Or set working directory to include shaders path
echo.
echo **Error: Validation layer not found**
echo - Install Vulkan SDK with validation layers
echo - Set `ctxCfg.validation = false` to disable ^(not recommended for development^)
echo.
echo ### API Reference
echo.
echo For full API documentation, see headers in `include/vkgfx/`:
echo.
echo - `vkgfx.h` - Master include, includes all public APIs
echo - `window.h` - Window creation and management
echo - `context.h` - Vulkan context and device
echo - `renderer.h` - Main rendering interface
echo - `scene.h` - Scene graph and entities
echo - `mesh.h` - 3D model loading
echo - `texture.h` - Texture loading and management
echo - `camera.h` - Camera utilities
echo - `collision.h` - Collision detection
echo - `config.h` - Configuration and settings
echo.
echo ### Support
echo.
echo GitHub: https://github.com/[your-username]/vkgfx
echo Issues: https://github.com/[your-username]/vkgfx/issues
) > "%RELEASE_DIR%\README.md"

REM Create ZIP archive
echo.
echo Creating ZIP archive...
if exist "%RELEASE_NAME%.zip" del "%RELEASE_NAME%.zip"
powershell -Command "Compress-Archive -Path '%RELEASE_NAME%' -DestinationPath '%RELEASE_NAME%.zip' -Force"

echo.
echo ============================================
echo Release package created successfully!
echo.
echo Archive: release\%RELEASE_NAME%.zip
echo Folder:  %RELEASE_DIR%\
echo.
echo To use:
echo   1. Extract the ZIP to your project
echo   2. Follow instructions in README.md
echo ============================================

endlocal
