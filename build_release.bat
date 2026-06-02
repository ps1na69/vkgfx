@echo off
REM build_release.bat - Build vkgfx release package for Windows
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
set LOG_FILE=build_%CONFIG%.log

echo ============================================
echo  Building vkgfx release package
echo  Platform : %OS_NAME% (%ARCH%)
echo  Config   : %BUILD_TYPE%
echo  Output   : %RELEASE_DIR%
echo  Log      : %LOG_FILE%
echo ============================================
echo.

REM ── Sanity checks ────────────────────────────────────────────────────────────

REM Check cmake is on PATH
where cmake >nul 2>&1
if errorlevel 1 (
    echo [ERROR] cmake not found on PATH.
    echo.
    echo   Common fixes:
    echo     1. Install CMake from https://cmake.org/ and tick "Add to PATH"
    echo     2. Or add Visual Studio's cmake manually:
    echo        C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin
    echo.
    goto :error
)

REM Check Visual Studio 2022 generator is available; fall back to 2019 silently
cmake --help | findstr /C:"Visual Studio 17 2022" >nul 2>&1
if errorlevel 1 (
    echo [WARN] Visual Studio 17 2022 generator not found, trying Visual Studio 16 2019...
    set VS_GENERATOR=Visual Studio 16 2019
) else (
    set VS_GENERATOR=Visual Studio 17 2022
)

REM ── Clean previous build ─────────────────────────────────────────────────────

if exist "%BUILD_DIR%" (
    echo Cleaning previous build directory...
    rmdir /s /q "%BUILD_DIR%"
)
if exist "%RELEASE_DIR%" (
    echo Cleaning previous release directory...
    rmdir /s /q "%RELEASE_DIR%"
)
mkdir "%BUILD_DIR%"
mkdir "%RELEASE_DIR%"
if exist "%LOG_FILE%" del "%LOG_FILE%"

REM Resolve absolute path BEFORE changing directory.
REM CMAKE_INSTALL_PREFIX must be absolute - with the Visual Studio multi-config
REM generator a relative path resolves against the source dir, not the build dir.
set ABS_RELEASE_DIR=%CD%\%RELEASE_DIR%
set ABS_BUILD_DIR=%CD%\%BUILD_DIR%
set ABS_LOG=%CD%\%LOG_FILE%

echo Build log: %ABS_LOG%
echo.

cd "%BUILD_DIR%"

REM ── [1/3] Configure ──────────────────────────────────────────────────────────

echo [1/3] Configuring with CMake (generator: %VS_GENERATOR%)...
cmake .. ^
    -G "%VS_GENERATOR%" ^
    -A x64 ^
    -DCMAKE_BUILD_TYPE="%BUILD_TYPE%" ^
    -DVKGFX_BUILD_EXAMPLES=OFF ^
    -DVKGFX_BUILD_TESTS=OFF ^
    -DVKGFX_ENABLE_VALIDATION=OFF ^
    -DCMAKE_INSTALL_PREFIX="%ABS_RELEASE_DIR%" ^
    >> "%ABS_LOG%" 2>&1

if errorlevel 1 (
    echo [ERROR] CMake configuration failed!
    echo.
    echo Last 30 lines of log:
    powershell -Command "Get-Content '%ABS_LOG%' -Tail 30"
    echo.
    echo Full log: %ABS_LOG%
    cd ..
    goto :error
)
echo   Configure OK.

REM ── [2/3] Build ──────────────────────────────────────────────────────────────

REM Build and install BOTH Release and Debug so users can match their project CRT.
REM  lib\Release\vkgfx.lib  -- use when your project Config = Release
REM  lib\Debug\vkgfx.lib    -- use when your project Config = Debug

for %%C in (Release Debug) do (
    echo [2-3/3] Building and installing %%C...
    cmake --build . --config %%C --target vkgfx -- /m >> "%ABS_LOG%" 2>>&1
    if errorlevel 1 (
        echo [ERROR] %%C build failed!
        powershell -Command "Get-Content '%ABS_LOG%' -Tail 50"
        cd ..
        goto :error
    )
    cmake --install . --config %%C --prefix "%ABS_RELEASE_DIR%" >> "%ABS_LOG%" 2>>&1
    if errorlevel 1 (
        echo [ERROR] %%C install failed!
        powershell -Command "Get-Content '%ABS_LOG%' -Tail 30"
        cd ..
        goto :error
    )
    echo   %%C OK.
)

cd ..

REM ── Post-install: fix up directory layout ───────────────────────────────────

REM cmake installs shaders to share\vkgfx\shaders\ (GNUInstallDirs on Windows).
REM Move them to bin\shaders\ which is what the README and users expect.
mkdir "%RELEASE_DIR%\bin\shaders" 2>nul
if exist "%RELEASE_DIR%\share\vkgfx\shaders" (
    echo Relocating shaders from share\vkgfx\shaders to bin\shaders...
    xcopy /e /y "%RELEASE_DIR%\share\vkgfx\shaders\*" "%RELEASE_DIR%\bin\shaders\" >nul
    rmdir /s /q "%RELEASE_DIR%\share"
)

REM Also copy directly from build dir in case install missed them
if exist "%BUILD_DIR%\shaders" (
    xcopy /e /y "%BUILD_DIR%\shaders\*.spv" "%RELEASE_DIR%\bin\shaders\" >nul 2>&1
)

REM ── Generate vkgfx_config.h and patch vkgfx.h ─────────────────
REM
REM Create a vkgfx_config.h that pre-defines every macro the library needs.
REM Then prepend an #include of it into the release copy of vkgfx.h so ALL
REM users (CMake AND manual Visual Studio) get the defines automatically.
REM

echo Generating vkgfx_config.h...
(
echo #pragma once
echo // vkgfx_config.h -- auto-generated by build_release.bat
echo // Defines required by vkgfx public headers. Included automatically
echo // via vkgfx.h -- do not include this file directly.
echo.
echo // volk: use function-pointer dispatch instead of linking vulkan-1.lib
echo #if !defined^(VK_NO_PROTOTYPES^)
echo #    define VK_NO_PROTOTYPES
echo #endif
echo.
echo // volk.h must be included before any vulkan/ header
echo #include ^<volk.h^>
echo.
echo // GLM configuration
echo #if !defined^(GLM_FORCE_RADIANS^)
echo #    define GLM_FORCE_RADIANS
echo #endif
echo #if !defined^(GLM_FORCE_DEPTH_ZERO_TO_ONE^)
echo #    define GLM_FORCE_DEPTH_ZERO_TO_ONE
echo #endif
echo #if !defined^(GLM_ENABLE_EXPERIMENTAL^)
echo #    define GLM_ENABLE_EXPERIMENTAL
echo #endif
) > "%RELEASE_DIR%\include\vkgfx\vkgfx_config.h"

REM Prepend #include <vkgfx/vkgfx_config.h> to the release copy of vkgfx.h
REM so the defines are active before any other header is pulled in.
echo Patching vkgfx.h to include vkgfx_config.h...
if exist "%RELEASE_DIR%\include\vkgfx\vkgfx.h" (
    powershell -Command "$h='#include <vkgfx/vkgfx_config.h>'+[System.Environment]::NewLine; $h+(Get-Content '%RELEASE_DIR%\include\vkgfx\vkgfx.h' -Raw) | Set-Content '%RELEASE_DIR%\include\vkgfx\vkgfx.h' -NoNewline"
    echo   vkgfx.h patched.
) else (
    echo   [WARN] vkgfx.h not found - could not patch.
)

REM ── Bundle ALL dependency headers ──────────────────────────────
REM
REM vkgfx public headers include <glm/...>, <GLFW/...>, <vulkan/...> and <volk.h>.
REM All are fetched by FetchContent and copied here from the build cache.
REM End-users need NO Vulkan SDK and NO separate library installs.
REM

REM GLM  (header-only, MIT licence)
echo Bundling GLM headers...
if exist "%BUILD_DIR%\_deps\glm-src\glm" (
    xcopy /e /i /y "%BUILD_DIR%\_deps\glm-src\glm" "%RELEASE_DIR%\include\glm\" >nul
    echo   GLM headers copied.
) else (
    echo   [WARN] GLM source not found in build cache. GLM headers NOT bundled.
)

REM GLFW  (zlib licence)
echo Bundling GLFW headers...
if exist "%BUILD_DIR%\_deps\glfw-src\include\GLFW" (
    xcopy /e /i /y "%BUILD_DIR%\_deps\glfw-src\include\GLFW" "%RELEASE_DIR%\include\GLFW\" >nul
    echo   GLFW headers copied.
) else (
    echo   [WARN] GLFW source not found in build cache. GLFW headers NOT bundled.
)

REM Vulkan headers (Apache 2.0 licence)
REM vulkan_core.h #includes vk_video/ headers alongside vulkan/ -- both must be bundled.
echo Bundling Vulkan headers...
if exist "%BUILD_DIR%\_deps\vulkan_headers-src\include\vulkan" (
    xcopy /e /i /y "%BUILD_DIR%\_deps\vulkan_headers-src\include\vulkan"   "%RELEASE_DIR%\include\vulkan\"   >nul
    xcopy /e /i /y "%BUILD_DIR%\_deps\vulkan_headers-src\include\vk_video" "%RELEASE_DIR%\include\vk_video\" >nul
    echo   Vulkan + vk_video headers copied.
) else (
    echo   [WARN] Vulkan headers not found in build cache. NOT bundled.
)

REM volk.h  (MIT licence - Vulkan meta-loader, eliminates need for vulkan-1.lib)
echo Bundling volk.h...
if exist "%BUILD_DIR%\_deps\volk-src\volk.h" (
    xcopy /y "%BUILD_DIR%\_deps\volk-src\volk.h" "%RELEASE_DIR%\include\" >nul
    echo   volk.h copied.
) else (
    echo   [WARN] volk.h not found in build cache. NOT bundled.
)

REM ── Copy example sources (repo folder is "example", singular) ────────────────

mkdir "%RELEASE_DIR%\examples" 2>nul
if exist "example\*.cpp" (
    copy "example\*.cpp" "%RELEASE_DIR%\examples\" >nul
    echo Copied example sources.
)

REM ── Generate user-facing example CMakeLists.txt ──────────────────────────────

(
echo cmake_minimum_required^(VERSION 3.16^)
echo project^(my_vkgfx_app^)
echo.
echo set^(CMAKE_CXX_STANDARD 20^)
echo set^(CMAKE_CXX_STANDARD_REQUIRED ON^)
echo.
echo # Path to the vkgfx release folder
echo set^(VKGFX_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.."^)
echo.
echo # Import vkgfx as a pre-built static library
echo add_library^(vkgfx STATIC IMPORTED^)
echo set_target_properties^(vkgfx PROPERTIES
echo     IMPORTED_LOCATION "${VKGFX_ROOT}/lib/vkgfx.lib"
echo     INTERFACE_INCLUDE_DIRECTORIES "${VKGFX_ROOT}/include"
echo     INTERFACE_COMPILE_DEFINITIONS
echo         "VK_NO_PROTOTYPES;GLM_FORCE_RADIANS;GLM_FORCE_DEPTH_ZERO_TO_ONE;GLM_ENABLE_EXPERIMENTAL"^)
echo.
echo add_executable^(my_app main.cpp^)
echo target_link_libraries^(my_app PRIVATE vkgfx^)
echo.
echo # Copy compiled shaders next to the executable
echo add_custom_command^(TARGET my_app POST_BUILD
echo     COMMAND ${CMAKE_COMMAND} -E copy_directory
echo         "${VKGFX_ROOT}/bin/shaders"
echo         "$^<TARGET_FILE_DIR:my_app^>/shaders"^)
) > "%RELEASE_DIR%\examples\CMakeLists.txt"

REM ── Generate README.md ───────────────────────────────────────────────────────

(
echo # vkgfx — Simple Vulkan Graphics Engine
echo.
echo Pre-built binary release for **%OS_NAME%** / **%ARCH%** / **%CONFIG%**
echo.
echo ## Prerequisites
echo.
echo - **Visual Studio 2019+** with C++ workload -- that is all.
echo.
echo Everything else ^(GLM, GLFW, Vulkan headers, volk loader^) is bundled.
echo No Vulkan SDK install required.
echo.
echo ## Directory Structure
echo.
echo ```
echo %RELEASE_NAME%\
echo ├── include\
echo │   ├── vkgfx\        ^(vkgfx public headers^)
echo │   ├── glm\          ^(GLM math -- bundled^)
echo │   ├── GLFW\         ^(GLFW window -- bundled^)
echo │   ├── vulkan\       ^(Vulkan headers -- bundled^)
echo │   └── volk.h        ^(Vulkan loader -- bundled^)
echo ├── lib\vkgfx.lib     ^(static library -- link only this^)
echo ├── bin\shaders\      ^(compiled SPIR-V *.spv files^)
echo ├── examples\         ^(sample source + CMakeLists.txt^)
echo └── README.md
echo ```
echo.
echo ## Quick CMake Integration
echo.
echo ```cmake
echo set^(VKGFX_ROOT "${CMAKE_SOURCE_DIR}/thirdparty/%RELEASE_NAME%"^)
echo.
echo add_library^(vkgfx STATIC IMPORTED^)
echo set_target_properties^(vkgfx PROPERTIES
echo     IMPORTED_LOCATION_RELEASE "${VKGFX_ROOT}/lib/Release/vkgfx.lib"
echo     IMPORTED_LOCATION_DEBUG   "${VKGFX_ROOT}/lib/Debug/vkgfx.lib"
echo     INTERFACE_INCLUDE_DIRECTORIES "${VKGFX_ROOT}/include"
echo     INTERFACE_COMPILE_DEFINITIONS
echo         "VK_NO_PROTOTYPES;GLM_FORCE_RADIANS;GLM_FORCE_DEPTH_ZERO_TO_ONE;GLM_ENABLE_EXPERIMENTAL"^)
echo.
echo add_executable^(my_app src/main.cpp^)
echo target_link_libraries^(my_app PRIVATE vkgfx^)
echo ```
echo.
echo ## Important Notes
echo.
echo - Copy `bin\shaders\` next to your executable before running
echo - Validation layers are DISABLED in release; build from source for dev builds
echo - All deps ^(GLFW, GLM, stb, tinyobjloader, fastgltf, VMA, volk^) are statically linked
echo - No Vulkan SDK required: volk loads vulkan-1.dll from your GPU driver at runtime
echo.
echo ## Manual Visual Studio Setup (without CMake)
echo.
echo 1. Add `include\` to **C/C++** ^> Additional Include Directories
echo 2. Add `lib\vkgfx.lib` to **Linker** ^> Additional Dependencies
echo 3. Add these **Preprocessor Definitions**:
echo    `VK_NO_PROTOTYPES;GLM_FORCE_RADIANS;GLM_FORCE_DEPTH_ZERO_TO_ONE;GLM_ENABLE_EXPERIMENTAL`
echo 4. Copy `bin\shaders\` next to your .exe
) > "%RELEASE_DIR%\README.md"

REM ── Create ZIP ───────────────────────────────────────────────────────────────

echo.
echo Creating ZIP archive...
if exist "release\%RELEASE_NAME%.zip" del "release\%RELEASE_NAME%.zip"
powershell -Command "Compress-Archive -Path '%RELEASE_DIR%' -DestinationPath 'release\%RELEASE_NAME%.zip' -Force"

if errorlevel 1 (
    echo [WARN] ZIP creation failed. The folder is still available at %RELEASE_DIR%
) else (
    echo   ZIP OK.
)

REM ── Summary ──────────────────────────────────────────────────────────────────

echo.
echo ============================================
echo  Release package created successfully!
echo.
echo  Archive : release\%RELEASE_NAME%.zip
echo  Folder  : %RELEASE_DIR%\
echo  Log     : %LOG_FILE%
echo.
echo  Release contents:
dir /b "%RELEASE_DIR%"
echo ============================================
echo.
pause
endlocal
exit /b 0

REM ── Error handler ────────────────────────────────────────────────────────────

:error
echo.
echo ============================================
echo  BUILD FAILED
echo  See full log: %LOG_FILE%
echo ============================================
echo.
pause
endlocal
exit /b 1
