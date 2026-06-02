@echo off
REM build_release_mingw.bat - Build vkgfx release package for MinGW / Dev-C++
REM Usage: build_release_mingw.bat [release|debug]
REM
REM Requirements (developer building the library):
REM   - MinGW-w64 with GCC 11+ (C++20 required)
REM     Recommended: TDM-GCC 64-bit  https://jmeubank.github.io/tdm-gcc/
REM                  or MSYS2 mingw64 https://www.msys2.org/
REM   - CMake 3.24+               https://cmake.org/
REM   - Vulkan SDK (for glslc)    https://vulkan.lunarg.com/
REM
REM The produced libvkgfx.a works in Dev-C++ 5 / Orwell Dev-C++ IF the
REM project compiler is upgraded to GCC 11+.  Dev-C++ 5's built-in GCC 4.9
REM cannot compile C++20 headers - see the COMPILER NOTE below.

setlocal enabledelayedexpansion

set CONFIG=%1
if "%CONFIG%"=="" set CONFIG=release

if /i "%CONFIG%"=="debug" (
    set BUILD_TYPE=Debug
) else (
    set BUILD_TYPE=Release
)

set OS_NAME=windows
set ARCH=x86_64
set RELEASE_NAME=vkgfx-%OS_NAME%-%ARCH%-mingw-%CONFIG%
set RELEASE_DIR=release\%RELEASE_NAME%
set BUILD_DIR=build_mingw_%CONFIG%
set LOG_FILE=build_mingw_%CONFIG%.log

echo ============================================
echo  Building vkgfx release package (MinGW)
echo  Config  : %BUILD_TYPE%
echo  Output  : %RELEASE_DIR%
echo  Log     : %LOG_FILE%
echo ============================================
echo.

REM ── Sanity checks ────────────────────────────────────────────────────────────

where cmake >nul 2>&1
if errorlevel 1 (
    echo [ERROR] cmake not found on PATH.
    echo   Install CMake from https://cmake.org/ and tick "Add to PATH"
    goto :error
)

where gcc >nul 2>&1
if errorlevel 1 (
    echo [ERROR] gcc not found on PATH.
    echo.
    echo   Install one of:
    echo     TDM-GCC 64-bit : https://jmeubank.github.io/tdm-gcc/
    echo     MSYS2 mingw64  : https://www.msys2.org/  then: pacman -S mingw-w64-x86_64-gcc
    echo.
    goto :error
)

REM Check GCC version supports C++20 (need GCC 11+)
for /f "tokens=3" %%v in ('gcc --version 2^>^&1 ^| findstr /r "[0-9][0-9]*\.[0-9]"') do set GCC_VER=%%v
for /f "tokens=1 delims=." %%m in ("%GCC_VER%") do set GCC_MAJOR=%%m

if "%GCC_MAJOR%"=="" (
    echo [WARN] Could not detect GCC version. Proceeding anyway.
    goto :gcc_ok
)

if %GCC_MAJOR% LSS 11 (
    echo [ERROR] GCC %GCC_VER% found, but GCC 11+ is required for C++20.
    echo.
    echo   COMPILER NOTE FOR DEV-C++ 5 USERS:
    echo   Dev-C++ 5 ships with GCC 4.9 which does NOT support C++20.
    echo   To build vkgfx you need a newer compiler. Options:
    echo.
    echo   Option A - TDM-GCC 64-bit ^(easiest^):
    echo     1. Download from https://jmeubank.github.io/tdm-gcc/
    echo     2. Install, tick "Add to PATH"
    echo     3. Re-run this script
    echo     4. In Dev-C++ 5: Tools ^> Compiler Options ^> Add a compiler
    echo        point it at the TDM-GCC install folder
    echo.
    echo   Option B - MSYS2 ^(more complete toolchain^):
    echo     1. Install MSYS2 from https://www.msys2.org/
    echo     2. Run: pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-cmake
    echo     3. Add C:\msys64\mingw64\bin to PATH
    echo     4. Re-run this script
    echo.
    goto :error
)

:gcc_ok
echo   GCC %GCC_VER% - OK

REM Check for mingw32-make or make
set MAKE_CMD=
where mingw32-make >nul 2>&1 && set MAKE_CMD=mingw32-make
if "%MAKE_CMD%"=="" (
    where make >nul 2>&1 && set MAKE_CMD=make
)
if "%MAKE_CMD%"=="" (
    echo [ERROR] Neither mingw32-make nor make found on PATH.
    echo   They ship with MinGW/TDM-GCC. Ensure the MinGW bin folder is on PATH.
    goto :error
)
echo   Make: %MAKE_CMD% - OK

REM Check glslc (needed to compile shaders - developer only)
where glslc >nul 2>&1
if errorlevel 1 (
    echo [ERROR] glslc not found. Install Vulkan SDK from https://vulkan.lunarg.com/
    echo   glslc is only needed when BUILDING vkgfx - not by end users.
    goto :error
)

REM ── Clean previous build ─────────────────────────────────────────────────────

if exist "%BUILD_DIR%"   rmdir /s /q "%BUILD_DIR%"
if exist "%RELEASE_DIR%" rmdir /s /q "%RELEASE_DIR%"
if exist "%LOG_FILE%"    del "%LOG_FILE%"
mkdir "%BUILD_DIR%"
mkdir "%RELEASE_DIR%"

set ABS_RELEASE_DIR=%CD%\%RELEASE_DIR%
set ABS_LOG=%CD%\%LOG_FILE%

echo Build log: %ABS_LOG%
echo.

cd "%BUILD_DIR%"

REM ── [1/3] Configure ──────────────────────────────────────────────────────────

echo [1/3] Configuring with CMake (MinGW Makefiles)...
cmake .. ^
    -G "MinGW Makefiles" ^
    -DCMAKE_BUILD_TYPE="%BUILD_TYPE%" ^
    -DCMAKE_C_COMPILER=gcc ^
    -DCMAKE_CXX_COMPILER=g++ ^
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
    cd ..
    goto :error
)
echo   Configure OK.

REM ── [2/3] Build ──────────────────────────────────────────────────────────────

echo [2/3] Building library...
cmake --build . --target vkgfx -- -j%NUMBER_OF_PROCESSORS% >> "%ABS_LOG%" 2>&1

if errorlevel 1 (
    echo [ERROR] Build failed!
    echo.
    echo Last 50 lines of log:
    powershell -Command "Get-Content '%ABS_LOG%' -Tail 50"
    cd ..
    goto :error
)
echo   Build OK.

REM ── [3/3] Install ────────────────────────────────────────────────────────────

echo [3/3] Installing to %ABS_RELEASE_DIR%...
cmake --install . --prefix "%ABS_RELEASE_DIR%" >> "%ABS_LOG%" 2>&1

if errorlevel 1 (
    echo [ERROR] Install failed!
    echo.
    echo Last 30 lines of log:
    powershell -Command "Get-Content '%ABS_LOG%' -Tail 30"
    cd ..
    goto :error
)
echo   Install OK.

cd ..

REM ── Post-install: fix shader layout ─────────────────────────────────────────

mkdir "%RELEASE_DIR%\bin\shaders" 2>nul
if exist "%RELEASE_DIR%\share\vkgfx\shaders" (
    xcopy /e /y "%RELEASE_DIR%\share\vkgfx\shaders\*" "%RELEASE_DIR%\bin\shaders\" >nul
    rmdir /s /q "%RELEASE_DIR%\share"
)
if exist "%BUILD_DIR%\shaders" (
    xcopy /e /y "%BUILD_DIR%\shaders\*.spv" "%RELEASE_DIR%\bin\shaders\" >nul 2>&1
)

REM ── Generate vkgfx_config.h and patch vkgfx.h ────────────────────────────────

echo Generating vkgfx_config.h...
(
echo #pragma once
echo // vkgfx_config.h -- auto-generated by build_release_mingw.bat
echo // Defines required by vkgfx public headers. Included automatically
echo // via vkgfx.h -- do not include this file directly.
echo.
echo // volk: use function-pointer dispatch instead of linking libvulkan
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

echo Patching vkgfx.h...
if exist "%RELEASE_DIR%\include\vkgfx\vkgfx.h" (
    powershell -Command "$h='#include <vkgfx/vkgfx_config.h>'+[System.Environment]::NewLine; $h+(Get-Content '%RELEASE_DIR%\include\vkgfx\vkgfx.h' -Raw) | Set-Content '%RELEASE_DIR%\include\vkgfx\vkgfx.h' -NoNewline"
    echo   vkgfx.h patched.
)

REM ── Bundle ALL dependency headers ────────────────────────────────────────────

echo Bundling GLM headers...
if exist "%BUILD_DIR%\_deps\glm-src\glm" (
    xcopy /e /i /y "%BUILD_DIR%\_deps\glm-src\glm" "%RELEASE_DIR%\include\glm\" >nul
    echo   GLM headers copied.
) else ( echo   [WARN] GLM not found. )

echo Bundling GLFW headers...
if exist "%BUILD_DIR%\_deps\glfw-src\include\GLFW" (
    xcopy /e /i /y "%BUILD_DIR%\_deps\glfw-src\include\GLFW" "%RELEASE_DIR%\include\GLFW\" >nul
    echo   GLFW headers copied.
) else ( echo   [WARN] GLFW not found. )

echo Bundling Vulkan headers...
if exist "%BUILD_DIR%\_deps\vulkan_headers-src\include\vulkan" (
    xcopy /e /i /y "%BUILD_DIR%\_deps\vulkan_headers-src\include\vulkan"   "%RELEASE_DIR%\include\vulkan\"   >nul
    xcopy /e /i /y "%BUILD_DIR%\_deps\vulkan_headers-src\include\vk_video" "%RELEASE_DIR%\include\vk_video\" >nul
    echo   Vulkan + vk_video headers copied.
) else ( echo   [WARN] Vulkan headers not found. )

echo Bundling volk.h...
if exist "%BUILD_DIR%\_deps\volk-src\volk.h" (
    xcopy /y "%BUILD_DIR%\_deps\volk-src\volk.h" "%RELEASE_DIR%\include\" >nul
    echo   volk.h copied.
) else ( echo   [WARN] volk.h not found. )

REM ── Copy example sources ─────────────────────────────────────────────────────

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
echo set^(VKGFX_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.."^)
echo.
echo add_library^(vkgfx STATIC IMPORTED^)
echo set_target_properties^(vkgfx PROPERTIES
echo     IMPORTED_LOCATION "${VKGFX_ROOT}/lib/libvkgfx.a"
echo     INTERFACE_INCLUDE_DIRECTORIES "${VKGFX_ROOT}/include"
echo     INTERFACE_COMPILE_DEFINITIONS
echo         "VK_NO_PROTOTYPES;GLM_FORCE_RADIANS;GLM_FORCE_DEPTH_ZERO_TO_ONE;GLM_ENABLE_EXPERIMENTAL"^)
echo.
echo add_executable^(my_app main.cpp^)
echo target_link_libraries^(my_app PRIVATE vkgfx^)
echo.
echo add_custom_command^(TARGET my_app POST_BUILD
echo     COMMAND ${CMAKE_COMMAND} -E copy_directory
echo         "${VKGFX_ROOT}/bin/shaders"
echo         "$^<TARGET_FILE_DIR:my_app^>/shaders"^)
) > "%RELEASE_DIR%\examples\CMakeLists.txt"

REM ── Generate README.md ───────────────────────────────────────────────────────

(
echo # vkgfx -- Simple Vulkan Graphics Engine
echo.
echo Pre-built binary release for **Windows** / **x86_64** / **MinGW** / **%CONFIG%**
echo.
echo ## Prerequisites
echo.
echo - **GCC 11+** ^(C++20 required^) and **Dev-C++** or any MinGW IDE
echo - No Vulkan SDK, no extra libraries needed
echo.
echo ## IMPORTANT: Dev-C++ 5 Compiler Note
echo.
echo Dev-C++ 5's built-in GCC 4.9 does **not** support C++20.
echo You must point Dev-C++ at a newer compiler:
echo.
echo **Option A -- TDM-GCC 64-bit ^(easiest^)**
echo 1. Download from https://jmeubank.github.io/tdm-gcc/
echo 2. Install
echo 3. In Dev-C++: Tools ^> Compiler Options ^> ^(+^) Add compiler
echo    Set the bin folder to `C:\TDM-GCC-64\bin`
echo 4. Select the new compiler in the dropdown and rebuild
echo.
echo **Option B -- MSYS2**
echo 1. Install from https://www.msys2.org/
echo 2. Run: `pacman -S mingw-w64-x86_64-gcc`
echo 3. Set Dev-C++ bin folder to `C:\msys64\mingw64\bin`
echo.
echo ## Directory Structure
echo.
echo ```
echo %RELEASE_NAME%\
echo +-- include\
echo ^|   +-- vkgfx\        ^(vkgfx public headers^)
echo ^|   +-- glm\          ^(GLM math -- bundled^)
echo ^|   +-- GLFW\         ^(GLFW window -- bundled^)
echo ^|   +-- vulkan\       ^(Vulkan headers -- bundled^)
echo ^|   +-- volk.h        ^(Vulkan loader -- bundled^)
echo +-- lib\libvkgfx.a    ^(static library -- link only this^)
echo +-- bin\shaders\      ^(compiled SPIR-V *.spv files^)
echo +-- examples\         ^(sample source + CMakeLists.txt^)
echo +-- README.md
echo ```
echo.
echo ## Dev-C++ 5 Manual Setup
echo.
echo 1. **Project** ^> **Project Options** ^> **Parameters** tab
echo    - Add to Linker: `path\to\lib\libvkgfx.a`
echo 2. **Project Options** ^> **Directories** ^> **Include Directories**
echo    - Add: `path\to\include`
echo 3. **Project Options** ^> **Compiler** ^> add to **Defines**:
echo    `VK_NO_PROTOTYPES GLM_FORCE_RADIANS GLM_FORCE_DEPTH_ZERO_TO_ONE GLM_ENABLE_EXPERIMENTAL`
echo 4. Copy `bin\shaders\` folder next to your .exe before running
echo.
echo ## CMake Integration
echo.
echo ```cmake
echo set^(VKGFX_ROOT "${CMAKE_SOURCE_DIR}/thirdparty/%RELEASE_NAME%"^)
echo.
echo add_library^(vkgfx STATIC IMPORTED^)
echo set_target_properties^(vkgfx PROPERTIES
echo     IMPORTED_LOCATION "${VKGFX_ROOT}/lib/libvkgfx.lib"
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
echo - Copy `bin\shaders\` next to your .exe before running
echo - All deps ^(GLFW, GLM, stb, tinyobjloader, fastgltf, VMA, volk^) are statically linked
echo - No Vulkan SDK required: volk loads vulkan-1.dll from your GPU driver at runtime
) > "%RELEASE_DIR%\README.md"

REM ── Create ZIP ───────────────────────────────────────────────────────────────

echo.
echo Creating ZIP archive...
if exist "release\%RELEASE_NAME%.zip" del "release\%RELEASE_NAME%.zip"
powershell -Command "Compress-Archive -Path '%RELEASE_DIR%' -DestinationPath 'release\%RELEASE_NAME%.zip' -Force"
if errorlevel 1 (
    echo [WARN] ZIP creation failed. Folder is still at %RELEASE_DIR%
) else ( echo   ZIP OK. )

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
