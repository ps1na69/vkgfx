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

echo [2/3] Building library...
cmake --build . --config "%BUILD_TYPE%" --target vkgfx -- /m >> "%ABS_LOG%" 2>&1

if errorlevel 1 (
    echo [ERROR] Build failed!
    echo.
    echo Last 50 lines of log:
    powershell -Command "Get-Content '%ABS_LOG%' -Tail 50"
    echo.
    echo Full log: %ABS_LOG%
    cd ..
    goto :error
)
echo   Build OK.

REM ── [3/3] Install ────────────────────────────────────────────────────────────

REM --prefix must be passed here explicitly: with the Visual Studio multi-config
REM generator cmake --install ignores the prefix baked in at configure time
REM unless it is overridden on the command line.
echo [3/3] Installing to %ABS_RELEASE_DIR%...
cmake --install . --config "%BUILD_TYPE%" --prefix "%ABS_RELEASE_DIR%" >> "%ABS_LOG%" 2>&1

if errorlevel 1 (
    echo [ERROR] Install failed!
    echo.
    echo Last 30 lines of log:
    powershell -Command "Get-Content '%ABS_LOG%' -Tail 30"
    echo.
    echo Full log: %ABS_LOG%
    cd ..
    goto :error
)
echo   Install OK.

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
echo     INTERFACE_INCLUDE_DIRECTORIES "${VKGFX_ROOT}/include"^)
echo.
echo # Find Vulkan SDK
echo find_package^(Vulkan REQUIRED^)
echo.
echo add_executable^(my_app main.cpp^)
echo target_link_libraries^(my_app PRIVATE vkgfx Vulkan::Vulkan^)
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
echo - **Vulkan SDK** ^(https://vulkan.lunarg.com/^) — set `VULKAN_SDK` env var
echo - **Visual Studio 2019+** with C++ workload
echo.
echo ## Directory Structure
echo.
echo ```
echo %RELEASE_NAME%\
echo ├── include\vkgfx\    ^(public headers^)
echo ├── lib\vkgfx.lib     ^(static library^)
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
echo     IMPORTED_LOCATION "${VKGFX_ROOT}/lib/vkgfx.lib"
echo     INTERFACE_INCLUDE_DIRECTORIES "${VKGFX_ROOT}/include"^)
echo.
echo find_package^(Vulkan REQUIRED^)
echo add_executable^(my_app src/main.cpp^)
echo target_link_libraries^(my_app PRIVATE vkgfx Vulkan::Vulkan^)
echo ```
echo.
echo ## Important Notes
echo.
echo - Copy `bin\shaders\` next to your executable before running
echo - Validation layers are DISABLED in release; build from source for dev builds
echo - All deps ^(GLFW, GLM, stb, tinyobjloader^) are statically linked
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
