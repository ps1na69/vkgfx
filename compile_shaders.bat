@echo off
setlocal enabledelayedexpansion
:: ============================================================================
::  compile_shaders.bat -- Compile GLSL shaders to SPIR-V  (Windows)
::  Requires: Vulkan SDK installed (glslc must be on PATH or in VULKAN_SDK\bin)
::
::  Usage:
::    compile_shaders.bat              -- compile into shaders\ (default)
::    compile_shaders.bat <output_dir> -- compile into <output_dir>
:: ============================================================================

:: -- Locate script directory ---------------------------------------------
set "SCRIPT_DIR=%~dp0"
set "SHADER_DIR=%SCRIPT_DIR%shaders"

:: -- Output dir (first argument or same as source) -----------------------
if "%~1"=="" (
    set "OUTPUT_DIR=%SCRIPT_DIR%shaders"
) else (
    set "OUTPUT_DIR=%~1"
)

:: -- Locate glslc --------------------------------------------------------
where glslc >nul 2>&1
if %errorlevel% equ 0 (
    set "GLSLC=glslc"
    goto :found_glslc
)

if defined VULKAN_SDK (
    if exist "%VULKAN_SDK%\bin\glslc.exe" (
        set "GLSLC=%VULKAN_SDK%\bin\glslc.exe"
        goto :found_glslc
    )
)

if defined VK_SDK_PATH (
    if exist "%VK_SDK_PATH%\bin\glslc.exe" (
        set "GLSLC=%VK_SDK_PATH%\bin\glslc.exe"
        goto :found_glslc
    )
)

echo [ERROR] glslc.exe not found.
echo         Install the Vulkan SDK from https://vulkan.lunarg.com/
echo         and make sure %%VULKAN_SDK%%\bin is on your PATH.
exit /b 1

:found_glslc
echo [VKGFX] glslc     : !GLSLC!

:: -- Ensure output directory exists -------------------------------------
if not exist "!OUTPUT_DIR!" mkdir "!OUTPUT_DIR!"

echo [VKGFX] Source dir : !SHADER_DIR!
echo [VKGFX] Output dir : !OUTPUT_DIR!
echo.

:: -- Shader list ---------------------------------------------------------
set SHADERS=pbr.vert pbr.frag phong.vert phong.frag unlit.vert unlit.frag postprocess.vert postprocess.frag shadow.vert

set /a SUCCESS=0
set /a FAIL=0
set /a SKIP=0

for %%S in (%SHADERS%) do (
    :: Use !VAR! (delayed expansion) for all paths.
    :: %VAR% inside a for do-block is expanded once at parse time before
    :: any iteration runs, so variables set in if/else branches above
    :: may bake in as empty. !VAR! is re-evaluated on every iteration.
    set "INPUT=!SHADER_DIR!\%%S"
    set "OUTPUT=!OUTPUT_DIR!\%%S.spv"

    if not exist "!INPUT!" (
        echo   [SKIP] %%S  (not found)
        set /a SKIP+=1
    ) else (
        "!GLSLC!" "!INPUT!" -o "!OUTPUT!" --target-env=vulkan1.3 -O
        if !errorlevel! equ 0 (
            for %%F in ("!OUTPUT!") do set "SIZE=%%~zF"
            echo   [OK]   %%S  ->  %%S.spv  (!SIZE! bytes)
            set /a SUCCESS+=1
        ) else (
            echo   [FAIL] %%S
            set /a FAIL+=1
        )
    )
)

echo.
echo -----------------------------------------
echo   Compiled : %SUCCESS%
echo   Failed   : %FAIL%
echo   Skipped  : %SKIP%
echo -----------------------------------------

if %FAIL% gtr 0 (
    echo [ERROR] Some shaders failed to compile.
    exit /b 1
)

echo [OK] All shaders compiled successfully.
exit /b 0
