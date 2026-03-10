@echo off

for %%f in (shaders\*.vert shaders\*.frag shaders\*.comp shaders\*.geom shaders\*.tesc shaders\*.tese) do (
    echo Compiling %%f
    glslc %%f -o %%f.spv
)

echo Done
pause