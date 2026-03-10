#!/bin/bash

for file in shaders/*.vert shaders/*.frag shaders/*.comp shaders/*.geom shaders/*.tesc shaders/*.tese
do
    echo Compiling $file
    glslc "$file" -o "$file.spv"
done

echo Done