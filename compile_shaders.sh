#!/bin/bash
# compile_shaders.sh — Compile all GLSL shaders to SPIR-V
# Requires: glslc (from Vulkan SDK)

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHADER_DIR="$SCRIPT_DIR/shaders"
OUTPUT_DIR="${1:-$SHADER_DIR}"    # Output to same dir by default

mkdir -p "$OUTPUT_DIR"

echo "Compiling shaders in: $SHADER_DIR"
echo "Output to:            $OUTPUT_DIR"
echo ""

SHADERS=(
    "pbr.vert"
    "pbr.frag"
    "phong.vert"
    "phong.frag"
    "unlit.vert"
    "unlit.frag"
)

SUCCESS=0
FAIL=0

for shader in "${SHADERS[@]}"; do
    INPUT="$SHADER_DIR/$shader"
    OUTPUT="$OUTPUT_DIR/$shader.spv"
    if [ -f "$INPUT" ]; then
        if glslc "$INPUT" -o "$OUTPUT" --target-env=vulkan1.3; then
            SIZE=$(wc -c < "$OUTPUT")
            echo "  [OK]   $shader  →  $shader.spv  (${SIZE} bytes)"
            ((SUCCESS++))
        else
            echo "  [FAIL] $shader"
            ((FAIL++))
        fi
    else
        echo "  [SKIP] $shader  (not found)"
    fi
done

echo ""
echo "Done: $SUCCESS compiled, $FAIL failed."
exit $FAIL
