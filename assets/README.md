# assets/

Place runtime assets here before building.

## Required for IBL

`sky.hdr` — an equirectangular HDR environment map.

Download any free `.hdr` from https://polyhaven.com/hdris
and save it as `assets/sky.hdr`.

The renderer disables IBL gracefully if this file is absent.
