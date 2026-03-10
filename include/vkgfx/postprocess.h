#pragma once
#include "types.h"

namespace vkgfx {

// ─── GPU-side uniform layout (std140 / alignas-16) ──────────────────────────
struct alignas(16) PostProcessUBO {
    float exposure       = 1.0f;
    float brightness     = 0.0f;
    float contrast       = 1.0f;
    float saturation     = 1.0f;
    Vec4  colorBalance   = {1.f, 1.f, 1.f, 1.f};  // rgb = per-channel tint
    float bloomThreshold = 0.8f;
    float bloomStrength  = 0.4f;
    float bloomRadius    = 1.5f;
    int   toneMapMode    = 2;   // 0=linear, 1=Reinhard, 2=ACES
    int   bloomEnabled   = 0;
    int   tonemapEnabled = 1;
    float _pad[2];
};

// ─── User-facing settings ─────────────────────────────────────────────────────
struct PostProcessSettings {
    bool  enabled = false;

    // ── Exposure / tone mapping ───────────────────────────────────────────────
    float exposure    = 1.0f;   ///< HDR exposure multiplier. >1 = brighter, <1 = darker.
    bool  toneMapping = true;   ///< Apply tone mapping (maps HDR → display range).
    enum class ToneMapMode { Linear = 0, Reinhard = 1, ACES = 2 }
          toneMapMode = ToneMapMode::ACES;

    // ── Colour grading ────────────────────────────────────────────────────────
    float brightness = 0.0f;             ///< Additive brightness offset.  Range: –1 .. +1
    float contrast   = 1.0f;             ///< Contrast multiplier.          1 = neutral.
    float saturation = 1.0f;             ///< Colour saturation.            0 = greyscale.
    Vec3  colorBalance = {1.f, 1.f, 1.f};///< Per-channel RGB tint multipliers.

    // ── Bloom ─────────────────────────────────────────────────────────────────
    bool  bloom          = false;
    float bloomThreshold = 0.8f;  ///< Luminance threshold for bloom contribution.
    float bloomStrength  = 0.4f;  ///< Overall bloom intensity.
    float bloomRadius    = 1.5f;  ///< Tap spread radius (texels).

    // ─── Convert to GPU layout ────────────────────────────────────────────────
    [[nodiscard]] PostProcessUBO toUBO() const {
        PostProcessUBO u;
        u.exposure       = exposure;
        u.brightness     = brightness;
        u.contrast       = contrast;
        u.saturation     = saturation;
        u.colorBalance   = {colorBalance.r, colorBalance.g, colorBalance.b, 1.f};
        u.bloomThreshold = bloomThreshold;
        u.bloomStrength  = bloomStrength;
        u.bloomRadius    = bloomRadius;
        u.toneMapMode    = static_cast<int>(toneMapMode);
        u.bloomEnabled   = bloom     ? 1 : 0;
        u.tonemapEnabled = toneMapping ? 1 : 0;
        return u;
    }
};

} // namespace vkgfx
