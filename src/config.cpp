// src/config.cpp
#include <vkgfx/config.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <stdexcept>

using json = nlohmann::json;

namespace vkgfx {

// ── Helpers ────────────────────────────────────────────────────────────────────

static void readVec3(const json& j, const std::string& key, float out[3]) {
    if (j.contains(key) && j[key].is_array() && j[key].size() >= 3) {
        out[0] = j[key][0];
        out[1] = j[key][1];
        out[2] = j[key][2];
    }
}

// ── RendererConfig::fromFile ──────────────────────────────────────────────────

RendererConfig RendererConfig::fromFile(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "[vkgfx] Config file not found: " << path
                  << " — using defaults\n";
        return {};
    }

    RendererConfig cfg;
    json j;
    try {
        f >> j;
    } catch (const json::parse_error& e) {
        std::cerr << "[vkgfx] Config parse error in " << path
                  << ": " << e.what() << " — using defaults\n";
        return {};
    }

    // Window / device
    if (j.contains("vsync"))      cfg.vsync     = j["vsync"];
    if (j.contains("shaderDir"))  cfg.shaderDir = j["shaderDir"];
    if (j.contains("assetDir"))   cfg.assetDir  = j["assetDir"];
    if (j.contains("msaa"))       cfg.msaa      = static_cast<MSAASamples>(j["msaa"].get<uint8_t>());

    // IBL
    if (j.contains("ibl")) {
        const auto& ib = j["ibl"];
        if (ib.contains("enabled"))        cfg.ibl.enabled        = ib["enabled"];
        if (ib.contains("hdrPath"))        cfg.ibl.hdrPath        = ib["hdrPath"];
        if (ib.contains("intensity"))      cfg.ibl.intensity      = ib["intensity"];
        if (ib.contains("envMapSize"))     cfg.ibl.envMapSize     = ib["envMapSize"];
        if (ib.contains("irradianceSize")) cfg.ibl.irradianceSize = ib["irradianceSize"];
    }

    // Sun
    if (j.contains("sun")) {
        const auto& s = j["sun"];
        if (s.contains("enabled"))   cfg.sun.enabled   = s["enabled"];
        if (s.contains("intensity")) cfg.sun.intensity = s["intensity"];
        readVec3(s, "direction", cfg.sun.direction);
        readVec3(s, "color",     cfg.sun.color);
    }

    // SSAO
    if (j.contains("ssao")) {
        const auto& ss = j["ssao"];
        if (ss.contains("enabled"))    cfg.ssao.enabled    = ss["enabled"];
        if (ss.contains("kernelSize")) cfg.ssao.kernelSize = ss["kernelSize"];
        if (ss.contains("radius"))     cfg.ssao.radius     = ss["radius"];
        if (ss.contains("bias"))       cfg.ssao.bias       = ss["bias"];
    }

    // Debug
    if (j.contains("gbufferDebug"))
        cfg.gbufferDebug = static_cast<GBufferDebugView>(j["gbufferDebug"].get<uint8_t>());

    return cfg;
}

// ── RendererConfig::save ──────────────────────────────────────────────────────

void RendererConfig::save(const std::string& path) const {
    json j;

    j["vsync"]     = vsync;
    j["shaderDir"] = shaderDir;
    j["assetDir"]  = assetDir;
    j["msaa"]      = static_cast<uint8_t>(msaa);

    j["ibl"] = {
        {"enabled",        ibl.enabled},
        {"hdrPath",        ibl.hdrPath},
        {"intensity",      ibl.intensity},
        {"envMapSize",     ibl.envMapSize},
        {"irradianceSize", ibl.irradianceSize}
    };

    j["sun"] = {
        {"enabled",   sun.enabled},
        {"direction", {sun.direction[0], sun.direction[1], sun.direction[2]}},
        {"color",     {sun.color[0],     sun.color[1],     sun.color[2]}},
        {"intensity", sun.intensity}
    };

    j["ssao"] = {
        {"enabled",    ssao.enabled},
        {"kernelSize", ssao.kernelSize},
        {"radius",     ssao.radius},
        {"bias",       ssao.bias}
    };

    j["gbufferDebug"] = static_cast<uint8_t>(gbufferDebug);

    std::ofstream f(path);
    if (!f.is_open()) {
        std::cerr << "[vkgfx] Cannot write config to: " << path << "\n";
        return;
    }
    f << j.dump(4) << "\n";
    std::cout << "[vkgfx] Config saved to: " << path << "\n";
}

} // namespace vkgfx
