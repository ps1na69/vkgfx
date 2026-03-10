#include "vkgfx/scene.h"
#include "vkgfx/thread_pool.h"
#include <algorithm>
#include <atomic>

namespace vkgfx {

Scene::Scene(Camera* camera) : m_camera(camera) {}

Scene& Scene::add(std::shared_ptr<Mesh> mesh) {
    m_meshes.push_back(std::move(mesh));
    return *this;
}

Scene& Scene::add(std::shared_ptr<Light> light) {
    if (m_lights.size() >= MAX_SCENE_LIGHTS)
        std::cerr << "[VKGFX] Warning: MAX_SCENE_LIGHTS reached, light ignored\n";
    else
        m_lights.push_back(std::move(light));
    return *this;
}

Scene& Scene::setCamera(Camera* camera) {
    m_camera = camera;
    return *this;
}

void Scene::remove(std::shared_ptr<Mesh> mesh) {
    m_meshes.erase(std::remove(m_meshes.begin(), m_meshes.end(), mesh), m_meshes.end());
}

void Scene::remove(std::shared_ptr<Light> light) {
    m_lights.erase(std::remove(m_lights.begin(), m_lights.end(), light), m_lights.end());
}

void Scene::clear() {
    m_meshes.clear();
    m_lights.clear();
}

SceneUBO Scene::buildSceneUBO() const {
    SceneUBO ubo;
    ubo.ambientColor = { m_ambientColor, m_ambientIntensity };
    ubo.lightCount   = static_cast<int>(std::min(m_lights.size(), (size_t)MAX_SCENE_LIGHTS));
    for (int i = 0; i < ubo.lightCount; ++i)
        ubo.lights[i] = m_lights[i]->toLightData();
    return ubo;
}

std::vector<Mesh*> Scene::visibleMeshes(ThreadPool* pool) const {
    const size_t n = m_meshes.size();
    std::vector<Mesh*> visible;
    visible.reserve(n);

    if (!m_camera) {
        for (auto& m : m_meshes) if (m->isVisible()) visible.push_back(m.get());
        return visible;
    }

    const auto& frustum = m_camera->frustum();

    // ── Parallel frustum culling for large scenes (threshold: 64 meshes) ─────
    // FIX: Previously spawned raw std::thread objects every frame (very expensive —
    // thread creation involves kernel calls and stack allocation).
    // Now uses the caller-supplied ThreadPool if available, falling back to
    // single-threaded for small scenes or when no pool is given.
    if (pool && n >= 64) {
        std::vector<uint8_t> flags(n, 0);

        pool->parallelFor(
            static_cast<uint32_t>(pool->threadCount()),
            [&](uint32_t t) {
                const uint32_t chunkSize = static_cast<uint32_t>((n + pool->threadCount() - 1) / pool->threadCount());
                const size_t start = static_cast<size_t>(t) * chunkSize;
                if (start >= n) return;
                const size_t end = std::min(start + chunkSize, n);
                for (size_t i = start; i < end; ++i) {
                    auto* m = m_meshes[i].get();
                    if (m->isVisible() && frustum.containsAABB(m->worldBounds()))
                        flags[i] = 1;
                }
            }
        );

        for (size_t i = 0; i < n; ++i)
            if (flags[i]) visible.push_back(m_meshes[i].get());
    } else {
        // Single-threaded: small scenes, or pool not available
        for (auto& m : m_meshes) {
            if (!m->isVisible()) continue;
            if (frustum.containsAABB(m->worldBounds()))
                visible.push_back(m.get());
        }
    }
    return visible;
}

} // namespace vkgfx
