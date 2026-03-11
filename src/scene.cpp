#include "vkgfx/scene.h"
#include <algorithm>
#include <atomic>

namespace vkgfx {

Scene::Scene(Camera* camera) : m_camera(camera) {}

Scene& Scene::add(std::shared_ptr<Mesh> mesh)  { m_meshes.push_back(std::move(mesh));  return *this; }
Scene& Scene::add(std::shared_ptr<Light> light) {
    if (m_lights.size() >= MAX_LIGHTS)
        std::cerr << "[VKGFX] MAX_LIGHTS reached, light ignored\n";
    else
        m_lights.push_back(std::move(light));
    return *this;
}
void Scene::remove(std::shared_ptr<Mesh> m) {
    m_meshes.erase(std::remove(m_meshes.begin(), m_meshes.end(), m), m_meshes.end());
}
void Scene::remove(std::shared_ptr<Light> l) {
    m_lights.erase(std::remove(m_lights.begin(), m_lights.end(), l), m_lights.end());
}
void Scene::clear() { m_meshes.clear(); m_lights.clear(); }

uint32_t Scene::buildLightBuffer(LightSSBO& out) const {
    uint32_t n = static_cast<uint32_t>(std::min(m_lights.size(), (size_t)MAX_LIGHTS));
    out.count  = n;
    for (uint32_t i = 0; i < n; ++i) out.lights[i] = m_lights[i]->toGpuLight();
    return n;
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

    if (pool && n >= 64) {
        std::vector<uint8_t> flags(n, 0);
        pool->parallelFor(static_cast<uint32_t>(pool->threadCount()), [&](uint32_t t) {
            const uint32_t chunk = static_cast<uint32_t>((n + pool->threadCount() - 1) / pool->threadCount());
            size_t start = static_cast<size_t>(t) * chunk;
            if (start >= n) return;
            size_t end = std::min(start + chunk, n);
            for (size_t i = start; i < end; ++i) {
                auto* m = m_meshes[i].get();
                if (m->isVisible() && frustum.containsAABB(m->worldBounds())) flags[i] = 1;
            }
        });
        for (size_t i = 0; i < n; ++i) if (flags[i]) visible.push_back(m_meshes[i].get());
    } else {
        for (auto& m : m_meshes)
            if (m->isVisible() && frustum.containsAABB(m->worldBounds()))
                visible.push_back(m.get());
    }
    return visible;
}

} // namespace vkgfx
