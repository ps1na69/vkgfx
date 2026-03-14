// example/main.cpp — deferred renderer demo
//
// Loads a glTF 2.0 scene and renders it with the deferred PBR pipeline.
// Press WASD to move, mouse drag to look, ESC to quit.

#include <vkgfx/vkgfx.h>
#include <iostream>
#include <filesystem>

int main(int argc, char** argv) {
    try {
        // ── Window ────────────────────────────────────────────────────────────
        vkgfx::WindowSettings ws;
        ws.title  = "vkgfx deferred";
        ws.width  = 1280;
        ws.height = 720;
        vkgfx::Window window(ws);

        // ── Renderer ──────────────────────────────────────────────────────────
        vkgfx::RendererSettings rs;
        rs.vsync       = true;
        rs.ssaoRadius  = 0.5f;
        rs.ssaoBias    = 0.025f;
        rs.exposure    = 0.f;     // 0 EV = neutral
        rs.tonemapOp   = 0;       // ACES filmic
        rs.shaderDir   = "shaders";
        vkgfx::Renderer renderer(window, rs);

        // ── Camera ────────────────────────────────────────────────────────────
        vkgfx::Camera camera;
        camera.setPosition({0.f, 1.5f, -4.f})
              .setYaw(-90.f)
              .setAspect(static_cast<float>(ws.width) / ws.height);

        // ── Scene ─────────────────────────────────────────────────────────────
        vkgfx::Scene scene(&camera);

        // Lights
        auto sun = std::make_shared<vkgfx::DirectionalLight>();
        sun->setDirection({-0.5f,-1.f,-0.5f})
            .setColor({1.f,0.95f,0.85f})
            .setIntensity(3.f);
        scene.add(sun);

        auto fill = std::make_shared<vkgfx::PointLight>();
        fill->setPosition({3.f,2.f,3.f})
             .setColor({0.4f,0.6f,1.f})
             .setIntensity(2.f);
        static_cast<vkgfx::PointLight*>(fill.get())->setRange(20.f);
        scene.add(fill);

        scene.setAmbient({0.1f,0.1f,0.15f}, 0.05f);

        // Geometry — load glTF if provided, otherwise use a unit cube.
        std::filesystem::path gltfPath = (argc > 1) ? argv[1] : "";
        if (!gltfPath.empty() && std::filesystem::exists(gltfPath)) {
            auto meshes = vkgfx::Mesh::fromGltf(renderer.contextPtr(), gltfPath);
            for (auto& m : meshes) scene.add(m);
            std::cout << "[demo] Loaded " << meshes.size() << " mesh(es) from " << gltfPath << "\n";
        } else {
            auto cube = vkgfx::Mesh::createCube(1.f);
            // Default white PBR material — metallic=0, roughness=0.6
            auto mat = std::make_shared<vkgfx::PBRMaterial>();
            mat->setAlbedo({0.8f,0.6f,0.4f,1.f}).setMetallic(0.f).setRoughness(0.6f);
            cube->setMaterial(mat);
            scene.add(cube);

            auto plane = vkgfx::Mesh::createPlane(8.f, 4);
            auto pmat  = std::make_shared<vkgfx::PBRMaterial>();
            pmat->setAlbedo({0.5f,0.5f,0.5f,1.f}).setMetallic(0.f).setRoughness(0.9f);
            plane->setMaterial(pmat);
            plane->setPosition({0.f,-0.5f,0.f});
            scene.add(plane);
        }

        // ── Input state ───────────────────────────────────────────────────────
        double lastX=0, lastY=0;
        bool   firstMouse=true, mouseDown=false;

        glfwSetWindowUserPointer(window.handle(), &mouseDown);
        glfwSetMouseButtonCallback(window.handle(), [](GLFWwindow* w, int btn, int action, int) {
            if (btn==GLFW_MOUSE_BUTTON_LEFT)
                *static_cast<bool*>(glfwGetWindowUserPointer(w)) = (action==GLFW_PRESS);
        });

        // ── Main loop ─────────────────────────────────────────────────────────
        float lastTime = static_cast<float>(glfwGetTime());
        while (!window.shouldClose()) {
            glfwPollEvents();

            float now = static_cast<float>(glfwGetTime());
            float dt  = now - lastTime;
            lastTime  = now;

            // Camera movement (WASD + QE)
            float speed = 3.f * dt;
            if (glfwGetKey(window.handle(),GLFW_KEY_W)==GLFW_PRESS) camera.translate( camera.forward()*speed);
            if (glfwGetKey(window.handle(),GLFW_KEY_S)==GLFW_PRESS) camera.translate(-camera.forward()*speed);
            if (glfwGetKey(window.handle(),GLFW_KEY_D)==GLFW_PRESS) camera.translate( camera.right()  *speed);
            if (glfwGetKey(window.handle(),GLFW_KEY_A)==GLFW_PRESS) camera.translate(-camera.right()  *speed);
            if (glfwGetKey(window.handle(),GLFW_KEY_E)==GLFW_PRESS) camera.translate( camera.up()     *speed);
            if (glfwGetKey(window.handle(),GLFW_KEY_Q)==GLFW_PRESS) camera.translate(-camera.up()     *speed);

            if (glfwGetKey(window.handle(),GLFW_KEY_ESCAPE)==GLFW_PRESS) break;

            // Mouse look
            double cx,cy;
            glfwGetCursorPos(window.handle(),&cx,&cy);
            if (mouseDown) {
                if (firstMouse) { lastX=cx; lastY=cy; firstMouse=false; }
                float dx=static_cast<float>(cx-lastX)*0.15f;
                float dy=static_cast<float>(cy-lastY)*0.15f;
                camera.rotate(dx,-dy);
            } else { firstMouse=true; }
            lastX=cx; lastY=cy;

            renderer.render(scene);

            // Update aspect on resize
            auto [w,h] = window.getFramebufferSize();
            if (w>0 && h>0)
                camera.setAspect(static_cast<float>(w)/static_cast<float>(h));
        }

        renderer.shutdown(&scene);
    } catch (const std::exception& e) {
        std::cerr << "[FATAL] " << e.what() << "\n";
        return 1;
    }
    return 0;
}