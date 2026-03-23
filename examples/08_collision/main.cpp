// examples/08_collision/main.cpp
// Demonstrates: CollisionWorld with AABB and sphere colliders.
//
// Scene setup:
//   - A static floor AABB that never moves
//   - A red sphere that the player moves with WASD
//   - A blue sphere that falls under simulated gravity
//   - Ray cast with Left Click (unlock cursor first with Tab)
//
// Collision response automatically separates overlapping objects.
// The callback prints a message when the moving spheres touch.

#include <vkgfx/vkgfx.h>
#include <chrono>
#include <iostream>

int main() {
    using namespace vkgfx;

    Window   window("08 – Collision", 1280, 720);
    window.setCursorLocked(true);

    RendererConfig cfg;
    cfg.ibl.enabled  = false;
    cfg.sun.enabled  = true;
    cfg.sun.intensity= 4.f;

    Renderer renderer(window, cfg);
    Context& ctx = renderer.context();

    // ── Scene ─────────────────────────────────────────────────────────────────
    Camera cam;
    cam.setPosition({0.f, 3.f, -8.f}).setFov(60.f);

    Scene scene;
    scene.setCamera(&cam);

    auto sun = std::make_shared<DirectionalLight>();
    sun->setDirection(-0.4f,-1.f,-0.3f).setIntensity(4.f);
    scene.add(sun);

    std::vector<std::shared_ptr<Mesh>> meshes;

    // Static floor (wide flat box)
    auto floor = Mesh::createBox({6.f, 0.2f, 6.f}, ctx);
    floor->setPosition({0.f, -1.5f, 0.f});
    auto floorMat = std::make_shared<PBRMaterial>();
    floorMat->setAlbedo(0.4f, 0.4f, 0.4f).setRoughness(0.9f);
    floor->setMaterial(floorMat);
    scene.add(floor); meshes.push_back(floor);

    // Player sphere (red, WASD-controlled)
    auto player = Mesh::createSphere(0.5f, 24, 24, ctx);
    player->setPosition({-2.f, 0.f, 0.f});
    auto playerMat = std::make_shared<PBRMaterial>();
    playerMat->setAlbedo(0.9f, 0.1f, 0.1f).setRoughness(0.4f).setMetallic(0.3f);
    player->setMaterial(playerMat);
    scene.add(player); meshes.push_back(player);

    // Dynamic sphere (blue, simulated gravity)
    auto ball = Mesh::createSphere(0.5f, 24, 24, ctx);
    ball->setPosition({1.5f, 3.f, 0.f});
    auto ballMat = std::make_shared<PBRMaterial>();
    ballMat->setAlbedo(0.1f, 0.3f, 1.f).setRoughness(0.2f).setMetallic(0.6f);
    ball->setMaterial(ballMat);
    scene.add(ball); meshes.push_back(ball);

    // ── Collision world ───────────────────────────────────────────────────────
    CollisionWorld physics(3.f);

    // Floor is static — never moved by collision response
    physics.add(floor.get(),  Collider::makeAABB({6.f, 0.2f, 6.f}), /*isStatic=*/true)
           .tag = "floor";
    // Player and ball are dynamic
    auto& playerObj = physics.add(player.get(), Collider::makeSphere(0.5f));
    playerObj.tag = "player";
    auto& ballObj   = physics.add(ball.get(),   Collider::makeSphere(0.5f));
    ballObj.tag = "ball";

    // Callback: print once when two named objects touch
    physics.setOnContact([](const CollisionEvent& ev) {
        std::string a = ev.objectA->tag, b = ev.objectB->tag;
        if ((a == "player" && b == "ball") || (a == "ball" && b == "player"))
            std::cout << "[collision] player ↔ ball  depth=" << ev.contact.depth << "\n";
    });

    // Simulated gravity velocity for the ball
    glm::vec3 ballVel{0.f, -2.f, 0.f};

    std::cout << "WASD = move player (red sphere)\n"
              << "Ball (blue) falls under gravity\n"
              << "Tab = unlock cursor, Left Click = ray cast\n";

    auto last = std::chrono::high_resolution_clock::now();

    while (!window.shouldClose()) {
        auto now = std::chrono::high_resolution_clock::now();
        float dt = std::chrono::duration<float>(now - last).count();
        last = now;
        dt = std::min(dt, 0.05f); // clamp large frames

        window.pollEvents();
        if (window.keyPressed(GLFW_KEY_ESCAPE)) break;
        if (window.keyPressed(GLFW_KEY_TAB))    window.toggleCursorLock();

        // Camera
        float d = 5.f * dt;
        if (window.keyHeld(GLFW_KEY_UP))    cam.moveForward( d);
        if (window.keyHeld(GLFW_KEY_DOWN))  cam.moveForward(-d);
        cam.rotateYaw  ( window.mouseDX() * 0.1f);
        cam.rotatePitch(-window.mouseDY() * 0.1f);

        // Player controlled with WASD — move in XZ plane
        float pd = 4.f * dt;
        glm::vec3 pp = glm::vec3(player->modelMatrix()[3]);
        if (window.keyHeld(GLFW_KEY_W)) pp.z += pd;
        if (window.keyHeld(GLFW_KEY_S)) pp.z -= pd;
        if (window.keyHeld(GLFW_KEY_A)) pp.x -= pd;
        if (window.keyHeld(GLFW_KEY_D)) pp.x += pd;
        player->setPosition(pp);

        // Integrate ball gravity
        ballVel.y -= 9.8f * dt;
        glm::vec3 bp = glm::vec3(ball->modelMatrix()[3]);
        bp += ballVel * dt;
        ball->setPosition(bp);

        // Run collision — separates overlapping objects
        physics.update(true);

        // Simple velocity zeroing after floor hit
        if (glm::vec3(ball->modelMatrix()[3]).y <= -0.8f) {
            if (ballVel.y < 0.f) ballVel.y = std::abs(ballVel.y) * 0.6f; // bounce
        }

        // Ray cast on left click (when cursor unlocked)
        if (!window.isCursorLocked()) {
            if (glfwGetMouseButton(window.handle(), GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
                glm::mat4 view = cam.view();
                glm::vec3 fwd  = glm::normalize(-glm::vec3(view[0][2], view[1][2], view[2][2]));
                auto hit = physics.castRay(cam.position(), fwd, 50.f);
                if (hit)
                    std::cout << "[raycast] hit '" << (hit.mesh == floor.get() ? "floor" :
                               hit.mesh == player.get() ? "player" : "ball")
                              << "' at t=" << hit.t << "\n";
            }
        }

        renderer.render(scene);
    }

    vkDeviceWaitIdle(ctx.device());
    for (auto& m : meshes) m->destroy(ctx);
    renderer.shutdown();
    return 0;
}
