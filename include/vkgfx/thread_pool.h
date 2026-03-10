#pragma once
// ─── ThreadPool ──────────────────────────────────────────────────────────────
// Simple lock-free-queue-backed thread pool using C++20.
// Designed for short parallel tasks (frustum culling, cmd recording, etc.)
// ─────────────────────────────────────────────────────────────────────────────
#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace vkgfx {

class ThreadPool {
public:
    explicit ThreadPool(uint32_t threadCount = 0) {
        if (threadCount == 0)
            threadCount = std::max(1u, std::thread::hardware_concurrency());
        m_workers.reserve(threadCount);
        for (uint32_t i = 0; i < threadCount; ++i) {
            m_workers.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock lock(m_mutex);
                        m_cv.wait(lock, [this] { return m_stop || !m_tasks.empty(); });
                        if (m_stop && m_tasks.empty()) return;
                        task = std::move(m_tasks.front());
                        m_tasks.pop();
                    }
                    task();
                    --m_pending;
                    m_doneCv.notify_one();
                }
            });
        }
    }

    ~ThreadPool() {
        {
            std::lock_guard lock(m_mutex);
            m_stop = true;
        }
        m_cv.notify_all();
        for (auto& w : m_workers) w.join();
    }

    template<typename F>
    auto submit(F&& f) -> std::future<std::invoke_result_t<F>> {
        using R = std::invoke_result_t<F>;
        auto task = std::make_shared<std::packaged_task<R()>>(std::forward<F>(f));
        auto future = task->get_future();
        {
            std::lock_guard lock(m_mutex);
            ++m_pending;
            m_tasks.emplace([task] { (*task)(); });
        }
        m_cv.notify_one();
        return future;
    }

    // Submit N parallel tasks, wait for all to finish
    template<typename F>
    void parallelFor(uint32_t count, F&& f) {
        if (count == 0) return;
        std::atomic<uint32_t> done{0};
        std::vector<std::future<void>> futures;
        futures.reserve(count);
        for (uint32_t i = 0; i < count; ++i) {
            futures.push_back(submit([i, &f, &done] {
                f(i);
                ++done;
            }));
        }
        for (auto& fut : futures) fut.wait();
    }

    void waitAll() {
        std::unique_lock lock(m_mutex);
        m_doneCv.wait(lock, [this] { return m_pending == 0; });
    }

    [[nodiscard]] uint32_t threadCount() const {
        return static_cast<uint32_t>(m_workers.size());
    }

private:
    std::vector<std::thread>          m_workers;
    std::queue<std::function<void()>> m_tasks;
    std::mutex                        m_mutex;
    std::condition_variable           m_cv;
    std::condition_variable           m_doneCv;
    std::atomic<int>                  m_pending{0};
    bool                              m_stop = false;
};

} // namespace vkgfx
