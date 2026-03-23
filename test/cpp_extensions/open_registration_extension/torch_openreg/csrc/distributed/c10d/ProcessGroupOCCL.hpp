#pragma once

#include <include/Macros.h>
#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/csrc/distributed/c10d/Work.hpp>

#include <chrono>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace c10d {

constexpr const char* OCCL_BACKEND_NAME = "occl";

class OPENREG_EXPORT ProcessGroupOCCL : public Backend {
 public:
  class OpenRegWork : public Work {
   public:
    OpenRegWork(
        std::function<void()> fn,
        std::vector<at::Tensor> outputTensors,
        OpType opType,
        const char* profilingTitle = nullptr,
        const std::optional<std::vector<at::Tensor>>& inputTensors =
            std::nullopt);

    ~OpenRegWork() override = default;

    static void execute(const c10::intrusive_ptr<OpenRegWork>& work);

    std::vector<at::Tensor> result() override;
    c10::intrusive_ptr<c10::ivalue::Future> getFuture() override;

   private:
    void finishWork();
    void finishWorkError(const std::exception_ptr& eptr);

    std::function<void()> fn_;
    const std::vector<at::Tensor> outputTensors_;
    c10::intrusive_ptr<at::ivalue::Future> future_;
  };

  struct TORCH_API Options : public Backend::Options {
    explicit Options(
        std::chrono::milliseconds timeout = kBackendDefaultTimeout);

    static c10::intrusive_ptr<Options> create(
        std::chrono::milliseconds timeout = kBackendDefaultTimeout) {
      return c10::make_intrusive<Options>(timeout);
    }

    int threads{2};
  };

  ProcessGroupOCCL(
      const c10::intrusive_ptr<Store>& store,
      int rank,
      int size,
      c10::intrusive_ptr<Options> options = Options::create());

  ~ProcessGroupOCCL() override;

  const std::string getBackendName() const override {
    return std::string(OCCL_BACKEND_NAME);
  }

 protected:
  void enqueue(c10::intrusive_ptr<OpenRegWork> work);

  c10::intrusive_ptr<Store> store_;
  c10::intrusive_ptr<Options> options_;

 private:
  void runLoop(int workerIndex);

  std::deque<c10::intrusive_ptr<OpenRegWork>> workQueue_;
  std::vector<std::thread> threads_;
  std::mutex workMutex_;
  std::condition_variable workProduceCV_;
  bool stop_{false};
};

OPENREG_EXPORT c10::intrusive_ptr<ProcessGroupOCCL> createProcessGroupOCCL(
    const c10::intrusive_ptr<Store>& store,
    int rank,
    int size,
    const std::chrono::duration<float>& timeout);

} // namespace c10d
