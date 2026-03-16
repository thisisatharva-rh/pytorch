#pragma once

#include <include/Macros.h>
#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/csrc/distributed/c10d/Work.hpp>

#include <chrono>

namespace c10d {

constexpr const char* OCCL_BACKEND_NAME = "occl";

class OPENREG_EXPORT ProcessGroupOCCL : public Backend {
 public:
  struct TORCH_API Options : public Backend::Options {
    explicit Options(
        std::chrono::milliseconds timeout = kBackendDefaultTimeout);

    static c10::intrusive_ptr<Options> create(
        std::chrono::milliseconds timeout = kBackendDefaultTimeout) {
      return c10::make_intrusive<Options>(timeout);
    }
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
  c10::intrusive_ptr<Store> store_;
  c10::intrusive_ptr<Options> options_;
};

OPENREG_EXPORT c10::intrusive_ptr<ProcessGroupOCCL> createProcessGroupOCCL(
    const c10::intrusive_ptr<Store>& store,
    int rank,
    int size,
    const std::chrono::duration<float>& timeout);

} // namespace c10d
