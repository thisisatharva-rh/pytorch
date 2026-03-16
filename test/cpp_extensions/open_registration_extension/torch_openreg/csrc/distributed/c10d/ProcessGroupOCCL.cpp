#include "ProcessGroupOCCL.hpp"

namespace c10d {

ProcessGroupOCCL::Options::Options(std::chrono::milliseconds timeout)
    : Backend::Options(OCCL_BACKEND_NAME, timeout) {}

ProcessGroupOCCL::ProcessGroupOCCL(
    const c10::intrusive_ptr<Store>& store,
    int rank,
    int size,
    c10::intrusive_ptr<Options> options)
    : Backend(rank, size),
      store_(store),
      options_(std::move(options)) {}

ProcessGroupOCCL::~ProcessGroupOCCL() = default;

c10::intrusive_ptr<ProcessGroupOCCL> createProcessGroupOCCL(
    const c10::intrusive_ptr<Store>& store,
    int rank,
    int size,
    const std::chrono::duration<float>& timeout) {
  auto options = ProcessGroupOCCL::Options::create(
      std::chrono::milliseconds(
          static_cast<int64_t>(timeout.count() * 1000)));
  return c10::make_intrusive<ProcessGroupOCCL>(store, rank, size, options);
}

} // namespace c10d
