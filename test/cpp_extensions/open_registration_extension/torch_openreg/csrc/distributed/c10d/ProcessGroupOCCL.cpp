#include "ProcessGroupOCCL.hpp"

#include <c10/util/irange.h>

namespace c10d {

namespace {

c10::intrusive_ptr<c10::ivalue::Future> createFuture(
    const std::vector<at::Tensor>& outputTensors) {
  std::vector<at::Device> devices;
  for (const auto& tensor : outputTensors) {
    if (!tensor.device().is_cpu()) {
      devices.push_back(tensor.device());
    }
  }
  return c10::make_intrusive<c10::ivalue::Future>(
      c10::ListType::create(c10::TensorType::get()), devices);
}

} // namespace

ProcessGroupOCCL::OpenRegWork::OpenRegWork(
    std::function<void()> fn,
    std::vector<at::Tensor> outputTensors,
    OpType opType,
    const char* profilingTitle,
    const std::optional<std::vector<at::Tensor>>& inputTensors)
    : Work(-1, opType, profilingTitle, inputTensors),
      fn_(std::move(fn)),
      outputTensors_(std::move(outputTensors)),
      future_(createFuture(outputTensors_)) {}

void ProcessGroupOCCL::OpenRegWork::execute(
    const c10::intrusive_ptr<OpenRegWork>& work) {
  try {
    work->fn_();
  } catch (...) {
    work->finishWorkError(std::current_exception());
    return;
  }
  work->finishWork();
}

void ProcessGroupOCCL::OpenRegWork::finishWork() {
  future_->markCompleted(c10::IValue(outputTensors_));
  finish();
}

void ProcessGroupOCCL::OpenRegWork::finishWorkError(
    const std::exception_ptr& eptr) {
  future_->setError(eptr);
  finish(eptr);
}

std::vector<at::Tensor> ProcessGroupOCCL::OpenRegWork::result() {
  TORCH_CHECK(
      isCompleted(),
      "Work needs to be completed before calling result(). "
      "Should call wait() before result().");
  return outputTensors_;
}

c10::intrusive_ptr<c10::ivalue::Future>
ProcessGroupOCCL::OpenRegWork::getFuture() {
  return future_;
}

ProcessGroupOCCL::Options::Options(std::chrono::milliseconds timeout)
    : Backend::Options(OCCL_BACKEND_NAME, timeout) {}

ProcessGroupOCCL::ProcessGroupOCCL(
    const c10::intrusive_ptr<Store>& store,
    int rank,
    int size,
    c10::intrusive_ptr<Options> options)
    : Backend(rank, size),
      store_(store),
      options_(std::move(options)) {
  threads_.resize(options_->threads);
  for (const auto i : c10::irange(threads_.size())) {
    threads_[i] = std::thread(&ProcessGroupOCCL::runLoop, this, i);
  }
}

ProcessGroupOCCL::~ProcessGroupOCCL() {
  {
    std::lock_guard<std::mutex> lock(workMutex_);
    stop_ = true;
  }
  workProduceCV_.notify_all();
  for (auto& thread : threads_) {
    thread.join();
  }
}

void ProcessGroupOCCL::runLoop(int /* workerIndex */) {
  std::unique_lock<std::mutex> lock(workMutex_);
  while (!stop_) {
    if (workQueue_.empty()) {
      workProduceCV_.wait(lock);
      continue;
    }
    auto work = std::move(workQueue_.front());
    workQueue_.pop_front();
    lock.unlock();
    OpenRegWork::execute(work);
    lock.lock();
  }
}

void ProcessGroupOCCL::enqueue(c10::intrusive_ptr<OpenRegWork> work) {
  {
    std::lock_guard<std::mutex> lock(workMutex_);
    workQueue_.push_back(std::move(work));
  }
  workProduceCV_.notify_one();
}

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
