#include "edge_inference_optimizer/onnx_engine.hpp"

#include <algorithm>
#include <numeric>
#include <sstream>
#include <stdexcept>

#include "edge_inference_optimizer/latency_probe.hpp"

namespace eio
{

std::string to_string(Backend b)
{
  switch (b) {
    case Backend::kCuda: return "CUDAExecutionProvider";
    case Backend::kCpu:
    default: return "CPUExecutionProvider";
  }
}

std::string TensorSpec::shape_string() const
{
  std::ostringstream os;
  os << '[';
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i) { os << ','; }
    if (shape[i] < 0) { os << '?'; } else { os << shape[i]; }
  }
  os << ']';
  return os.str();
}

int64_t TensorSpec::element_count(int64_t dynamic_as) const
{
  int64_t n = 1;
  for (int64_t d : shape) { n *= (d < 0 ? dynamic_as : d); }
  return n;
}

OnnxEngine::OnnxEngine(Options opts)
: opts_(std::move(opts))
{
  build_session();
  introspect();
}

OnnxEngine::~OnnxEngine() = default;

void OnnxEngine::build_session()
{
  env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, opts_.log_tag.c_str());

  Ort::SessionOptions so;
  so.SetIntraOpNumThreads(opts_.intra_op_threads);
  so.SetInterOpNumThreads(opts_.inter_op_threads);
  so.SetGraphOptimizationLevel(opts_.opt_level);

  actual_backend_ = Backend::kCpu;

  if (opts_.backend == Backend::kCuda) {
    try {
      OrtCUDAProviderOptions cuda_opts{};
      cuda_opts.device_id = opts_.cuda_device_id;
      so.AppendExecutionProvider_CUDA(cuda_opts);
      actual_backend_ = Backend::kCuda;
    } catch (const Ort::Exception & e) {
      actual_backend_ = Backend::kCpu;
      cuda_error_ = e.what();
    }
  }

  session_ = std::make_unique<Ort::Session>(*env_, opts_.model_path.c_str(), so);

  // 注册成功不代表真的用上了(缺 cuDNN、算子不支持等),再问一次运行时
  if (actual_backend_ == Backend::kCuda) {
    bool cuda_present = false;
    for (const auto & p : Ort::GetAvailableProviders()) {
      if (p == "CUDAExecutionProvider") { cuda_present = true; break; }
    }
    if (!cuda_present) {
      actual_backend_ = Backend::kCpu;
      cuda_error_ = "CUDAExecutionProvider absent from Ort::GetAvailableProviders()";
    }
  }

  memory_info_ = std::make_unique<Ort::MemoryInfo>(
    Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
}

void OnnxEngine::introspect()
{
  const size_t n_in = session_->GetInputCount();
  const size_t n_out = session_->GetOutputCount();

  inputs_.reserve(n_in);
  outputs_.reserve(n_out);

  // GetTensorTypeAndShapeInfo() 返回的是对 TypeInfo 的非拥有视图。
  // 写成一行链式调用会让临时 TypeInfo 在分号处析构,GetShape() 静默返回空 vector。
  for (size_t i = 0; i < n_in; ++i) {
    TensorSpec spec;
    auto held = session_->GetInputNameAllocated(i, allocator_);
    spec.name = held.get();
    Ort::TypeInfo type_info = session_->GetInputTypeInfo(i);
    auto info = type_info.GetTensorTypeAndShapeInfo();
    spec.shape = info.GetShape();
    spec.type = info.GetElementType();
    inputs_.push_back(std::move(spec));
  }

  for (size_t i = 0; i < n_out; ++i) {
    TensorSpec spec;
    auto held = session_->GetOutputNameAllocated(i, allocator_);
    spec.name = held.get();
    Ort::TypeInfo type_info = session_->GetOutputTypeInfo(i);
    auto info = type_info.GetTensorTypeAndShapeInfo();
    spec.shape = info.GetShape();
    spec.type = info.GetElementType();
    outputs_.push_back(std::move(spec));
  }

  // 必须等两个 vector 都填完再取 c_str(),否则扩容会让指针悬空
  input_name_ptrs_.clear();
  output_name_ptrs_.clear();
  input_name_ptrs_.reserve(inputs_.size());
  output_name_ptrs_.reserve(outputs_.size());
  for (const auto & s : inputs_) { input_name_ptrs_.push_back(s.name.c_str()); }
  for (const auto & s : outputs_) { output_name_ptrs_.push_back(s.name.c_str()); }
}

Ort::Value OnnxEngine::make_tensor(
  float * data, size_t count, const std::vector<int64_t> & shape) const
{
  return Ort::Value::CreateTensor<float>(
    *memory_info_, data, count, shape.data(), shape.size());
}

Ort::Value OnnxEngine::make_tensor(
  int64_t * data, size_t count, const std::vector<int64_t> & shape) const
{
  return Ort::Value::CreateTensor<int64_t>(
    *memory_info_, data, count, shape.data(), shape.size());
}

std::vector<float> OnnxEngine::run(
  const std::vector<float> & input, const std::vector<int64_t> & shape)
{
  if (inputs_.size() != 1) {
    throw std::runtime_error(
            "OnnxEngine::run(single) called on a model with " +
            std::to_string(inputs_.size()) + " inputs; use run_named()");
  }

  int64_t expected = 1;
  for (int64_t d : shape) { expected *= d; }
  if (static_cast<int64_t>(input.size()) != expected) {
    throw std::runtime_error(
            "input size " + std::to_string(input.size()) +
            " does not match shape product " + std::to_string(expected));
  }

  // CreateTensor 只借用缓冲区;ORT 不写输入张量,且 input 的生命周期覆盖 Run
  auto tensor = make_tensor(
    const_cast<float *>(input.data()), input.size(), shape);

  const int64_t t0 = steady_ns();
  auto out = session_->Run(
    Ort::RunOptions{nullptr},
    input_name_ptrs_.data(), &tensor, 1,
    output_name_ptrs_.data(), output_name_ptrs_.size());
  last_inference_ms_ = ns_to_ms(steady_ns() - t0);

  auto info = out.front().GetTensorTypeAndShapeInfo();
  const size_t n = info.GetElementCount();
  const float * p = out.front().GetTensorData<float>();
  return std::vector<float>(p, p + n);
}

std::vector<Ort::Value> OnnxEngine::run(const std::vector<Ort::Value> & inputs)
{
  const int64_t t0 = steady_ns();
  auto out = session_->Run(
    Ort::RunOptions{nullptr},
    input_name_ptrs_.data(), inputs.data(), inputs.size(),
    output_name_ptrs_.data(), output_name_ptrs_.size());
  last_inference_ms_ = ns_to_ms(steady_ns() - t0);
  return out;
}

std::vector<Ort::Value> OnnxEngine::run_named(
  const std::vector<std::string> & input_names,
  const std::vector<Ort::Value> & inputs,
  const std::vector<std::string> & output_names)
{
  std::vector<const char *> in_ptrs, out_ptrs;
  in_ptrs.reserve(input_names.size());
  out_ptrs.reserve(output_names.size());
  for (const auto & s : input_names) { in_ptrs.push_back(s.c_str()); }
  for (const auto & s : output_names) { out_ptrs.push_back(s.c_str()); }

  const int64_t t0 = steady_ns();
  auto out = session_->Run(
    Ort::RunOptions{nullptr},
    in_ptrs.data(), inputs.data(), inputs.size(),
    out_ptrs.data(), out_ptrs.size());
  last_inference_ms_ = ns_to_ms(steady_ns() - t0);
  return out;
}

std::string OnnxEngine::describe() const
{
  std::ostringstream os;
  os << "OnnxEngine{model=" << opts_.model_path
     << ", requested=" << to_string(opts_.backend)
     << ", actual=" << to_string(actual_backend_)
     << ", intra_op=" << opts_.intra_op_threads
     << ", inter_op=" << opts_.inter_op_threads;
  if (!cuda_error_.empty()) { os << ", cuda_fallback_reason=\"" << cuda_error_ << '"'; }
  os << ", in=";
  for (const auto & s : inputs_) { os << s.name << s.shape_string() << ' '; }
  os << "out=";
  for (const auto & s : outputs_) { os << s.name << s.shape_string() << ' '; }
  os << '}';
  return os.str();
}

}  // namespace eio
