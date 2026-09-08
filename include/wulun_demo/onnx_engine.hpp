// ONNX 封装。线程数要自己写，别跟 Python 对比时核数不一样。

#ifndef WULUN_DEMO__ONNX_ENGINE_HPP_
#define WULUN_DEMO__ONNX_ENGINE_HPP_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "onnxruntime/core/session/onnxruntime_cxx_api.h"

namespace eio
{

struct TensorSpec
{
  std::string name;
  std::vector<int64_t> shape;   ///< -1 表示动态轴
  ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;

  std::string shape_string() const;
  /// 元素个数。
  int64_t element_count(int64_t dynamic_as = 1) const;
};

enum class Backend
{
  kCpu,
  kCuda,
};

std::string to_string(Backend b);

class OnnxEngine
{
public:
  struct Options
  {
    std::string model_path;
    Backend backend = Backend::kCpu;

    int intra_op_threads = 1;
    int inter_op_threads = 1;

    GraphOptimizationLevel opt_level = GraphOptimizationLevel::ORT_ENABLE_ALL;
    int cuda_device_id = 0;
    std::string log_tag = "eio";
  };

  explicit OnnxEngine(Options opts);
  ~OnnxEngine();

  OnnxEngine(const OnnxEngine &) = delete;
  OnnxEngine & operator=(const OnnxEngine &) = delete;

  /// 单输入单输出。尺寸不对就抛错。
  std::vector<float> run(const std::vector<float> & input, const std::vector<int64_t> & shape);

  std::vector<Ort::Value> run(const std::vector<Ort::Value> & inputs);

  /// 按名字指定输入输出。
  std::vector<Ort::Value> run_named(
    const std::vector<std::string> & input_names,
    const std::vector<Ort::Value> & inputs,
    const std::vector<std::string> & output_names);

  /// 只有模型推理的时间。
  double last_inference_ms() const { return last_inference_ms_; }

  const std::vector<TensorSpec> & inputs() const { return inputs_; }
  const std::vector<TensorSpec> & outputs() const { return outputs_; }

  /// 实际用的后端。CUDA 挂了会退回 CPU，别按请求值报。
  Backend backend() const { return actual_backend_; }
  /// CUDA 退回 CPU 的原因，没有就空着。
  const std::string & cuda_fallback_reason() const { return cuda_error_; }
  const std::string & model_path() const { return opts_.model_path; }

  /// 用现有内存做张量，不拷贝。
  Ort::Value make_tensor(float * data, size_t count, const std::vector<int64_t> & shape) const;
  Ort::Value make_tensor(int64_t * data, size_t count, const std::vector<int64_t> & shape) const;

  std::string describe() const;

private:
  void build_session();
  void introspect();

  Options opts_;
  Backend actual_backend_ = Backend::kCpu;
  std::string cuda_error_;

  std::unique_ptr<Ort::Env> env_;
  std::unique_ptr<Ort::Session> session_;
  std::unique_ptr<Ort::MemoryInfo> memory_info_;
  Ort::AllocatorWithDefaultOptions allocator_;

  std::vector<TensorSpec> inputs_;
  std::vector<TensorSpec> outputs_;
  std::vector<const char *> input_name_ptrs_;
  std::vector<const char *> output_name_ptrs_;

  double last_inference_ms_ = 0.0;
};

}  // namespace eio

#endif  // WULUN_DEMO__ONNX_ENGINE_HPP_
