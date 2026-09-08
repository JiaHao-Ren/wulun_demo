// piper TTS。espeak-ng 没有可用的开发头,所以运行时 dlopen。

#ifndef EDGE_INFERENCE_OPTIMIZER__PIPER_TTS_HPP_
#define EDGE_INFERENCE_OPTIMIZER__PIPER_TTS_HPP_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "edge_inference_optimizer/onnx_engine.hpp"

namespace eio
{

/// 文本 -> IPA 音素,运行时加载 libespeak-ng
class EspeakPhonemizer
{
public:
  /// lib_path 为 .so 路径,data_path 为 espeak-ng-data 目录;加载失败抛异常
  EspeakPhonemizer(
    const std::string & lib_path, const std::string & data_path, const std::string & voice);
  ~EspeakPhonemizer();

  EspeakPhonemizer(const EspeakPhonemizer &) = delete;
  EspeakPhonemizer & operator=(const EspeakPhonemizer &) = delete;

  /// 按顺序返回音素,每个音素一个 UTF-8 字符串
  std::vector<std::string> phonemize(const std::string & text) const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

class PiperTts
{
public:
  struct Options
  {
    std::string model_path;         ///< zh_CN-huayan-medium.onnx
    std::string phoneme_map_path;   ///< phoneme_map.bin
    std::string espeak_lib;
    std::string espeak_data;
    std::string voice = "cmn";      ///< 普通话
    Backend backend = Backend::kCpu;
    int threads = 4;

    // VITS 采样参数,默认值取自该音色自带的 json
    float noise_scale = 0.667f;
    float length_scale = 1.0f;      ///< 大于 1 语速变慢
    float noise_w = 0.8f;
  };

  struct Audio
  {
    std::vector<float> samples;
    int sample_rate = 22050;
    double phonemize_ms = 0.0;
    double infer_ms = 0.0;
    int phoneme_count = 0;
    double duration_s() const
    {
      return sample_rate > 0 ? static_cast<double>(samples.size()) / sample_rate : 0.0;
    }
  };

  explicit PiperTts(Options opts);
  ~PiperTts();

  Audio synthesize(const std::string & text);

  int sample_rate() const { return sample_rate_; }

  /// 写单声道 PCM16 WAV,供 CLI 使用,也用于无声卡机器上留存输出
  static bool write_wav(
    const std::string & path, const std::vector<float> & samples, int sample_rate);

private:
  Options opts_;
  std::unique_ptr<EspeakPhonemizer> phonemizer_;
  std::unique_ptr<OnnxEngine> engine_;
  std::unordered_map<std::string, int64_t> phoneme_ids_;
  int sample_rate_ = 22050;
};

}  // namespace eio

#endif  // EDGE_INFERENCE_OPTIMIZER__PIPER_TTS_HPP_
