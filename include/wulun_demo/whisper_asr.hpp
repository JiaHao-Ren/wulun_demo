// Whisper 语音识别。

#ifndef WULUN_DEMO__WHISPER_ASR_HPP_
#define WULUN_DEMO__WHISPER_ASR_HPP_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "wulun_demo/onnx_engine.hpp"

namespace eio
{

/// Whisper 音频参数，encoder 输入长度是写死的。
struct WhisperAudioSpec
{
  static constexpr int kSampleRate = 16000;
  static constexpr int kNFft = 400;
  static constexpr int kHop = 160;
  static constexpr int kNMels = 80;
  static constexpr int kNFrames = 3000;          // 30 s / hop
  static constexpr int kNSamples = 480000;       // 30 s
  static constexpr int kNFreq = kNFft / 2 + 1;   // 201
};

class MelExtractor
{
public:
  /// 加载 mel 滤波。文件不对就报错。
  explicit MelExtractor(const std::string & mel_filters_path);

  /// 补齐或截到 30 秒。
  std::vector<float> compute(const std::vector<float> & pcm) const;

  const std::vector<float> & filters() const { return filters_; }

private:
  std::vector<float> filters_;    // kNMels x kNFreq
  std::vector<float> window_;     // 周期 Hann,kNFft
};

class WhisperVocab
{
public:
  explicit WhisperVocab(const std::string & vocab_path);

  std::string decode(const std::vector<int64_t> & ids) const;

  size_t size() const { return entries_.size(); }

private:
  std::vector<std::string> entries_;
};

class WhisperAsr
{
public:
  struct Options
  {
    std::string encoder_path;
    std::string decoder_path;
    /// 带缓存的 decoder。空着就每步重算。
    std::string decoder_with_past_path;
    std::string mel_filters_path;
    std::string vocab_path;
    Backend backend = Backend::kCpu;
    int threads = 1;
    int max_tokens = 128;

    /// 语言写死。短句自己检测不准；中文别标成英文，会翻译。
    int64_t language_token = 50259;

    /// 中文用简体提示，免得简繁混着出。
    std::vector<int64_t> initial_prompt_tokens;
  };

  /// 简体中文提示词。
  static const std::vector<int64_t> & simplified_chinese_prompt();

  struct Result
  {
    std::string text;
    std::vector<int64_t> tokens;
    double preprocess_ms = 0.0;
    double encoder_ms = 0.0;
    double decoder_ms = 0.0;
    int decoded_tokens = 0;
  };

  explicit WhisperAsr(Options opts);

  /// 转写。pcm 要 16 kHz 单声道。
  Result transcribe(const std::vector<float> & pcm);

  const OnnxEngine & encoder() const { return *encoder_; }

  // 前缀里要带 <|notimestamps|>，不然会出一堆空 token。
  static constexpr int64_t kSot = 50258;
  static constexpr int64_t kEn = 50259;
  static constexpr int64_t kZh = 50260;
  static constexpr int64_t kTranscribe = 50359;
  static constexpr int64_t kNoTimestamps = 50363;
  static constexpr int64_t kEot = 50257;
  static constexpr int64_t kStartOfPrev = 50361;

private:
  Options opts_;
  std::unique_ptr<MelExtractor> mel_;
  std::unique_ptr<WhisperVocab> vocab_;
  std::unique_ptr<OnnxEngine> encoder_;
  std::unique_ptr<OnnxEngine> decoder_;         ///< 无状态,用于第一次前向
  std::unique_ptr<OnnxEngine> decoder_past_;    ///< 增量,用于后续每一步

  int n_layers_ = 0;
  std::vector<std::string> past_input_names_;
  std::vector<std::string> past_output_names_;
};

}  // namespace eio

#endif  // WULUN_DEMO__WHISPER_ASR_HPP_
