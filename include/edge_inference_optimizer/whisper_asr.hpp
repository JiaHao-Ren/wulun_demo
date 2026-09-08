// Whisper ASR: log-mel、encoder、带 KV cache 的解码。

#ifndef EDGE_INFERENCE_OPTIMIZER__WHISPER_ASR_HPP_
#define EDGE_INFERENCE_OPTIMIZER__WHISPER_ASR_HPP_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "edge_inference_optimizer/onnx_engine.hpp"

namespace eio
{

/// Whisper 固定的音频参数,不可调:导出的 encoder 输入形状里写死了 3000
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
  /// 加载 mel 滤波。文件不对直接抛,别默默用全零。
  explicit MelExtractor(const std::string & mel_filters_path);

  /// 补齐或截断到 30 秒,返回 80*3000 个 float(mel 优先序)
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
    /// decoder_with_past_model.onnx。为空时每步都重算整个前缀,复杂度 O(n^2)
    std::string decoder_with_past_path;
    std::string mel_filters_path;
    std::string vocab_path;
    Backend backend = Backend::kCpu;
    int threads = 1;
    int max_tokens = 128;

    /// 强制语言。短句(1-3 秒)上 Whisper 自己检测语言不可靠,
    /// 而且指定 <|en|> 处理中文音频时它会「翻译」而不是报错。
    int64_t language_token = 50259;

    /// <|startofprev|> 后的引导。中文用简体提示避免简繁混用。
    std::vector<int64_t> initial_prompt_tokens;
  };

  /// 预编码好的简体中文引导词:「以下是普通话的对话内容,请使用简体中文转写。」
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

  /// 转写一段话,pcm 必须是 16 kHz 单声道
  Result transcribe(const std::vector<float> & pcm);

  const OnnxEngine & encoder() const { return *encoder_; }

  // Whisper 的强制前缀。缺 <|notimestamps|> 会让模型输出时间戳 token,
  // 这些 token 在这里解码为空串,看上去像模型坏了。
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

#endif  // EDGE_INFERENCE_OPTIMIZER__WHISPER_ASR_HPP_
