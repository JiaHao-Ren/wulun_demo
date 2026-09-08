#include "edge_inference_optimizer/whisper_asr.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>

#include "opencv2/core.hpp"

#include "edge_inference_optimizer/latency_probe.hpp"

namespace eio
{

namespace
{
constexpr int kNMels = WhisperAudioSpec::kNMels;
constexpr int kNFreq = WhisperAudioSpec::kNFreq;
constexpr int kNFft = WhisperAudioSpec::kNFft;
constexpr int kHop = WhisperAudioSpec::kHop;
constexpr int kNFrames = WhisperAudioSpec::kNFrames;
constexpr int kNSamples = WhisperAudioSpec::kNSamples;
}  // namespace

const std::vector<int64_t> & WhisperAsr::simplified_chinese_prompt()
{
  // 「以下是普通话的对话内容,请使用简体中文转写。」用 Whisper tokenizer 预先编码。
  // 存 id 而非文本:C++ 侧只有 id->字节 的解码表,没有编码器。
  static const std::vector<int64_t> kPrompt{
    3588, 4438, 1541, 29993, 19550, 21596, 1546, 8713, 21596, 34742, 25750, 11,
    27908, 22982, 9254, 11249, 222, 29485, 5975, 17174, 17819, 105, 5676, 247, 1543};
  return kPrompt;
}


MelExtractor::MelExtractor(const std::string & path)
{
  std::ifstream f(path, std::ios::binary);
  if (!f) { throw std::runtime_error("cannot open mel filters: " + path); }

  uint32_t rows = 0, cols = 0;
  f.read(reinterpret_cast<char *>(&rows), 4);
  f.read(reinterpret_cast<char *>(&cols), 4);
  if (!f || rows != static_cast<uint32_t>(kNMels) || cols != static_cast<uint32_t>(kNFreq)) {
    throw std::runtime_error(
            "mel filter shape is " + std::to_string(rows) + "x" + std::to_string(cols) +
            ", expected " + std::to_string(kNMels) + "x" + std::to_string(kNFreq));
  }

  filters_.resize(static_cast<size_t>(rows) * cols);
  f.read(reinterpret_cast<char *>(filters_.data()),
    static_cast<std::streamsize>(filters_.size() * sizeof(float)));
  if (f.gcount() != static_cast<std::streamsize>(filters_.size() * sizeof(float))) {
    throw std::runtime_error("mel filter file truncated: " + path);
  }

  // 周期 Hann(分母是 N,不是 N-1)。torch.hann_window 默认 periodic=True;
  // 用对称版本会让每个频点轻微偏移,转写照样出结果,只是悄悄错了。
  window_.resize(kNFft);
  for (int n = 0; n < kNFft; ++n) {
    window_[n] = 0.5f - 0.5f * std::cos(2.0f * static_cast<float>(CV_PI) *
        static_cast<float>(n) / static_cast<float>(kNFft));
  }
}

std::vector<float> MelExtractor::compute(const std::vector<float> & pcm) const
{
  // 1. 补齐或截断到 30 秒
  std::vector<float> audio(kNSamples, 0.0f);
  const size_t n = std::min(pcm.size(), static_cast<size_t>(kNSamples));
  std::copy(pcm.begin(), pcm.begin() + static_cast<std::ptrdiff_t>(n), audio.begin());

  // 2. 两端各做 n_fft/2 的镜像填充,对应 torch.stft 的 center=True。
  //    用 center=False 会让整个帧网格偏移 200 个采样点,与参考实现全部错位。
  const int pad = kNFft / 2;
  std::vector<float> padded(static_cast<size_t>(kNSamples + 2 * pad));
  for (int i = 0; i < pad; ++i) {
    padded[i] = audio[static_cast<size_t>(pad - i)];                          // 镜像
    padded[kNSamples + pad + i] = audio[static_cast<size_t>(kNSamples - 2 - i)];
  }
  std::copy(audio.begin(), audio.end(), padded.begin() + pad);

  // 3. STFT -> 功率谱 -> mel -> 取对数
  std::vector<float> power(static_cast<size_t>(kNFrames) * kNFreq);
  cv::Mat frame(1, kNFft, CV_32F);
  cv::Mat spectrum;

  for (int t = 0; t < kNFrames; ++t) {
    float * dst = frame.ptr<float>();
    const float * src = padded.data() + static_cast<size_t>(t) * kHop;
    for (int i = 0; i < kNFft; ++i) { dst[i] = src[i] * window_[i]; }

    cv::dft(frame, spectrum, cv::DFT_COMPLEX_OUTPUT);
    const cv::Vec2f * s = spectrum.ptr<cv::Vec2f>();
    float * p = power.data() + static_cast<size_t>(t) * kNFreq;
    for (int k = 0; k < kNFreq; ++k) {
      p[k] = s[k][0] * s[k][0] + s[k][1] * s[k][1];   // 幅度平方
    }
  }

  // filters (80 x 201) 乘 power 转置 (201 x 3000),得到 (80 x 3000)
  std::vector<float> mel(static_cast<size_t>(kNMels) * kNFrames);
  for (int m = 0; m < kNMels; ++m) {
    const float * fr = filters_.data() + static_cast<size_t>(m) * kNFreq;
    float * out = mel.data() + static_cast<size_t>(m) * kNFrames;
    for (int t = 0; t < kNFrames; ++t) {
      const float * p = power.data() + static_cast<size_t>(t) * kNFreq;
      float acc = 0.0f;
      for (int k = 0; k < kNFreq; ++k) { acc += fr[k] * p[k]; }
      out[t] = acc;
    }
  }

  float max_log = -1e30f;
  for (auto & v : mel) {
    v = std::log10(std::max(v, 1e-10f));
    max_log = std::max(max_log, v);
  }
  // 动态范围下限取峰值以下 8 个数量级,再缩放到约 [-1, 1]
  const float floor_v = max_log - 8.0f;
  for (auto & v : mel) { v = (std::max(v, floor_v) + 4.0f) / 4.0f; }

  return mel;
}


WhisperVocab::WhisperVocab(const std::string & path)
{
  std::ifstream f(path, std::ios::binary);
  if (!f) { throw std::runtime_error("cannot open vocab: " + path); }

  uint32_t count = 0;
  f.read(reinterpret_cast<char *>(&count), 4);
  if (!f) { throw std::runtime_error("vocab header truncated: " + path); }

  entries_.reserve(count);
  for (uint32_t i = 0; i < count; ++i) {
    uint32_t len = 0;
    f.read(reinterpret_cast<char *>(&len), 4);
    if (!f) { throw std::runtime_error("vocab truncated at token " + std::to_string(i)); }
    std::string s(len, '\0');
    if (len) { f.read(s.data(), len); }
    entries_.push_back(std::move(s));
  }
}

std::string WhisperVocab::decode(const std::vector<int64_t> & ids) const
{
  std::string out;
  for (int64_t id : ids) {
    if (id < 0 || static_cast<size_t>(id) >= entries_.size()) { continue; }
    out += entries_[static_cast<size_t>(id)];
  }
  return out;
}


WhisperAsr::WhisperAsr(Options opts)
: opts_(std::move(opts))
{
  mel_ = std::make_unique<MelExtractor>(opts_.mel_filters_path);
  vocab_ = std::make_unique<WhisperVocab>(opts_.vocab_path);

  OnnxEngine::Options eo;
  eo.backend = opts_.backend;
  eo.intra_op_threads = opts_.threads;
  eo.inter_op_threads = opts_.threads;

  eo.model_path = opts_.encoder_path;
  eo.log_tag = "whisper_enc";
  encoder_ = std::make_unique<OnnxEngine>(eo);

  eo.model_path = opts_.decoder_path;
  eo.log_tag = "whisper_dec";
  decoder_ = std::make_unique<OnnxEngine>(eo);

  if (!opts_.decoder_with_past_path.empty()) {
    eo.model_path = opts_.decoder_with_past_path;
    eo.log_tag = "whisper_dec_past";
    decoder_past_ = std::make_unique<OnnxEngine>(eo);

    // 层数从计算图读取,base/small/medium 都能直接用
    int self_attn = 0;
    for (const auto & in : decoder_past_->inputs()) {
      if (in.name.rfind("past_key_values.", 0) == 0 &&
        in.name.find(".decoder.key") != std::string::npos)
      {
        ++self_attn;
      }
    }
    n_layers_ = self_attn;

    // 输入顺序必须和计算图一致。
    past_input_names_ = {"input_ids"};
    past_output_names_ = {"logits"};
    for (int i = 0; i < n_layers_; ++i) {
      const auto p = "past_key_values." + std::to_string(i) + ".";
      past_input_names_.push_back(p + "decoder.key");
      past_input_names_.push_back(p + "decoder.value");
      past_input_names_.push_back(p + "encoder.key");
      past_input_names_.push_back(p + "encoder.value");
      // 只返回 self-attention 的状态:cross-attention 的 KV 只取决于 encoder 输出,
      // 第一次前向算完之后,后续每个 token 原样复用。
      const auto q = "present." + std::to_string(i) + ".";
      past_output_names_.push_back(q + "decoder.key");
      past_output_names_.push_back(q + "decoder.value");
    }
  }
}

WhisperAsr::Result WhisperAsr::transcribe(const std::vector<float> & pcm)
{
  Result r;

  const int64_t t0 = steady_ns();
  std::vector<float> mel = mel_->compute(pcm);
  r.preprocess_ms = ns_to_ms(steady_ns() - t0);

  const int64_t t1 = steady_ns();
  std::vector<Ort::Value> enc_in;
  enc_in.push_back(
    encoder_->make_tensor(mel.data(), mel.size(), {1, kNMels, kNFrames}));
  auto enc_out = encoder_->run_named({"input_features"}, enc_in, {"last_hidden_state"});
  r.encoder_ms = ns_to_ms(steady_ns() - t1);

  auto enc_info = enc_out.front().GetTensorTypeAndShapeInfo();
  const auto enc_shape = enc_info.GetShape();
  const float * enc_data = enc_out.front().GetTensorData<float>();
  const size_t enc_count = enc_info.GetElementCount();
  // 拷出来:下面解码循环每步都会新建张量,不能持有可能被重新分配的借用指针
  std::vector<float> enc_states(enc_data, enc_data + enc_count);

  std::vector<int64_t> tokens;
  if (!opts_.initial_prompt_tokens.empty()) {
    tokens.push_back(kStartOfPrev);
    tokens.insert(
      tokens.end(), opts_.initial_prompt_tokens.begin(), opts_.initial_prompt_tokens.end());
  }
  tokens.insert(tokens.end(), {kSot, opts_.language_token, kTranscribe, kNoTimestamps});
  // 到这里为止都是脚手架,只有模型在此之后生成的内容才算转写结果
  const size_t prompt_len = tokens.size();

  const int64_t t2 = steady_ns();

  auto argmax_last = [](const Ort::Value & v) -> int64_t {
      auto info = v.GetTensorTypeAndShapeInfo();
      const auto shape = info.GetShape();             // [1, seq, vocab]
      const int64_t vocab = shape.back();
      const float * last = v.GetTensorData<float>() + (shape[1] - 1) * vocab;
      int64_t best = 0;
      float best_v = -1e30f;
      for (int64_t i = 0; i < vocab; ++i) {
        if (last[i] > best_v) { best_v = last[i]; best = i; }
      }
      return best;
    };

  if (!decoder_past_) {
    // 无缓存回退路径:每步重算整个前缀。
    // 保留它是为了在没导出 decoder_with_past_model.onnx 时流水线仍能跑。
    for (int step = 0; step < opts_.max_tokens; ++step) {
      std::vector<int64_t> ids = tokens;
      std::vector<Ort::Value> dec_in;
      dec_in.push_back(
        decoder_->make_tensor(ids.data(), ids.size(), {1, static_cast<int64_t>(ids.size())}));
      dec_in.push_back(
        decoder_->make_tensor(enc_states.data(), enc_states.size(), enc_shape));
      auto dec_out = decoder_->run_named(
        {"input_ids", "encoder_hidden_states"}, dec_in, {"logits"});
      const int64_t best = argmax_last(dec_out.front());
      if (best == kEot) { break; }
      tokens.push_back(best);
    }
  } else {
    // ---- 第一次前向:整个 prompt,同时产出两部分缓存 ----
    std::vector<std::string> first_outputs{"logits"};
    for (int i = 0; i < n_layers_; ++i) {
      const auto q = "present." + std::to_string(i) + ".";
      first_outputs.push_back(q + "decoder.key");
      first_outputs.push_back(q + "decoder.value");
      first_outputs.push_back(q + "encoder.key");
      first_outputs.push_back(q + "encoder.value");
    }

    std::vector<int64_t> ids = tokens;
    std::vector<Ort::Value> dec_in;
    dec_in.push_back(
      decoder_->make_tensor(ids.data(), ids.size(), {1, static_cast<int64_t>(ids.size())}));
    dec_in.push_back(
      decoder_->make_tensor(enc_states.data(), enc_states.size(), enc_shape));
    auto first = decoder_->run_named(
      {"input_ids", "encoder_hidden_states"}, dec_in, first_outputs);

    // self_cache 每步增长一行,cross_cache 始终不变
    std::vector<Ort::Value> self_cache, cross_cache;
    self_cache.reserve(static_cast<size_t>(n_layers_) * 2);
    cross_cache.reserve(static_cast<size_t>(n_layers_) * 2);
    for (int i = 0; i < n_layers_; ++i) {
      self_cache.push_back(std::move(first[1 + 4 * i + 0]));
      self_cache.push_back(std::move(first[1 + 4 * i + 1]));
      cross_cache.push_back(std::move(first[1 + 4 * i + 2]));
      cross_cache.push_back(std::move(first[1 + 4 * i + 3]));
    }

    int64_t next = argmax_last(first.front());

    // ---- 后续每一步:一次一个 token ----
    for (int step = 0; step < opts_.max_tokens; ++step) {
      if (next == kEot) { break; }
      tokens.push_back(next);

      std::vector<int64_t> one{next};
      std::vector<Ort::Value> in;
      in.reserve(1 + static_cast<size_t>(n_layers_) * 4);
      in.push_back(decoder_past_->make_tensor(one.data(), 1, {1, 1}));
      for (int i = 0; i < n_layers_; ++i) {
        in.push_back(std::move(self_cache[2 * i]));
        in.push_back(std::move(self_cache[2 * i + 1]));
        in.push_back(std::move(cross_cache[2 * i]));
        in.push_back(std::move(cross_cache[2 * i + 1]));
      }

      auto out2 = decoder_past_->run_named(past_input_names_, in, past_output_names_);

      // Run() 只借用输入不消耗,cross-attention 张量仍然完好,可以直接移回来。
      // 若改成每个 token 拷贝 37 MB 不变的缓存,开销比前向本身还大。
      for (int i = 0; i < n_layers_; ++i) {
        cross_cache[2 * i] = std::move(in[1 + 4 * i + 2]);
        cross_cache[2 * i + 1] = std::move(in[1 + 4 * i + 3]);
        self_cache[2 * i] = std::move(out2[1 + 2 * i]);
        self_cache[2 * i + 1] = std::move(out2[1 + 2 * i + 1]);
      }
      next = argmax_last(out2.front());
    }
  }
  r.decoder_ms = ns_to_ms(steady_ns() - t2);

  std::vector<int64_t> emitted(tokens.begin() + static_cast<std::ptrdiff_t>(prompt_len),
    tokens.end());
  r.decoded_tokens = static_cast<int>(emitted.size());
  r.tokens = emitted;
  r.text = vocab_->decode(emitted);
  return r;
}

}  // namespace eio
