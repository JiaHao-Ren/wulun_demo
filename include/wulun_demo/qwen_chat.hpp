// 本地 Qwen 对话。

#ifndef WULUN_DEMO__QWEN_CHAT_HPP_
#define WULUN_DEMO__QWEN_CHAT_HPP_

#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "wulun_demo/bpe_tokenizer.hpp"
#include "wulun_demo/onnx_engine.hpp"

namespace eio
{

class QwenChat
{
public:
  struct Options
  {
    std::string model_path;
    std::string vocab_path;
    std::string merges_path;
    Backend backend = Backend::kCpu;
    int threads = 4;

    /// 回复最长多少 token。
    int max_new_tokens = 64;

    /// 记住几轮对话。
    int history_turns = 3;

    std::string system_prompt =
      "你是一个机器人助手。请用简短、自然的中文口语回答,一到两句话即可。";

    int64_t im_start = 151644;
    int64_t im_end = 151645;
    int64_t eos = 151643;

    // 默认贪心，同样的问题给同样的答案。
    bool greedy = true;
    float temperature = 0.7f;
    int top_k = 20;
    uint64_t seed = 20260907;
  };

  struct Reply
  {
    std::string text;
    int prompt_tokens = 0;
    int generated_tokens = 0;
    double tokenize_ms = 0.0;
    double prefill_ms = 0.0;
    double decode_ms = 0.0;
    double ms_per_token = 0.0;
  };

  explicit QwenChat(Options opts);
  ~QwenChat();

  /// 一轮对话。
  Reply chat(const std::string & user_text);

  /// 清空历史。
  void reset();

  const OnnxEngine & engine() const { return *engine_; }

private:
  /// 拼 prompt。
  std::vector<int64_t> build_prompt(const std::string & user_text) const;
  /// 一次前向。
  std::vector<Ort::Value> forward(
    const std::vector<int64_t> & input_ids, int64_t past_len);
  int64_t pick_token(const float * logits, int64_t vocab);

  Options opts_;
  std::unique_ptr<BpeTokenizer> tok_;
  std::unique_ptr<OnnxEngine> engine_;

  int n_layers_ = 0;
  std::vector<std::string> past_names_;      ///< past_key_values.{i}.{key,value}
  std::vector<std::string> present_names_;   ///< present.{i}.{key,value}
  std::vector<std::string> input_names_;     ///< ids, mask, position + past
  std::vector<std::string> output_names_;    ///< logits + present

  std::vector<Ort::Value> cache_;            ///< 当前的 KV cache
  std::deque<std::pair<std::string, std::string>> history_;   ///< (用户, 助手)
  uint64_t rng_state_ = 0;
};

}  // namespace eio

#endif  // WULUN_DEMO__QWEN_CHAT_HPP_
