// Qwen2.5-0.5B 本地对话,带 KV cache。

#ifndef EDGE_INFERENCE_OPTIMIZER__QWEN_CHAT_HPP_
#define EDGE_INFERENCE_OPTIMIZER__QWEN_CHAT_HPP_

#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "edge_inference_optimizer/bpe_tokenizer.hpp"
#include "edge_inference_optimizer/onnx_engine.hpp"

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

    /// 回复长度上限。对话机器人上短回复是优点——用户在等,
    /// 200 token 的独白再好也不如 20 token 的答案。
    int max_new_tokens = 64;

    /// 保留多少轮历史。每保留一轮,下次 prefill 都要重新编码一遍,
    /// 是上下文与延迟的直接权衡。
    int history_turns = 3;

    std::string system_prompt =
      "你是一个机器人助手。请用简短、自然的中文口语回答,一到两句话即可。";

    int64_t im_start = 151644;
    int64_t im_end = 151645;
    int64_t eos = 151643;

    // 默认贪心解码:可复现,同一问题给同一答案,便于对比和演示
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
    double prefill_ms = 0.0;   ///< 对整个 prompt 的一次前向
    double decode_ms = 0.0;    ///< 自回归循环,所有步累加
    double ms_per_token = 0.0;
  };

  explicit QwenChat(Options opts);
  ~QwenChat();

  /// 一轮对话,成功后追加到内部历史
  Reply chat(const std::string & user_text);

  /// 清空对话历史(开始新会话)
  void reset();

  const OnnxEngine & engine() const { return *engine_; }

private:
  /// 把 system + 历史 + 当前用户输入拼成 token id 序列
  std::vector<int64_t> build_prompt(const std::string & user_text) const;
  /// 一次前向,past_len 是进入时的缓存长度
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

#endif  // EDGE_INFERENCE_OPTIMIZER__QWEN_CHAT_HPP_
