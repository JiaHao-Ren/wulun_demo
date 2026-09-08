#include "wulun_demo/qwen_chat.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include "wulun_demo/latency_probe.hpp"

namespace eio
{

namespace
{
/// 该模型的 KV cache 是 float32,形状 [1, 2, seq, 64]
constexpr int64_t kKvHeads = 2;
constexpr int64_t kHeadDim = 64;
}  // namespace

QwenChat::QwenChat(Options opts)
: opts_(std::move(opts)), rng_state_(opts_.seed ? opts_.seed : 1)
{
  tok_ = std::make_unique<BpeTokenizer>(opts_.vocab_path, opts_.merges_path);

  OnnxEngine::Options eo;
  eo.model_path = opts_.model_path;
  eo.backend = opts_.backend;
  eo.intra_op_threads = opts_.threads;
  eo.inter_op_threads = opts_.threads;
  eo.log_tag = "qwen";
  engine_ = std::make_unique<OnnxEngine>(eo);

  // 层数从模型里读,不写死 24,换别的 Qwen 尺寸无需改代码
  for (const auto & in : engine_->inputs()) {
    if (in.name.rfind("past_key_values.", 0) == 0) { ++n_layers_; }
  }
  n_layers_ /= 2;   // 每层 key + value
  if (n_layers_ <= 0) {
    throw std::runtime_error("model exposes no past_key_values inputs; "
            "it was exported without a KV cache");
  }

  input_names_ = {"input_ids", "attention_mask", "position_ids"};
  output_names_ = {"logits"};
  for (int i = 0; i < n_layers_; ++i) {
    for (const char * kind : {"key", "value"}) {
      past_names_.push_back("past_key_values." + std::to_string(i) + "." + kind);
      present_names_.push_back("present." + std::to_string(i) + "." + kind);
    }
  }
  input_names_.insert(input_names_.end(), past_names_.begin(), past_names_.end());
  output_names_.insert(output_names_.end(), present_names_.begin(), present_names_.end());

  reset();
}

QwenChat::~QwenChat() = default;

void QwenChat::reset()
{
  history_.clear();
  cache_.clear();
}

std::vector<int64_t> QwenChat::build_prompt(const std::string & user_text) const
{
  // ChatML。特殊 token 直接插 id,不要当普通文本编码。
  std::vector<int64_t> ids;
  auto put_text = [&](const std::string & s) {
      const auto t = tok_->encode(s);
      ids.insert(ids.end(), t.begin(), t.end());
    };
  auto put_turn = [&](const std::string & role, const std::string & body) {
      ids.push_back(opts_.im_start);
      put_text(role + "\n" + body);
      ids.push_back(opts_.im_end);
      put_text("\n");
    };

  put_turn("system", opts_.system_prompt);
  for (const auto & [u, a] : history_) {
    put_turn("user", u);
    put_turn("assistant", a);
  }
  put_turn("user", user_text);

  // 打开 assistant 轮次后停下,让模型从这里往下生成
  ids.push_back(opts_.im_start);
  put_text("assistant\n");
  return ids;
}

std::vector<Ort::Value> QwenChat::forward(
  const std::vector<int64_t> & input_ids, int64_t past_len)
{
  const auto cur = static_cast<int64_t>(input_ids.size());
  const int64_t total = past_len + cur;

  std::vector<int64_t> ids = input_ids;
  std::vector<int64_t> mask(static_cast<size_t>(total), 1);
  std::vector<int64_t> pos(static_cast<size_t>(cur));
  // 位置编号接着缓存前缀往下排;从 0 重新开始会让每个生成的 token
  // 都以为自己在句首
  std::iota(pos.begin(), pos.end(), past_len);

  std::vector<Ort::Value> inputs;
  inputs.reserve(3 + past_names_.size());
  inputs.push_back(engine_->make_tensor(ids.data(), ids.size(), {1, cur}));
  inputs.push_back(engine_->make_tensor(mask.data(), mask.size(), {1, total}));
  inputs.push_back(engine_->make_tensor(pos.data(), pos.size(), {1, cur}));

  // 首次前向缓存为空:传零长度张量而不是省略输入,计算图要求这些输入必须存在
  static float dummy = 0.0f;
  if (cache_.empty()) {
    for (size_t i = 0; i < past_names_.size(); ++i) {
      // 指针非空但 count 为 0:即使形状为空,ORT 构造张量时仍会解引用该指针
      inputs.push_back(
        engine_->make_tensor(&dummy, 0, {1, kKvHeads, 0, kHeadDim}));
    }
  } else {
    for (auto & v : cache_) { inputs.push_back(std::move(v)); }
  }

  auto out = engine_->run_named(input_names_, inputs, output_names_);

  // 输出 1..N 是 present.*,作为下一步的缓存
  cache_.clear();
  cache_.reserve(out.size() - 1);
  for (size_t i = 1; i < out.size(); ++i) { cache_.push_back(std::move(out[i])); }

  std::vector<Ort::Value> logits;
  logits.push_back(std::move(out[0]));
  return logits;
}

int64_t QwenChat::pick_token(const float * logits, int64_t vocab)
{
  if (opts_.greedy) {
    int64_t best = 0;
    float best_v = -1e30f;
    for (int64_t v = 0; v < vocab; ++v) {
      if (logits[v] > best_v) { best_v = logits[v]; best = v; }
    }
    return best;
  }

  // top-k + temperature。只做部分排序:对 151936 个 logit 全排序,
  // 每个 token 的开销比模型前向本身还大。
  const int k = std::max(1, std::min(opts_.top_k, static_cast<int>(vocab)));
  std::vector<int64_t> idx(static_cast<size_t>(vocab));
  std::iota(idx.begin(), idx.end(), 0);
  std::partial_sort(
    idx.begin(), idx.begin() + k, idx.end(),
    [logits](int64_t a, int64_t b) { return logits[a] > logits[b]; });

  const float t = std::max(1e-3f, opts_.temperature);
  float max_l = logits[idx[0]];
  float sum = 0.0f;
  std::vector<float> probs(static_cast<size_t>(k));
  for (int i = 0; i < k; ++i) {
    probs[static_cast<size_t>(i)] = std::exp((logits[idx[static_cast<size_t>(i)]] - max_l) / t);
    sum += probs[static_cast<size_t>(i)];
  }

  // xorshift64*:给定种子可复现,不用带 <random> 的状态
  rng_state_ ^= rng_state_ >> 12;
  rng_state_ ^= rng_state_ << 25;
  rng_state_ ^= rng_state_ >> 27;
  const auto r = static_cast<float>((rng_state_ * 2685821657736338717ULL) >> 11) /
    static_cast<float>(1ULL << 53);

  float acc = 0.0f;
  const float target = r * sum;
  for (int i = 0; i < k; ++i) {
    acc += probs[static_cast<size_t>(i)];
    if (acc >= target) { return idx[static_cast<size_t>(i)]; }
  }
  return idx[0];
}

QwenChat::Reply QwenChat::chat(const std::string & user_text)
{
  Reply r;
  cache_.clear();   // 每轮重新编码历史,见 Options::history_turns

  const int64_t t0 = steady_ns();
  const auto prompt = build_prompt(user_text);
  r.tokenize_ms = ns_to_ms(steady_ns() - t0);
  r.prompt_tokens = static_cast<int>(prompt.size());

  const int64_t t1 = steady_ns();
  auto logits_v = forward(prompt, 0);
  r.prefill_ms = ns_to_ms(steady_ns() - t1);

  auto info = logits_v.front().GetTensorTypeAndShapeInfo();
  auto shape = info.GetShape();                 // [1, 序列长, 词表大小]
  const int64_t vocab = shape.back();
  const float * lg = logits_v.front().GetTensorData<float>();
  int64_t next = pick_token(lg + (shape[1] - 1) * vocab, vocab);

  std::vector<int64_t> generated;
  const int64_t t2 = steady_ns();
  int64_t past_len = static_cast<int64_t>(prompt.size());

  for (int step = 0; step < opts_.max_new_tokens; ++step) {
    if (next == opts_.im_end || next == opts_.eos) { break; }
    generated.push_back(next);

    auto lv = forward({next}, past_len);
    past_len += 1;

    auto i2 = lv.front().GetTensorTypeAndShapeInfo();
    auto s2 = i2.GetShape();
    const float * l2 = lv.front().GetTensorData<float>();
    next = pick_token(l2 + (s2[1] - 1) * s2.back(), s2.back());
  }
  r.decode_ms = ns_to_ms(steady_ns() - t2);
  r.generated_tokens = static_cast<int>(generated.size());
  r.ms_per_token = r.generated_tokens > 0
    ? r.decode_ms / r.generated_tokens
    : 0.0;

  r.text = tok_->decode(generated);

  history_.emplace_back(user_text, r.text);
  while (static_cast<int>(history_.size()) > std::max(0, opts_.history_turns)) {
    history_.pop_front();
  }
  return r;
}

}  // namespace eio
