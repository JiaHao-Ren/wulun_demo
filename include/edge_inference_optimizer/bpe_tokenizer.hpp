// Qwen 用的 byte-level BPE。词表来自 scripts/export_qwen.py。

#ifndef EDGE_INFERENCE_OPTIMIZER__BPE_TOKENIZER_HPP_
#define EDGE_INFERENCE_OPTIMIZER__BPE_TOKENIZER_HPP_

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace eio
{

class BpeTokenizer
{
public:
  /// 词表或 merges 文件缺失、损坏时抛 std::runtime_error
  BpeTokenizer(const std::string & vocab_path, const std::string & merges_path);

  std::vector<int64_t> encode(const std::string & text) const;
  std::string decode(const std::vector<int64_t> & ids) const;

  /// 某个 id 对应的字节;特殊 token 和越界 id 返回空串
  const std::string & token_bytes(int64_t id) const;

  size_t vocab_size() const { return id_to_bytes_.size(); }
  size_t merge_count() const { return merge_rank_.size(); }

private:
  /// 先切成 piece,再在每个 piece 内部做 BPE
  static std::vector<std::string> split_words(const std::string & text);
  /// piece 内部按 rank 从小到大贪心合并
  std::vector<std::string> bpe(const std::string & word) const;

  std::vector<std::string> id_to_bytes_;
  std::unordered_map<std::string, int64_t> bytes_to_id_;
  /// key 为 a + '\x01' + b;'\x01' 不会出现在 UTF-8 token 内部
  std::unordered_map<std::string, int> merge_rank_;
  std::string empty_;
};

}  // namespace eio

#endif  // EDGE_INFERENCE_OPTIMIZER__BPE_TOKENIZER_HPP_
