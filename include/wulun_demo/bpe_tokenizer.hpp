// Qwen 分词。词表用 scripts/export_qwen.py 导出来。

#ifndef WULUN_DEMO__BPE_TOKENIZER_HPP_
#define WULUN_DEMO__BPE_TOKENIZER_HPP_

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace eio
{

class BpeTokenizer
{
public:
  /// 词表坏了就抛错。
  BpeTokenizer(const std::string & vocab_path, const std::string & merges_path);

  std::vector<int64_t> encode(const std::string & text) const;
  std::string decode(const std::vector<int64_t> & ids) const;

  /// id 对应的字节。特殊 token 和越界返回空。
  const std::string & token_bytes(int64_t id) const;

  size_t vocab_size() const { return id_to_bytes_.size(); }
  size_t merge_count() const { return merge_rank_.size(); }

private:
  /// 先切开，再在每段里面合并。
  static std::vector<std::string> split_words(const std::string & text);
  /// 按优先级合并。
  std::vector<std::string> bpe(const std::string & word) const;

  std::vector<std::string> id_to_bytes_;
  std::unordered_map<std::string, int64_t> bytes_to_id_;
  /// 合并规则的 key。
  std::unordered_map<std::string, int> merge_rank_;
  std::string empty_;
};

}  // namespace eio

#endif  // WULUN_DEMO__BPE_TOKENIZER_HPP_
