#include "edge_inference_optimizer/bpe_tokenizer.hpp"

#include <fstream>
#include <limits>
#include <stdexcept>

namespace
{

/// 以 c 开头的 UTF-8 序列长度。非法首字节返回 1,保证遍历能终止。
size_t utf8_len(unsigned char c)
{
  if (c < 0x80) { return 1; }
  if ((c & 0xE0) == 0xC0) { return 2; }
  if ((c & 0xF0) == 0xE0) { return 3; }
  if ((c & 0xF8) == 0xF0) { return 4; }
  return 1;
}

bool is_ascii_space(unsigned char c)
{
  return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

bool is_ascii_alnum(unsigned char c)
{
  return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

}  // namespace

namespace eio
{

BpeTokenizer::BpeTokenizer(const std::string & vocab_path, const std::string & merges_path)
{
  {
    std::ifstream f(vocab_path, std::ios::binary);
    if (!f) { throw std::runtime_error("cannot open vocab: " + vocab_path); }
    uint32_t n = 0;
    f.read(reinterpret_cast<char *>(&n), 4);
    if (!f) { throw std::runtime_error("vocab header truncated: " + vocab_path); }
    id_to_bytes_.resize(n);
    for (uint32_t i = 0; i < n; ++i) {
      uint32_t len = 0;
      f.read(reinterpret_cast<char *>(&len), 4);
      if (!f) { throw std::runtime_error("vocab truncated at " + std::to_string(i)); }
      std::string s(len, '\0');
      if (len) { f.read(s.data(), len); }
      // 特殊 token 导出时已置空;空串不建反查,否则所有空查询都会命中 token 0
      if (!s.empty()) { bytes_to_id_.emplace(s, static_cast<int64_t>(i)); }
      id_to_bytes_[i] = std::move(s);
    }
  }
  {
    std::ifstream f(merges_path, std::ios::binary);
    if (!f) { throw std::runtime_error("cannot open merges: " + merges_path); }
    uint32_t n = 0;
    f.read(reinterpret_cast<char *>(&n), 4);
    if (!f) { throw std::runtime_error("merges header truncated: " + merges_path); }
    merge_rank_.reserve(n * 2);
    for (uint32_t i = 0; i < n; ++i) {
      auto read_piece = [&f](std::string & out) {
          uint32_t len = 0;
          f.read(reinterpret_cast<char *>(&len), 4);
          out.assign(len, '\0');
          if (len) { f.read(out.data(), len); }
        };
      std::string a, b;
      read_piece(a);
      read_piece(b);
      if (!f) { throw std::runtime_error("merges truncated at " + std::to_string(i)); }
      // rank 就是优先级,靠前的规则先合并。丢掉这个顺序就会按任意次序合并,
      // 分词结果不同但不会报错。
      merge_rank_.emplace(a + '\x01' + b, static_cast<int>(i));
    }
  }
}

const std::string & BpeTokenizer::token_bytes(int64_t id) const
{
  if (id < 0 || static_cast<size_t>(id) >= id_to_bytes_.size()) { return empty_; }
  return id_to_bytes_[static_cast<size_t>(id)];
}

std::string BpeTokenizer::decode(const std::vector<int64_t> & ids) const
{
  std::string out;
  for (int64_t id : ids) { out += token_bytes(id); }
  return out;
}

std::vector<std::string> BpeTokenizer::split_words(const std::string & text)
{
  // 汉字必须成片切,按字切开 BPE 合并不了词,模型照样答、只是质量变差。
  std::vector<std::string> out;
  size_t i = 0;
  while (i < text.size()) {
    std::string piece;
    if (is_ascii_space(static_cast<unsigned char>(text[i]))) {
      // 连续空格归属后文,只有最后一个空格并入下一个 token(GPT-2 的约定)
      size_t j = i;
      while (j < text.size() && is_ascii_space(static_cast<unsigned char>(text[j]))) { ++j; }
      if (j - i > 1) {
        out.emplace_back(text.substr(i, j - i - 1));
        i = j - 1;
      }
      piece += text[i++];
      if (i >= text.size()) { out.push_back(piece); break; }
    }
    const auto c = static_cast<unsigned char>(text[i]);
    if (is_ascii_alnum(c)) {
      while (i < text.size() && is_ascii_alnum(static_cast<unsigned char>(text[i]))) {
        piece += text[i++];
      }
    } else if (c >= 0x80) {
      // 连续非 ASCII 保持在一起,BPE 才能合成词
      while (i < text.size() && static_cast<unsigned char>(text[i]) >= 0x80) {
        const size_t n = utf8_len(static_cast<unsigned char>(text[i]));
        piece += text.substr(i, n);
        i += n;
      }
    } else {
      piece += text[i++];   // 单个 ASCII 标点
    }
    out.push_back(piece);
  }
  return out;
}

std::vector<std::string> BpeTokenizer::bpe(const std::string & word) const
{
  // 从单字节起步:词表是 byte-level 的,任何输入都可表示,不会因生僻字失败
  std::vector<std::string> parts;
  parts.reserve(word.size());
  for (char c : word) { parts.emplace_back(1, c); }

  while (parts.size() > 1) {
    int best_rank = std::numeric_limits<int>::max();
    size_t best_i = 0;
    bool found = false;
    for (size_t i = 0; i + 1 < parts.size(); ++i) {
      auto it = merge_rank_.find(parts[i] + '\x01' + parts[i + 1]);
      if (it != merge_rank_.end() && it->second < best_rank) {
        best_rank = it->second;
        best_i = i;
        found = true;
      }
    }
    if (!found) { break; }
    parts[best_i] += parts[best_i + 1];
    parts.erase(parts.begin() + static_cast<std::ptrdiff_t>(best_i) + 1);
  }
  return parts;
}

std::vector<int64_t> BpeTokenizer::encode(const std::string & text) const
{
  std::vector<int64_t> ids;
  for (const auto & word : split_words(text)) {
    for (const auto & piece : bpe(word)) {
      auto it = bytes_to_id_.find(piece);
      if (it != bytes_to_id_.end()) {
        ids.push_back(it->second);
      } else {
        // byte-level 词表下不该走到这里,兜底逐字节编码而不是悄悄丢字
        for (char c : piece) {
          auto b = bytes_to_id_.find(std::string(1, c));
          if (b != bytes_to_id_.end()) { ids.push_back(b->second); }
        }
      }
    }
  }
  return ids;
}

}  // namespace eio
