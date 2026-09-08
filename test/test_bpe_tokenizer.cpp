// 对照 qwen_probe.txt,检查 C++ BPE 是否和参考 tokenizer 一致。

#include <gtest/gtest.h>

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "edge_inference_optimizer/bpe_tokenizer.hpp"

namespace
{

std::string dir()
{
  const char * e = std::getenv("EIO_MODEL_DIR");
  return (e ? std::string(e) : std::string("/home/nb/edge_ws/models")) + "/qwen05b_onnx";
}

bool load_probe(std::string & text, std::vector<int64_t> & ids)
{
  std::ifstream f(dir() + "/qwen_probe.txt");
  if (!f) { return false; }
  if (!std::getline(f, text)) { return false; }
  std::string line;
  if (!std::getline(f, line)) { return false; }
  std::istringstream is(line);
  int64_t v;
  while (is >> v) { ids.push_back(v); }
  return !ids.empty();
}

bool tables_present()
{
  std::ifstream a(dir() + "/qwen_vocab.bin", std::ios::binary);
  std::ifstream b(dir() + "/qwen_merges.bin", std::ios::binary);
  return a.good() && b.good();
}

}  // namespace

TEST(BpeTokenizer, LoadsTables)
{
  if (!tables_present()) { GTEST_SKIP() << "tokenizer tables not exported"; }
  eio::BpeTokenizer t(dir() + "/qwen_vocab.bin", dir() + "/qwen_merges.bin");
  EXPECT_GT(t.vocab_size(), 150000u);
  EXPECT_GT(t.merge_count(), 100000u);
}

TEST(BpeTokenizer, RejectsMissingTables)
{
  EXPECT_THROW(
    eio::BpeTokenizer("/nonexistent/v.bin", "/nonexistent/m.bin"), std::runtime_error);
}

TEST(BpeTokenizer, MatchesReferenceEncodingOnChinese)
{
  if (!tables_present()) { GTEST_SKIP() << "tokenizer tables not exported"; }
  std::string text;
  std::vector<int64_t> expected;
  if (!load_probe(text, expected)) { GTEST_SKIP() << "qwen_probe.txt missing"; }

  eio::BpeTokenizer t(dir() + "/qwen_vocab.bin", dir() + "/qwen_merges.bin");
  const auto got = t.encode(text);

  std::ostringstream g, e;
  for (auto v : got) { g << v << ' '; }
  for (auto v : expected) { e << v << ' '; }
  EXPECT_EQ(got, expected)
    << "text     : " << text << "\ngot      : " << g.str() << "\nexpected : " << e.str();
}

TEST(BpeTokenizer, RoundTripsChineseText)
{
  if (!tables_present()) { GTEST_SKIP() << "tokenizer tables not exported"; }
  eio::BpeTokenizer t(dir() + "/qwen_vocab.bin", dir() + "/qwen_merges.bin");

  for (const std::string s :
    {"你好", "今天天气怎么样", "我很好,谢谢你", "机器人", "Hello world"})
  {
    EXPECT_EQ(t.decode(t.encode(s)), s) << "round trip failed for: " << s;
  }
}

TEST(BpeTokenizer, SpecialTokensDecodeToNothing)
{
  if (!tables_present()) { GTEST_SKIP() << "tokenizer tables not exported"; }
  eio::BpeTokenizer t(dir() + "/qwen_vocab.bin", dir() + "/qwen_merges.bin");
  // 特殊 token 必须解成空,否则 TTS 会把 chat 标记念出来。
  EXPECT_TRUE(t.token_bytes(151644).empty());
  EXPECT_TRUE(t.token_bytes(151645).empty());
  EXPECT_TRUE(t.decode({151644, 151645}).empty());
}

int main(int argc, char ** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
