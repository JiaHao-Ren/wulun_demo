// 检查音频前端数值对不对。

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "wulun_demo/whisper_asr.hpp"

namespace
{

std::string dir()
{
  const char * e = std::getenv("EIO_MODEL_DIR");
  return (e ? std::string(e) : std::string("/home/nb/edge_ws/models")) + "/whisper_tiny_onnx";
}

bool read_floats(const std::string & path, std::vector<float> & out)
{
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) { return false; }
  const auto bytes = static_cast<size_t>(f.tellg());
  if (bytes == 0 || bytes % sizeof(float) != 0) { return false; }
  f.seekg(0);
  out.resize(bytes / sizeof(float));
  f.read(reinterpret_cast<char *>(out.data()), static_cast<std::streamsize>(bytes));
  return static_cast<size_t>(f.gcount()) == bytes;
}

}  // namespace

TEST(MelExtractor, RejectsMissingFilterFile)
{
  EXPECT_THROW(eio::MelExtractor("/nonexistent/mel_filters.bin"), std::runtime_error);
}

TEST(MelExtractor, MatchesReferenceImplementation)
{
  std::vector<float> audio, reference;
  if (!read_floats(dir() + "/audio_reference.bin", audio) ||
    !read_floats(dir() + "/mel_reference.bin", reference))
  {
    GTEST_SKIP() << "reference tensors not present in " << dir();
  }

  eio::MelExtractor mel(dir() + "/mel_filters.bin");
  const auto got = mel.compute(audio);

  ASSERT_EQ(got.size(), reference.size())
    << "expected " << eio::WhisperAudioSpec::kNMels << "x"
    << eio::WhisperAudioSpec::kNFrames;

  double max_abs = 0.0, sum_abs = 0.0;
  size_t worst = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    const double d = std::abs(static_cast<double>(got[i]) - reference[i]);
    sum_abs += d;
    if (d > max_abs) { max_abs = d; worst = i; }
  }
  const double mean_abs = sum_abs / static_cast<double>(got.size());

  EXPECT_LT(mean_abs, 1e-4) << "mean |diff| too large";
  EXPECT_LT(max_abs, 2e-3)
    << "max |diff| at index " << worst
    << " (mel bin " << worst / eio::WhisperAudioSpec::kNFrames
    << ", frame " << worst % eio::WhisperAudioSpec::kNFrames << ")"
    << ": got " << got[worst] << " vs " << reference[worst];
}

TEST(WhisperVocab, DecodesToBytesAndSkipsSpecials)
{
  std::vector<float> probe;
  std::ifstream check(dir() + "/vocab.bin", std::ios::binary);
  if (!check) { GTEST_SKIP() << "vocab.bin not present"; }
  check.close();

  eio::WhisperVocab v(dir() + "/vocab.bin");
  EXPECT_GT(v.size(), 50000u);

  const std::string specials = v.decode(
    {eio::WhisperAsr::kSot, eio::WhisperAsr::kEn,
      eio::WhisperAsr::kTranscribe, eio::WhisperAsr::kNoTimestamps});
  EXPECT_TRUE(specials.empty()) << "got: '" << specials << "'";

  EXPECT_NO_THROW(v.decode({-1, 999999999}));
}

int main(int argc, char ** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
