// 不经过 ROS 的转写工具,用来单独查 ASR。

#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "edge_inference_optimizer/whisper_asr.hpp"

namespace
{

bool read_wav(const std::string & path, std::vector<float> & out, uint32_t & rate)
{
  std::ifstream f(path, std::ios::binary);
  if (!f) { return false; }
  char id[4];
  uint32_t sz = 0;
  f.read(id, 4); f.read(reinterpret_cast<char *>(&sz), 4); f.read(id, 4);

  uint16_t fmt = 0, ch = 0, bits = 0;
  while (f && f.peek() != EOF) {
    char cid[4];
    uint32_t csz = 0;
    f.read(cid, 4);
    f.read(reinterpret_cast<char *>(&csz), 4);
    if (!f) { break; }
    if (std::string(cid, 4) == "fmt ") {
      std::vector<char> b(csz);
      f.read(b.data(), csz);
      std::memcpy(&fmt, b.data(), 2);
      std::memcpy(&ch, b.data() + 2, 2);
      std::memcpy(&rate, b.data() + 4, 4);
      std::memcpy(&bits, b.data() + 14, 2);
    } else if (std::string(cid, 4) == "data") {
      std::vector<char> b(csz);
      f.read(b.data(), csz);
      const size_t got = static_cast<size_t>(f.gcount());
      if (fmt == 1 && bits == 16) {
        const size_t n = got / 2;
        out.resize(n / (ch ? ch : 1));
        for (size_t i = 0; i < out.size(); ++i) {
          int32_t acc = 0;
          for (uint16_t c = 0; c < (ch ? ch : 1); ++c) {
            int16_t s;
            std::memcpy(&s, b.data() + (i * (ch ? ch : 1) + c) * 2, 2);
            acc += s;
          }
          out[i] = static_cast<float>(acc) / (ch ? ch : 1) / 32768.0f;
        }
        return true;
      }
      return false;
    } else {
      f.seekg(csz + (csz & 1), std::ios::cur);
    }
  }
  return false;
}

}  // namespace

int main(int argc, char ** argv)
{
  std::string model_dir, wav, language = "zh";
  int threads = 4, max_tokens = 64;
  bool cuda = false;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    auto nx = [&]() { return (i + 1 < argc) ? argv[++i] : ""; };
    if (k == "--model-dir") { model_dir = nx(); } else if (k == "--wav") { wav = nx(); } else if (
      k == "--threads") { threads = std::stoi(nx()); } else if (k == "--max-tokens") {
      max_tokens = std::stoi(nx());
    } else if (k == "--language") { language = nx(); } else if (k == "--cuda") { cuda = true; }
  }
  if (model_dir.empty() || wav.empty()) {
    std::cerr << "usage: asr_cli --model-dir DIR --wav FILE "
              << "[--language zh|en] [--threads N] [--cuda]\n";
    return 2;
  }

  std::vector<float> pcm;
  uint32_t rate = 0;
  if (!read_wav(wav, pcm, rate)) {
    std::cerr << "cannot read " << wav << '\n';
    return 1;
  }
  std::cout << "audio    : " << pcm.size() << " samples @ " << rate << " Hz ("
            << std::fixed << std::setprecision(2)
            << static_cast<double>(pcm.size()) / (rate ? rate : 1) << " s)\n";

  eio::WhisperAsr::Options o;
  o.encoder_path = model_dir + "/encoder_model.onnx";
  o.decoder_path = model_dir + "/decoder_model.onnx";
  o.decoder_with_past_path = model_dir + "/decoder_with_past_model.onnx";
  o.mel_filters_path = model_dir + "/mel_filters.bin";
  o.vocab_path = model_dir + "/vocab.bin";
  o.backend = cuda ? eio::Backend::kCuda : eio::Backend::kCpu;
  o.threads = threads;
  o.max_tokens = max_tokens;
  // 中文音频硬指定 <|en|> 时 Whisper 会翻译成英文,看起来像转写成功。
  o.language_token = (language == "en") ? eio::WhisperAsr::kEn : eio::WhisperAsr::kZh;
  std::cout << "language : " << language << '\n';

  eio::WhisperAsr asr(std::move(o));
  const auto r = asr.transcribe(pcm);

  std::cout << "mel      : " << r.preprocess_ms << " ms\n"
            << "encoder  : " << r.encoder_ms << " ms\n"
            << "decoder  : " << r.decoder_ms << " ms (" << r.decoded_tokens << " tokens)\n"
            << "tokens   : ";
  for (auto t : r.tokens) { std::cout << t << ' '; }
  std::cout << "\ntext     : \"" << r.text << "\"\n";
  return 0;
}
