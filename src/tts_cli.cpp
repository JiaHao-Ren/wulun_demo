// 文本进、WAV 出。没声卡也能查 TTS。

#include <iomanip>
#include <iostream>
#include <string>

#include "edge_inference_optimizer/piper_tts.hpp"

int main(int argc, char ** argv)
{
  std::string model_dir, espeak_lib, espeak_data, text, out = "/tmp/tts_out.wav";
  std::string voice = "cmn";
  int threads = 4;
  float length_scale = 1.0f;
  bool cuda = false;

  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    auto nx = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
    if (k == "--model-dir") { model_dir = nx(); } else if (k == "--espeak-lib") {
      espeak_lib = nx();
    } else if (k == "--espeak-data") { espeak_data = nx(); } else if (k == "--text") {
      text = nx();
    } else if (k == "--out") { out = nx(); } else if (k == "--voice") { voice = nx(); } else if (
      k == "--threads") { threads = std::stoi(nx()); } else if (k == "--length-scale") {
      length_scale = std::stof(nx());
    } else if (k == "--cuda") { cuda = true; }
  }
  if (model_dir.empty() || text.empty() || espeak_lib.empty() || espeak_data.empty()) {
    std::cerr << "usage: tts_cli --model-dir DIR --espeak-lib SO --espeak-data DIR "
              << "--text \"你好\" [--out FILE] [--length-scale 1.0]\n";
    return 2;
  }

  eio::PiperTts::Options o;
  o.model_path = model_dir + "/zh_CN-huayan-medium.onnx";
  o.phoneme_map_path = model_dir + "/phoneme_map.bin";
  o.espeak_lib = espeak_lib;
  o.espeak_data = espeak_data;
  o.voice = voice;
  o.threads = threads;
  o.length_scale = length_scale;
  o.backend = cuda ? eio::Backend::kCuda : eio::Backend::kCpu;

  try {
    eio::PiperTts tts(std::move(o));
    const auto a = tts.synthesize(text);
    if (a.samples.empty()) {
      std::cerr << "synthesis produced no audio (phonemisation may have failed)\n";
      return 1;
    }
    if (!eio::PiperTts::write_wav(out, a.samples, a.sample_rate)) {
      std::cerr << "cannot write " << out << '\n';
      return 1;
    }
    std::cout << std::fixed << std::setprecision(2)
              << "text      : " << text << '\n'
              << "phonemes  : " << a.phoneme_count << '\n'
              << "phonemize : " << a.phonemize_ms << " ms\n"
              << "synthesize: " << a.infer_ms << " ms\n"
              << "audio     : " << a.duration_s() << " s @ " << a.sample_rate << " Hz\n"
              << "RTF       : " << (a.infer_ms / 1000.0) / std::max(1e-6, a.duration_s()) << '\n'
              << "wrote     : " << out << '\n';
  } catch (const std::exception & e) {
    std::cerr << "fatal: " << e.what() << '\n';
    return 1;
  }
  return 0;
}
