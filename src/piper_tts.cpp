#include "wulun_demo/piper_tts.hpp"

#include <dlfcn.h>

#include <cstring>
#include <fstream>
#include <stdexcept>

#include "wulun_demo/latency_probe.hpp"

namespace eio
{

namespace
{

// espeak-ng 的 C ABI,手写声明(原因见头文件)。常量取自 speak_lib.h。
constexpr int kAudioOutputSynchronous = 0x02;
constexpr int kCharsUtf8 = 1;
constexpr int kPhonemesIpa = 0x02;

using FnInitialize = int (*)(int, int, const char *, int);
using FnSetVoiceByName = int (*)(const char *);
using FnTextToPhonemes = const char * (*)(const void **, int, int);
using FnTerminate = int (*)();

/// 按码点拆分 UTF-8 字符串。
/// piper 的 phoneme_id_map 就是按码点建的,多字符 IPA 序列拆成各自的码点分别给 id。
std::vector<std::string> utf8_codepoints(const std::string & s)
{
  std::vector<std::string> out;
  size_t i = 0;
  while (i < s.size()) {
    const auto c = static_cast<unsigned char>(s[i]);
    size_t n = 1;
    if ((c & 0xE0) == 0xC0) { n = 2; } else if ((c & 0xF0) == 0xE0) { n = 3; } else if (
      (c & 0xF8) == 0xF0) { n = 4; }
    if (i + n > s.size()) { n = 1; }
    out.push_back(s.substr(i, n));
    i += n;
  }
  return out;
}

}  // namespace


struct EspeakPhonemizer::Impl
{
  void * handle = nullptr;
  FnTextToPhonemes text_to_phonemes = nullptr;
  FnTerminate terminate = nullptr;

  ~Impl()
  {
    if (terminate) { terminate(); }
    if (handle) { dlclose(handle); }
  }
};

EspeakPhonemizer::EspeakPhonemizer(
  const std::string & lib_path, const std::string & data_path, const std::string & voice)
: impl_(std::make_unique<Impl>())
{
  impl_->handle = dlopen(lib_path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!impl_->handle) {
    throw std::runtime_error("dlopen failed for " + lib_path + ": " + dlerror());
  }

  auto sym = [&](const char * name) -> void * {
      dlerror();
      void * p = dlsym(impl_->handle, name);
      const char * err = dlerror();
      if (err) { throw std::runtime_error(std::string("dlsym ") + name + ": " + err); }
      return p;
    };

  auto initialize = reinterpret_cast<FnInitialize>(sym("espeak_Initialize"));
  auto set_voice = reinterpret_cast<FnSetVoiceByName>(sym("espeak_SetVoiceByName"));
  impl_->text_to_phonemes = reinterpret_cast<FnTextToPhonemes>(sym("espeak_TextToPhonemes"));
  impl_->terminate = reinterpret_cast<FnTerminate>(sym("espeak_Terminate"));

  // 成功返回采样率,失败返回负数。最常见的原因是 data 路径不对,
  // 不拦住的话后面只会得到静音而不是报错。
  const int rc = initialize(kAudioOutputSynchronous, 0, data_path.c_str(), 0);
  if (rc < 0) {
    throw std::runtime_error("espeak_Initialize failed (data path: " + data_path + ")");
  }
  if (set_voice(voice.c_str()) != 0) {
    throw std::runtime_error("espeak_SetVoiceByName failed for voice '" + voice + "'");
  }
}

EspeakPhonemizer::~EspeakPhonemizer() = default;

std::vector<std::string> EspeakPhonemizer::phonemize(const std::string & text) const
{
  std::vector<std::string> out;
  if (text.empty()) { return out; }

  // espeak 每次调用推进 ptr 并返回一个小句
  const char * ptr = text.c_str();
  const void * vptr = ptr;
  while (vptr != nullptr) {
    const char * phonemes = impl_->text_to_phonemes(&vptr, kCharsUtf8, kPhonemesIpa);
    if (!phonemes) { break; }
    for (auto & cp : utf8_codepoints(phonemes)) { out.push_back(cp); }
  }
  return out;
}


PiperTts::PiperTts(Options opts)
: opts_(std::move(opts))
{
  {
    std::ifstream f(opts_.phoneme_map_path, std::ios::binary);
    if (!f) { throw std::runtime_error("cannot open " + opts_.phoneme_map_path); }
    uint32_t n = 0;
    f.read(reinterpret_cast<char *>(&n), 4);
    for (uint32_t i = 0; i < n; ++i) {
      uint32_t len = 0;
      f.read(reinterpret_cast<char *>(&len), 4);
      std::string k(len, '\0');
      if (len) { f.read(k.data(), len); }
      int32_t id = 0;
      f.read(reinterpret_cast<char *>(&id), 4);
      if (!f) { throw std::runtime_error("phoneme map truncated"); }
      phoneme_ids_.emplace(std::move(k), id);
    }
  }

  phonemizer_ = std::make_unique<EspeakPhonemizer>(
    opts_.espeak_lib, opts_.espeak_data, opts_.voice);

  OnnxEngine::Options eo;
  eo.model_path = opts_.model_path;
  eo.backend = opts_.backend;
  eo.intra_op_threads = opts_.threads;
  eo.inter_op_threads = opts_.threads;
  eo.log_tag = "piper";
  engine_ = std::make_unique<OnnxEngine>(eo);
}

PiperTts::~PiperTts() = default;

PiperTts::Audio PiperTts::synthesize(const std::string & text)
{
  Audio a;
  a.sample_rate = sample_rate_;

  const int64_t t0 = steady_ns();
  const auto phonemes = phonemizer_->phonemize(text);
  a.phonemize_ms = ns_to_ms(steady_ns() - t0);
  if (phonemes.empty()) { return a; }

  // VITS: BOS + 音素夹 pad + EOS。漏 pad 不会报错,语速会乱。
  const int64_t bos = phoneme_ids_.count("^") ? phoneme_ids_.at("^") : 1;
  const int64_t eos = phoneme_ids_.count("$") ? phoneme_ids_.at("$") : 2;
  const int64_t pad = phoneme_ids_.count("_") ? phoneme_ids_.at("_") : 0;

  std::vector<int64_t> ids;
  ids.reserve(phonemes.size() * 2 + 2);
  ids.push_back(bos);
  ids.push_back(pad);
  for (const auto & p : phonemes) {
    auto it = phoneme_ids_.find(p);
    if (it == phoneme_ids_.end()) { continue; }   // 表里没有的符号直接丢弃
    ids.push_back(it->second);
    ids.push_back(pad);
  }
  ids.push_back(eos);
  a.phoneme_count = static_cast<int>(phonemes.size());

  std::vector<int64_t> lengths{static_cast<int64_t>(ids.size())};
  std::vector<float> scales{opts_.noise_scale, opts_.length_scale, opts_.noise_w};

  const int64_t t1 = steady_ns();
  std::vector<Ort::Value> inputs;
  inputs.push_back(
    engine_->make_tensor(ids.data(), ids.size(), {1, static_cast<int64_t>(ids.size())}));
  inputs.push_back(engine_->make_tensor(lengths.data(), 1, {1}));
  inputs.push_back(engine_->make_tensor(scales.data(), 3, {3}));

  auto out = engine_->run_named(
    {"input", "input_lengths", "scales"}, inputs, {"output"});
  a.infer_ms = ns_to_ms(steady_ns() - t1);

  auto info = out.front().GetTensorTypeAndShapeInfo();
  const size_t n = info.GetElementCount();
  const float * p = out.front().GetTensorData<float>();
  a.samples.assign(p, p + n);
  return a;
}

bool PiperTts::write_wav(
  const std::string & path, const std::vector<float> & samples, int sample_rate)
{
  std::ofstream f(path, std::ios::binary);
  if (!f) { return false; }

  const uint32_t data_bytes = static_cast<uint32_t>(samples.size() * 2);
  const uint32_t riff_size = 36 + data_bytes;
  const uint16_t channels = 1, bits = 16, fmt = 1;
  const uint32_t byte_rate = static_cast<uint32_t>(sample_rate) * channels * bits / 8;
  const uint16_t block_align = channels * bits / 8;
  const uint32_t fmt_size = 16;

  auto w32 = [&f](uint32_t v) { f.write(reinterpret_cast<const char *>(&v), 4); };
  auto w16 = [&f](uint16_t v) { f.write(reinterpret_cast<const char *>(&v), 2); };

  f.write("RIFF", 4); w32(riff_size); f.write("WAVE", 4);
  f.write("fmt ", 4); w32(fmt_size); w16(fmt); w16(channels);
  w32(static_cast<uint32_t>(sample_rate)); w32(byte_rate); w16(block_align); w16(bits);
  f.write("data", 4); w32(data_bytes);

  for (float v : samples) {
    // 先夹紧再转定点:VITS 在爆破音处可能超出 [-1, 1],
    // 直接截断会溢出成一声爆响
    const float c = v < -1.0f ? -1.0f : (v > 1.0f ? 1.0f : v);
    const auto s = static_cast<int16_t>(c * 32767.0f);
    f.write(reinterpret_cast<const char *>(&s), 2);
  }
  return f.good();
}

}  // namespace eio
