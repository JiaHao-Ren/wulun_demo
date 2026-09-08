// 采音，发 /audio_raw。麦克风和文件回放后面一样。

#include <chrono>
#include <cstdint>
#include <cstring>
#include <deque>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "portaudio.h"
#include "rclcpp/rclcpp.hpp"

#include "wulun_demo/latency_probe.hpp"
#include "wulun_demo/msg/audio_chunk.hpp"

namespace
{

struct WavData
{
  std::vector<float> samples;   // 单声道,归一化到 [-1, 1]
  uint32_t sample_rate = 0;
};

std::vector<float> downmix(const std::vector<float> & interleaved, uint16_t channels);

/// 读 PCM16 / float32 WAV。按 chunk 走,不要假定 44 字节头。
bool read_wav(const std::string & path, WavData & out, std::string & err)
{
  std::ifstream f(path, std::ios::binary);
  if (!f) { err = "cannot open " + path; return false; }

  char riff[4], wave[4];
  f.read(riff, 4);
  uint32_t riff_size = 0;
  f.read(reinterpret_cast<char *>(&riff_size), 4);
  f.read(wave, 4);
  if (std::strncmp(riff, "RIFF", 4) != 0 || std::strncmp(wave, "WAVE", 4) != 0) {
    err = "not a RIFF/WAVE file";
    return false;
  }

  uint16_t audio_format = 0, channels = 0, bits = 0;
  uint32_t rate = 0;
  bool have_fmt = false;

  while (f && f.peek() != EOF) {
    char id[4];
    uint32_t size = 0;
    f.read(id, 4);
    f.read(reinterpret_cast<char *>(&size), 4);
    if (!f) { break; }

    if (std::strncmp(id, "fmt ", 4) == 0) {
      std::vector<char> fmt(size);
      f.read(fmt.data(), size);
      if (size >= 16) {
        std::memcpy(&audio_format, fmt.data() + 0, 2);
        std::memcpy(&channels, fmt.data() + 2, 2);
        std::memcpy(&rate, fmt.data() + 4, 4);
        std::memcpy(&bits, fmt.data() + 14, 2);
        have_fmt = true;
      }
    } else if (std::strncmp(id, "data", 4) == 0) {
      if (!have_fmt) { err = "data chunk before fmt chunk"; return false; }
      std::vector<char> raw(size);
      f.read(raw.data(), size);
      const size_t n_read = static_cast<size_t>(f.gcount());

      if (audio_format == 1 && bits == 16) {
        const size_t n = n_read / 2;
        std::vector<float> interleaved(n);
        for (size_t i = 0; i < n; ++i) {
          int16_t s;
          std::memcpy(&s, raw.data() + i * 2, 2);
          interleaved[i] = static_cast<float>(s) / 32768.0f;
        }
        out.samples = downmix(interleaved, channels);
      } else if (audio_format == 3 && bits == 32) {
        const size_t n = n_read / 4;
        std::vector<float> interleaved(n);
        std::memcpy(interleaved.data(), raw.data(), n * 4);
        out.samples = downmix(interleaved, channels);
      } else {
        err = "unsupported WAV format=" + std::to_string(audio_format) +
          " bits=" + std::to_string(bits);
        return false;
      }
      out.sample_rate = rate;
      return true;
    } else {
      f.seekg(size + (size & 1), std::ios::cur);   // chunk 按字对齐
    }
  }
  err = "no data chunk found";
  return false;
}

std::vector<float> downmix(const std::vector<float> & interleaved, uint16_t channels)
{
  if (channels <= 1) { return interleaved; }
  const size_t frames = interleaved.size() / channels;
  std::vector<float> mono(frames);
  for (size_t i = 0; i < frames; ++i) {
    float acc = 0.0f;
    for (uint16_t c = 0; c < channels; ++c) { acc += interleaved[i * channels + c]; }
    mono[i] = acc / static_cast<float>(channels);
  }
  return mono;
}

}  // namespace

class AudioCaptureNode : public rclcpp::Node
{
public:
  AudioCaptureNode()
  : Node("audio_capture")
  {
    // file 回放文件，mic 用麦克风。后面处理一样。
    source_ = declare_parameter<std::string>("source", "file");
    wav_path_ = declare_parameter<std::string>("wav_path", "");
    chunk_ = declare_parameter<int>("chunk_samples", 512);
    loop_ = declare_parameter<bool>("loop", true);
    realtime_ = declare_parameter<bool>("realtime", true);
    lead_in_ms_ = declare_parameter<int>("lead_in_silence_ms", 500);
    tail_ms_ = declare_parameter<int>("tail_silence_ms", 2000);
    const auto topic = declare_parameter<std::string>("output_topic", "/audio_raw");
    // ROS 2 参数一律是 int64,这里推导出来是 long,显式转换避免进到格式化字符串
    const int mic_rate = static_cast<int>(declare_parameter<int>("mic_sample_rate", 16000));

    if (source_ == "mic") {
      start_microphone(mic_rate);
      pub_ = create_publisher<wulun_demo::msg::AudioChunk>(topic, 50);
      timer_ = create_wall_timer(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::duration<double>(static_cast<double>(chunk_) / mic_rate)),
        std::bind(&AudioCaptureNode::tick_mic, this));
      RCLCPP_INFO(
        get_logger(), "publishing %s from MICROPHONE @ %d Hz, %d samples/chunk",
        topic.c_str(), mic_rate, chunk_);
      return;
    }

    std::string err;
    if (!wav_path_.empty() && read_wav(wav_path_, wav_, err)) {
      RCLCPP_INFO(
        get_logger(), "loaded %s: %zu samples @ %u Hz (%.2f s)",
        wav_path_.c_str(), wav_.samples.size(), wav_.sample_rate,
        wav_.sample_rate ? static_cast<double>(wav_.samples.size()) / wav_.sample_rate : 0.0);
    } else {
      if (!wav_path_.empty()) { RCLCPP_ERROR(get_logger(), "%s", err.c_str()); }
      RCLCPP_WARN(
        get_logger(),
        "没有可用的 WAV,将输出静音。VAD 不会触发,对话链路的瀑布图会是空的。"
        "请传入 wav_path:=<文件>。");
      wav_.sample_rate = 16000;
      wav_.samples.assign(wav_.sample_rate * 2, 0.0f);
    }

    // VAD 是按 16 kHz 训的，采样率不对不会报错，结果却是错的。
    if (wav_.sample_rate != 16000) {
      RCLCPP_WARN(
        get_logger(), "sample rate is %u Hz, not the 16000 Hz Silero expects; "
        "VAD probabilities will be unreliable", wav_.sample_rate);
    }

    build_stream();

    pub_ = create_publisher<wulun_demo::msg::AudioChunk>(topic, 50);
    const double period = realtime_
      ? static_cast<double>(chunk_) / static_cast<double>(wav_.sample_rate)
      : 0.001;
    timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(period)),
      std::bind(&AudioCaptureNode::tick, this));

    RCLCPP_INFO(
      get_logger(), "publishing %s, %d samples/chunk, %s",
      topic.c_str(), chunk_, realtime_ ? "real-time rate" : "as fast as possible");
  }

  ~AudioCaptureNode() override
  {
    if (mic_stream_) {
      Pa_StopStream(mic_stream_);
      Pa_CloseStream(mic_stream_);
    }
    if (pa_ready_) { Pa_Terminate(); }
  }

private:
  /// PortAudio 回调里不碰 ROS,只写入缓冲。
  static int mic_callback(
    const void * input, void * /*output*/, unsigned long frames,
    const PaStreamCallbackTimeInfo * /*t*/, PaStreamCallbackFlags /*flags*/, void * user)
  {
    auto * self = static_cast<AudioCaptureNode *>(user);
    if (input) {
      const auto * in = static_cast<const float *>(input);
      std::lock_guard<std::mutex> lk(self->mic_mutex_);
      self->mic_buffer_.insert(self->mic_buffer_.end(), in, in + frames);
      // 限制缓冲上限:消费端卡住时丢最旧的音频而不是无限增长。
      // 对话场景里旧音频没有价值。
      const size_t cap = static_cast<size_t>(self->mic_rate_) * 5;
      while (self->mic_buffer_.size() > cap) { self->mic_buffer_.pop_front(); }
    }
    return paContinue;
  }

  void start_microphone(int rate)
  {
    mic_rate_ = rate;
    const auto err = Pa_Initialize();
    if (err != paNoError) {
      RCLCPP_FATAL(get_logger(), "Pa_Initialize: %s", Pa_GetErrorText(err));
      throw std::runtime_error("PortAudio init failed");
    }
    pa_ready_ = true;

    if (Pa_GetDefaultInputDevice() == paNoDevice) {
      throw std::runtime_error(
              "no default input device. This node needs a microphone; on a headless "
              "server use source:=file with wav_path instead.");
    }

    const auto e2 = Pa_OpenDefaultStream(
      &mic_stream_, 1, 0, paFloat32, rate,
      static_cast<unsigned long>(chunk_), &AudioCaptureNode::mic_callback, this);
    if (e2 != paNoError) {
      throw std::runtime_error(std::string("Pa_OpenDefaultStream: ") + Pa_GetErrorText(e2));
    }
    if (Pa_StartStream(mic_stream_) != paNoError) {
      throw std::runtime_error("Pa_StartStream failed");
    }
  }

  void tick_mic()
  {
    std::vector<float> chunk;
    {
      std::lock_guard<std::mutex> lk(mic_mutex_);
      if (mic_buffer_.size() < static_cast<size_t>(chunk_)) { return; }
      chunk.assign(mic_buffer_.begin(), mic_buffer_.begin() + chunk_);
      mic_buffer_.erase(mic_buffer_.begin(), mic_buffer_.begin() + chunk_);
    }

    wulun_demo::msg::AudioChunk msg;
    msg.header.stamp = this->now();
    msg.header.frame_id = "mic";
    msg.samples = std::move(chunk);
    msg.sample_rate = static_cast<uint32_t>(mic_rate_);
    msg.trace_id = ++seq_;
    msg.capture_ns = eio::steady_ns();
    pub_->publish(msg);
  }

  /// 文件前后补静音,否则 VAD 看不到端点窗口。
  void build_stream()
  {
    const auto sr = wav_.sample_rate ? wav_.sample_rate : 16000;
    const size_t lead = static_cast<size_t>(sr) * lead_in_ms_ / 1000;
    const size_t tail = static_cast<size_t>(sr) * tail_ms_ / 1000;
    stream_.clear();
    stream_.reserve(lead + wav_.samples.size() + tail);
    stream_.insert(stream_.end(), lead, 0.0f);
    stream_.insert(stream_.end(), wav_.samples.begin(), wav_.samples.end());
    stream_.insert(stream_.end(), tail, 0.0f);
  }

  void tick()
  {
    if (pos_ + static_cast<size_t>(chunk_) > stream_.size()) {
      if (!loop_) { return; }
      pos_ = 0;
      ++utterance_;
    }

    wulun_demo::msg::AudioChunk msg;
    msg.header.stamp = this->now();
    msg.header.frame_id = "mic";
    msg.samples.assign(stream_.begin() + pos_, stream_.begin() + pos_ + chunk_);
    msg.sample_rate = wav_.sample_rate;
    msg.trace_id = ++seq_;
    msg.capture_ns = eio::steady_ns();
    pub_->publish(msg);

    pos_ += static_cast<size_t>(chunk_);
  }

  rclcpp::Publisher<wulun_demo::msg::AudioChunk>::SharedPtr pub_;
  rclcpp::TimerBase::SharedPtr timer_;
  WavData wav_;
  std::vector<float> stream_;
  std::string wav_path_, source_;
  size_t pos_ = 0;
  uint64_t seq_ = 0, utterance_ = 0;
  int chunk_ = 512, lead_in_ms_ = 500, tail_ms_ = 2000;
  bool loop_ = true, realtime_ = true;

  PaStream * mic_stream_ = nullptr;
  std::deque<float> mic_buffer_;
  std::mutex mic_mutex_;
  int mic_rate_ = 16000;
  bool pa_ready_ = false;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<AudioCaptureNode>());
  rclcpp::shutdown();
  return 0;
}
