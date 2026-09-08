// 播 /tts_audio。跟合成拆开,没声卡也能跑完整条链路。

#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include "portaudio.h"
#include "rclcpp/rclcpp.hpp"

#include "edge_inference_optimizer/latency_probe.hpp"
#include "edge_inference_optimizer/msg/audio_chunk.hpp"

class AudioPlaybackNode : public rclcpp::Node
{
public:
  AudioPlaybackNode()
  : Node("audio_playback")
  {
    const auto err = Pa_Initialize();
    if (err != paNoError) {
      RCLCPP_FATAL(get_logger(), "Pa_Initialize: %s", Pa_GetErrorText(err));
      throw std::runtime_error("PortAudio init failed");
    }
    pa_ready_ = true;

    const int n = Pa_GetDeviceCount();
    RCLCPP_INFO(get_logger(), "PortAudio: %d devices, default output index %d",
      n, Pa_GetDefaultOutputDevice());
    if (Pa_GetDefaultOutputDevice() == paNoDevice) {
      RCLCPP_WARN(
        get_logger(),
        "未找到默认输出设备,音频将被丢弃。无声卡服务器上属正常现象,"
        "可给 tts_node 设置 dump_dir 把音频存成文件检查。");
    }

    sub_ = create_subscription<edge_inference_optimizer::msg::AudioChunk>(
      "/tts_audio", 5,
      std::bind(&AudioPlaybackNode::on_audio, this, std::placeholders::_1));
    RCLCPP_INFO(get_logger(), "waiting for audio on /tts_audio");
  }

  ~AudioPlaybackNode() override
  {
    close_stream();
    if (pa_ready_) { Pa_Terminate(); }
  }

private:
  void close_stream()
  {
    if (stream_) {
      Pa_StopStream(stream_);
      Pa_CloseStream(stream_);
      stream_ = nullptr;
    }
  }

  bool ensure_stream(int sample_rate)
  {
    if (stream_ && sample_rate == open_rate_) { return true; }
    close_stream();
    if (Pa_GetDefaultOutputDevice() == paNoDevice) { return false; }

    const auto err = Pa_OpenDefaultStream(
      &stream_, 0, 1, paFloat32, sample_rate, paFramesPerBufferUnspecified,
      nullptr, nullptr);
    if (err != paNoError) {
      RCLCPP_ERROR(get_logger(), "Pa_OpenDefaultStream: %s", Pa_GetErrorText(err));
      stream_ = nullptr;
      return false;
    }
    if (Pa_StartStream(stream_) != paNoError) {
      close_stream();
      return false;
    }
    open_rate_ = sample_rate;
    return true;
  }

  void on_audio(const edge_inference_optimizer::msg::AudioChunk::ConstSharedPtr & msg)
  {
    if (msg->samples.empty()) { return; }
    if (!ensure_stream(static_cast<int>(msg->sample_rate))) {
      RCLCPP_WARN_ONCE(get_logger(), "no output device; dropping audio");
      return;
    }

    const int64_t t0 = eio::steady_ns();
    const auto err = Pa_WriteStream(
      stream_, msg->samples.data(), static_cast<unsigned long>(msg->samples.size()));
    const double ms = eio::ns_to_ms(eio::steady_ns() - t0);

    if (err != paNoError && err != paOutputUnderflowed) {
      RCLCPP_ERROR(get_logger(), "Pa_WriteStream: %s", Pa_GetErrorText(err));
      return;
    }
    const double dur = static_cast<double>(msg->samples.size()) / msg->sample_rate;
    // 写操作会阻塞约等于音频时长,这是播放时间不是计算时间,
    // 但确实是人在等待的部分。
    RCLCPP_INFO(get_logger(), "played %.2f s (blocked %.0f ms)", dur, ms);
  }

  rclcpp::Subscription<edge_inference_optimizer::msg::AudioChunk>::SharedPtr sub_;
  PaStream * stream_ = nullptr;
  int open_rate_ = 0;
  bool pa_ready_ = false;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<AudioPlaybackNode>());
  } catch (const std::exception & e) {
    RCLCPP_FATAL(rclcpp::get_logger("audio_playback"), "fatal: %s", e.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
