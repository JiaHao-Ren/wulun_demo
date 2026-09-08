// TTS,订 /llm_reply,发 /tts_audio。自己不播。

#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"

#include "edge_inference_optimizer/latency_probe.hpp"
#include "edge_inference_optimizer/msg/audio_chunk.hpp"
#include "edge_inference_optimizer/msg/llm_reply.hpp"
#include "edge_inference_optimizer/piper_tts.hpp"
#include "edge_inference_optimizer/ros_trace.hpp"

class TtsNode : public rclcpp::Node
{
public:
  TtsNode()
  : Node("tts_node")
  {
    const auto dir = declare_parameter<std::string>("model_dir", "");
    const auto lib = declare_parameter<std::string>("espeak_lib", "");
    const auto data = declare_parameter<std::string>("espeak_data", "");
    dump_dir_ = declare_parameter<std::string>("dump_dir", "");

    if (dir.empty() || lib.empty() || data.empty()) {
      RCLCPP_FATAL(
        get_logger(), "parameters 'model_dir', 'espeak_lib' and 'espeak_data' are required");
      throw std::runtime_error("tts paths not set");
    }

    eio::PiperTts::Options o;
    o.model_path = dir + "/zh_CN-huayan-medium.onnx";
    o.phoneme_map_path = dir + "/phoneme_map.bin";
    o.espeak_lib = lib;
    o.espeak_data = data;
    o.voice = declare_parameter<std::string>("voice", "cmn");
    o.threads = declare_parameter<int>("threads", 4);
    // 大于 1 语速变慢。稍慢一点更容易听清,也能盖掉偶尔的发音瑕疵。
    o.length_scale = static_cast<float>(declare_parameter<double>("length_scale", 1.0));
    const auto backend = declare_parameter<std::string>("backend", "cpu");
    o.backend = (backend == "cuda") ? eio::Backend::kCuda : eio::Backend::kCpu;

    tts_ = std::make_unique<eio::PiperTts>(std::move(o));

    pub_ = create_publisher<edge_inference_optimizer::msg::AudioChunk>("/tts_audio", 5);
    sub_ = create_subscription<edge_inference_optimizer::msg::LlmReply>(
      "/llm_reply", 10,
      std::bind(&TtsNode::on_reply, this, std::placeholders::_1));

    RCLCPP_INFO(get_logger(), "waiting for replies on /llm_reply");
  }

private:
  void on_reply(const edge_inference_optimizer::msg::LlmReply::ConstSharedPtr & msg)
  {
    if (msg->reply_text.empty()) { return; }

    eio::PiperTts::Audio a;
    try {
      a = tts_->synthesize(msg->reply_text);
    } catch (const std::exception & e) {
      RCLCPP_ERROR(get_logger(), "synthesis failed: %s", e.what());
      return;
    }
    if (a.samples.empty()) {
      RCLCPP_WARN(get_logger(), "no audio for '%s'", msg->reply_text.c_str());
      return;
    }

    edge_inference_optimizer::msg::AudioChunk out;
    out.header.stamp = this->now();
    out.header.frame_id = "tts";
    out.samples = a.samples;
    out.sample_rate = static_cast<uint32_t>(a.sample_rate);
    out.trace_id = msg->trace.trace_id;
    out.capture_ns = eio::steady_ns();

    // 收尾时间线。合成在关键路径上——合完之前人什么都听不到,
    // 漏掉它会让上报的端到端延迟短于人实际等待的时间。
    out.trace = msg->trace;
    const double transport = eio::transport_ms(rclcpp::Time(msg->header.stamp), out.header.stamp);
    eio::append_stage(out.trace, "dds_transport_llm", out.trace.total_ms, transport, false);
    eio::append_stage(out.trace, "tts_phonemize", out.trace.total_ms, a.phonemize_ms, true);
    eio::append_stage(out.trace, "tts_synthesize", out.trace.total_ms, a.infer_ms, true);

    pub_->publish(out);

    if (!dump_dir_.empty()) {
      const auto path = dump_dir_ + "/reply_" + std::to_string(++seq_) + ".wav";
      if (eio::PiperTts::write_wav(path, a.samples, a.sample_rate)) {
        RCLCPP_INFO(get_logger(), "wrote %s", path.c_str());
      }
    }

    RCLCPP_INFO(
      get_logger(),
      "说: \"%s\"\n  [phonemize %.1f ms | synth %.0f ms | %.2f s audio | RTF %.3f]",
      msg->reply_text.c_str(), a.phonemize_ms, a.infer_ms, a.duration_s(),
      (a.infer_ms / 1000.0) / std::max(1e-6, a.duration_s()));
  }

  std::unique_ptr<eio::PiperTts> tts_;
  rclcpp::Publisher<edge_inference_optimizer::msg::AudioChunk>::SharedPtr pub_;
  rclcpp::Subscription<edge_inference_optimizer::msg::LlmReply>::SharedPtr sub_;
  std::string dump_dir_;
  uint64_t seq_ = 0;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<TtsNode>());
  } catch (const std::exception & e) {
    RCLCPP_FATAL(rclcpp::get_logger("tts_node"), "fatal: %s", e.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
