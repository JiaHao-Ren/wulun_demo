// ASR 节点。订 /utterance,发 /asr_result。encoder / decoder 分开计时。

#include <memory>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"

#include "edge_inference_optimizer/latency_probe.hpp"
#include "edge_inference_optimizer/msg/asr_result.hpp"
#include "edge_inference_optimizer/msg/audio_chunk.hpp"
#include "edge_inference_optimizer/ros_trace.hpp"
#include "edge_inference_optimizer/whisper_asr.hpp"

class AsrNode : public rclcpp::Node
{
public:
  AsrNode()
  : Node("asr_node")
  {
    const auto dir = declare_parameter<std::string>("model_dir", "");
    const auto backend = declare_parameter<std::string>("backend", "cpu");
    endpoint_wait_ms_ = declare_parameter<double>("endpoint_wait_ms", 800.0);

    if (dir.empty()) {
      RCLCPP_FATAL(get_logger(), "parameter 'model_dir' is required");
      throw std::runtime_error("model_dir not set");
    }

    eio::WhisperAsr::Options o;
    o.encoder_path = dir + "/encoder_model.onnx";
    o.decoder_path = dir + "/decoder_model.onnx";
    o.decoder_with_past_path = dir + "/decoder_with_past_model.onnx";
    o.mel_filters_path = dir + "/mel_filters.bin";
    o.vocab_path = dir + "/vocab.bin";
    o.backend = (backend == "cuda") ? eio::Backend::kCuda : eio::Backend::kCpu;
    o.threads = declare_parameter<int>("threads", 1);
    o.max_tokens = declare_parameter<int>("max_tokens", 64);

    // 对话里每句只有一两秒,语言检测不稳,这里直接指定。
    const auto lang = declare_parameter<std::string>("language", "zh");
    o.language_token = (lang == "en") ? eio::WhisperAsr::kEn : eio::WhisperAsr::kZh;

    // 简体引导,避免同一段对话里简繁混用。
    if (lang == "zh" && declare_parameter<bool>("simplified_chinese", true)) {
      o.initial_prompt_tokens = eio::WhisperAsr::simplified_chinese_prompt();
    }
    RCLCPP_INFO(get_logger(), "language forced to '%s'", lang.c_str());

    asr_ = std::make_unique<eio::WhisperAsr>(std::move(o));
    RCLCPP_INFO(get_logger(), "encoder: %s", asr_->encoder().describe().c_str());

    pub_ = create_publisher<edge_inference_optimizer::msg::AsrResult>("/asr_result", 10);
    sub_ = create_subscription<edge_inference_optimizer::msg::AudioChunk>(
      "/utterance", 5,
      std::bind(&AsrNode::on_utterance, this, std::placeholders::_1));

    RCLCPP_INFO(get_logger(), "waiting for utterances on /utterance");
  }

private:
  void on_utterance(const edge_inference_optimizer::msg::AudioChunk::ConstSharedPtr & msg)
  {
    if (msg->samples.empty()) { return; }

    const rclcpp::Time now = this->now();
    const double transport = eio::transport_ms(rclcpp::Time(msg->header.stamp), now);

    eio::WhisperAsr::Result res;
    try {
      res = asr_->transcribe(msg->samples);
    } catch (const std::exception & e) {
      RCLCPP_ERROR(get_logger(), "transcription failed: %s", e.what());
      return;
    }

    const double audio_s = msg->sample_rate
      ? static_cast<double>(msg->samples.size()) / msg->sample_rate
      : 0.0;
    const double infer_ms = res.preprocess_ms + res.encoder_ms + res.decoder_ms;

    edge_inference_optimizer::msg::AsrResult out;
    out.header.stamp = now;
    out.header.frame_id = "utterance";
    out.text = res.text;
    out.audio_duration_s = static_cast<float>(audio_s);
    out.preprocess_ms = static_cast<float>(res.preprocess_ms);
    out.encoder_ms = static_cast<float>(res.encoder_ms);
    out.decoder_ms = static_cast<float>(res.decoder_ms);
    out.decoded_tokens = res.decoded_tokens;
    out.inference_ms = static_cast<float>(infer_ms);
    out.rtf = audio_s > 0.0 ? static_cast<float>(infer_ms / (audio_s * 1000.0)) : 0.0f;

    // endpoint_wait 在上游已经发生,这里按配置补进瀑布图,否则对话延迟会被低估。
    out.trace.trace_id = msg->trace_id;
    out.trace.trace_start = msg->header.stamp;
    out.trace.total_ms = 0.0;
    eio::append_stage(out.trace, "endpoint_wait", 0.0, endpoint_wait_ms_, /*is_compute=*/false);
    eio::append_stage(out.trace, "dds_transport_utterance", out.trace.total_ms,
      transport, /*is_compute=*/false);
    eio::append_stage(out.trace, "mel_preprocess", out.trace.total_ms,
      res.preprocess_ms, true);
    eio::append_stage(out.trace, "whisper_encoder", out.trace.total_ms, res.encoder_ms, true);
    eio::append_stage(out.trace, "whisper_decoder", out.trace.total_ms, res.decoder_ms, true);

    pub_->publish(out);

    RCLCPP_INFO(
      get_logger(),
      "\"%s\"\n  audio %.2f s | mel %.1f | enc %.1f | dec %.1f ms (%d tokens) | RTF %.3f",
      res.text.c_str(), audio_s, res.preprocess_ms, res.encoder_ms,
      res.decoder_ms, res.decoded_tokens, out.rtf);
  }

  std::unique_ptr<eio::WhisperAsr> asr_;
  rclcpp::Publisher<edge_inference_optimizer::msg::AsrResult>::SharedPtr pub_;
  rclcpp::Subscription<edge_inference_optimizer::msg::AudioChunk>::SharedPtr sub_;
  double endpoint_wait_ms_ = 800.0;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<AsrNode>());
  } catch (const std::exception & e) {
    RCLCPP_FATAL(rclcpp::get_logger("asr_node"), "fatal: %s", e.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
