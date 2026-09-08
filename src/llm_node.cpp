// 生成回复。订识别结果，发 /llm_reply。

#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"

#include "wulun_demo/latency_probe.hpp"
#include "wulun_demo/msg/asr_result.hpp"
#include "wulun_demo/msg/llm_reply.hpp"
#include "wulun_demo/qwen_chat.hpp"
#include "wulun_demo/ros_trace.hpp"

class LlmNode : public rclcpp::Node
{
public:
  LlmNode()
  : Node("llm_node")
  {
    const auto dir = declare_parameter<std::string>("model_dir", "");
    const auto backend = declare_parameter<std::string>("backend", "cpu");
    const auto onnx_file = declare_parameter<std::string>("onnx_file", "onnx/model_q4f16.onnx");

    if (dir.empty()) {
      RCLCPP_FATAL(get_logger(), "parameter 'model_dir' is required");
      throw std::runtime_error("model_dir not set");
    }

    eio::QwenChat::Options o;
    o.model_path = dir + "/" + onnx_file;
    o.vocab_path = dir + "/qwen_vocab.bin";
    o.merges_path = dir + "/qwen_merges.bin";
    o.backend = (backend == "cuda") ? eio::Backend::kCuda : eio::Backend::kCpu;
    o.threads = declare_parameter<int>("threads", 8);
    o.max_new_tokens = declare_parameter<int>("max_new_tokens", 48);
    o.history_turns = declare_parameter<int>("history_turns", 3);
    o.system_prompt = declare_parameter<std::string>(
      "system_prompt",
      "你是一个机器人助手。请用简短、自然的中文口语回答,一到两句话即可。");
    o.greedy = declare_parameter<bool>("greedy", true);

    chat_ = std::make_unique<eio::QwenChat>(std::move(o));
    RCLCPP_INFO(get_logger(), "%s", chat_->engine().describe().substr(0, 150).c_str());

    pub_ = create_publisher<wulun_demo::msg::LlmReply>("/llm_reply", 10);
    sub_ = create_subscription<wulun_demo::msg::AsrResult>(
      "/asr_result", 10,
      std::bind(&LlmNode::on_transcript, this, std::placeholders::_1));

    RCLCPP_INFO(get_logger(), "waiting for transcripts on /asr_result");
  }

private:
  void on_transcript(const wulun_demo::msg::AsrResult::ConstSharedPtr & msg)
  {
    const std::string text = trim(msg->text);
    if (text.empty()) { return; }

    // 这些是静音/噪声标记，别拿去生成回复。
    for (const char * junk : {"[MUSIC PLAYING]", "(upbeat music)", "[Music]", "(laughs)"}) {
      if (text.find(junk) != std::string::npos) {
        RCLCPP_WARN(get_logger(), "ignoring ASR artefact: '%s'", text.c_str());
        return;
      }
    }

    const rclcpp::Time now = this->now();
    const double transport = eio::transport_ms(rclcpp::Time(msg->header.stamp), now);

    eio::QwenChat::Reply r;
    try {
      r = chat_->chat(text);
    } catch (const std::exception & e) {
      RCLCPP_ERROR(get_logger(), "generation failed: %s", e.what());
      return;
    }

    wulun_demo::msg::LlmReply out;
    out.header.stamp = this->now();
    out.header.frame_id = "dialogue";
    out.user_text = text;
    out.reply_text = r.text;
    out.prompt_tokens = r.prompt_tokens;
    out.generated_tokens = r.generated_tokens;
    out.tokenize_ms = static_cast<float>(r.tokenize_ms);
    out.prefill_ms = static_cast<float>(r.prefill_ms);
    out.decode_ms = static_cast<float>(r.decode_ms);
    out.ms_per_token = static_cast<float>(r.ms_per_token);

    // 接上游 trace 继续记录,让报告端看到从麦克风采集到回复就绪的完整时间线
    out.trace = msg->trace;
    eio::append_stage(out.trace, "dds_transport_asr", out.trace.total_ms, transport, false);
    eio::append_stage(out.trace, "llm_prefill", out.trace.total_ms, r.prefill_ms, true);
    eio::append_stage(out.trace, "llm_decode", out.trace.total_ms, r.decode_ms, true);

    pub_->publish(out);

    RCLCPP_INFO(
      get_logger(), "用户: %s\n  机器人: %s\n  [prompt %d tok | prefill %.0f ms | "
      "decode %.0f ms / %d tok = %.1f ms/tok]",
      text.c_str(), r.text.c_str(), r.prompt_tokens, r.prefill_ms,
      r.decode_ms, r.generated_tokens, r.ms_per_token);
  }

  static std::string trim(const std::string & s)
  {
    const auto b = s.find_first_not_of(" \t\n\r");
    if (b == std::string::npos) { return ""; }
    const auto e = s.find_last_not_of(" \t\n\r");
    return s.substr(b, e - b + 1);
  }

  std::unique_ptr<eio::QwenChat> chat_;
  rclcpp::Publisher<wulun_demo::msg::LlmReply>::SharedPtr pub_;
  rclcpp::Subscription<wulun_demo::msg::AsrResult>::SharedPtr sub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<LlmNode>());
  } catch (const std::exception & e) {
    RCLCPP_FATAL(rclcpp::get_logger("llm_node"), "fatal: %s", e.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
