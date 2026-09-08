// 判断人说完没。说完了把整段发到 /utterance。

#include <algorithm>
#include <deque>
#include <memory>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"

#include "wulun_demo/latency_probe.hpp"
#include "wulun_demo/msg/audio_chunk.hpp"
#include "wulun_demo/msg/vad_result.hpp"
#include "wulun_demo/onnx_engine.hpp"
#include "wulun_demo/ros_trace.hpp"

class VadNode : public rclcpp::Node
{
public:
  VadNode()
  : Node("vad_node")
  {
    const auto model = declare_parameter<std::string>("model_path", "");
    const auto backend = declare_parameter<std::string>("backend", "cpu");
    threshold_ = declare_parameter<double>("threshold", 0.5);
    min_silence_ms_ = declare_parameter<double>("min_silence_ms", 800.0);
    min_speech_ms_ = declare_parameter<double>("min_speech_ms", 250.0);
    // 前面多带一点声音，不然词头会被切掉。
    pre_roll_ms_ = declare_parameter<double>("pre_roll_ms", 300.0);
    min_consec_speech_ = declare_parameter<int>("min_consec_speech_frames", 3);

    if (model.empty()) {
      RCLCPP_FATAL(get_logger(), "parameter 'model_path' is required");
      throw std::runtime_error("model_path not set");
    }

    eio::OnnxEngine::Options opts;
    opts.model_path = model;
    opts.backend = (backend == "cuda") ? eio::Backend::kCuda : eio::Backend::kCpu;
    opts.intra_op_threads = declare_parameter<int>("threads", 1);
    opts.inter_op_threads = opts.intra_op_threads;
    engine_ = std::make_unique<eio::OnnxEngine>(opts);
    RCLCPP_INFO(get_logger(), "%s", engine_->describe().c_str());

    reset_state();

    pub_vad_ = create_publisher<wulun_demo::msg::VadResult>("/vad_result", 50);
    pub_utt_ = create_publisher<wulun_demo::msg::AudioChunk>("/utterance", 5);
    sub_ = create_subscription<wulun_demo::msg::AudioChunk>(
      "/audio_raw", 50,
      std::bind(&VadNode::on_audio, this, std::placeholders::_1));

    RCLCPP_INFO(
      get_logger(), "threshold %.2f, endpoint after %.0f ms of silence",
      threshold_, min_silence_ms_);
  }

private:
  void reset_state()
  {
    // 状态要跨帧留着，每帧清掉会不准，而且还不报错。
    state_.assign(2 * 1 * 128, 0.0f);
  }

  void on_audio(const wulun_demo::msg::AudioChunk::ConstSharedPtr & msg)
  {
    if (msg->samples.empty() || msg->sample_rate == 0) { return; }

    const double frame_ms =
      1000.0 * static_cast<double>(msg->samples.size()) / static_cast<double>(msg->sample_rate);

    std::vector<float> chunk = msg->samples;
    std::vector<int64_t> sr_val{static_cast<int64_t>(msg->sample_rate)};

    const int64_t t0 = eio::steady_ns();
    float prob = 0.0f;
    try {
      std::vector<Ort::Value> inputs;
      inputs.push_back(
        engine_->make_tensor(
          chunk.data(), chunk.size(),
          {1, static_cast<int64_t>(chunk.size())}));
      inputs.push_back(engine_->make_tensor(state_.data(), state_.size(), {2, 1, 128}));
      // sr 在模型里是 0 维标量,形状向量必须为空;
      // 传 {1} 在部分 ORT 版本能过、部分会拒绝。
      inputs.push_back(engine_->make_tensor(sr_val.data(), 1, {}));

      auto outs = engine_->run_named(
        {"input", "state", "sr"}, inputs, {"output", "stateN"});

      prob = outs[0].GetTensorData<float>()[0];
      const float * new_state = outs[1].GetTensorData<float>();
      std::copy(new_state, new_state + state_.size(), state_.begin());
    } catch (const std::exception & e) {
      RCLCPP_ERROR(get_logger(), "VAD inference failed: %s", e.what());
      return;
    }
    const double infer_ms = eio::ns_to_ms(eio::steady_ns() - t0);

    const bool is_speech = prob >= threshold_;
    advance(is_speech, frame_ms, infer_ms, msg);

    wulun_demo::msg::VadResult out;
    out.header.stamp = this->now();
    out.header.frame_id = msg->header.frame_id;
    out.is_speech = is_speech;
    out.speech_prob = prob;
    out.utterance_end = fired_this_frame_;
    out.inference_ms = static_cast<float>(infer_ms);
    out.trailing_silence_ms = static_cast<float>(silence_ms_);
    out.trace_id = msg->trace_id;
    pub_vad_->publish(out);
  }

  void advance(
    bool is_speech, double frame_ms, double infer_ms,
    const wulun_demo::msg::AudioChunk::ConstSharedPtr & msg)
  {
    fired_this_frame_ = false;
    sample_rate_ = msg->sample_rate;

    // 静音里 Silero 会冒孤立假阳性;任意正帧都清零的话端点永远不触发。
    if (is_speech) { ++consec_speech_; } else { consec_speech_ = 0; }
    const bool speech_confirmed =
      in_speech_ ? (consec_speech_ >= min_consec_speech_) : is_speech;

    if (speech_confirmed) {
      if (!in_speech_) {
        in_speech_ = true;
        speech_ms_ = 0.0;
        vad_compute_ms_ = 0.0;
        utterance_start_ns_ = eio::steady_ns();
        // 用回溯缓冲给这段语音开头,保住词头
        buffer_.assign(pre_roll_.begin(), pre_roll_.end());
      }
      speech_ms_ += frame_ms;
      silence_ms_ = 0.0;
    } else if (in_speech_) {
      silence_ms_ += frame_ms;
    }

    if (in_speech_) {
      buffer_.insert(buffer_.end(), msg->samples.begin(), msg->samples.end());
      vad_compute_ms_ += infer_ms;
      pre_roll_.clear();
    } else {
      // 保留最近 pre_roll_ms_ 长度的非语音音频备用
      pre_roll_.insert(pre_roll_.end(), msg->samples.begin(), msg->samples.end());
      const size_t cap = static_cast<size_t>(
        pre_roll_ms_ * 0.001 * static_cast<double>(msg->sample_rate));
      if (pre_roll_.size() > cap) {
        pre_roll_.erase(
          pre_roll_.begin(),
          pre_roll_.begin() + static_cast<std::ptrdiff_t>(pre_roll_.size() - cap));
      }
    }

    if (in_speech_ && silence_ms_ >= min_silence_ms_) {
      // 过滤瞬时噪声:40 ms 的「语音」多半是关门声,
      // 送去 ASR 只会白跑一次完整推理然后得到空结果。
      if (speech_ms_ >= min_speech_ms_) {
        emit_utterance();
      } else {
        RCLCPP_DEBUG(
          get_logger(), "discarded %.0f ms blip (< %.0f ms)", speech_ms_, min_speech_ms_);
      }
      in_speech_ = false;
      fired_this_frame_ = true;
      silence_ms_ = 0.0;
      speech_ms_ = 0.0;
      buffer_.clear();
      pre_roll_.clear();
      consec_speech_ = 0;
      reset_state();
    }
  }

  void emit_utterance()
  {
    wulun_demo::msg::AudioChunk utt;
    utt.header.stamp = this->now();
    utt.header.frame_id = "utterance";
    utt.samples = buffer_;
    utt.sample_rate = sample_rate_;
    utt.trace_id = ++utterance_id_;
    utt.capture_ns = utterance_start_ns_;
    pub_utt_->publish(utt);

    const double dur_s =
      sample_rate_ ? static_cast<double>(buffer_.size()) / sample_rate_ : 0.0;
    RCLCPP_INFO(
      get_logger(),
      "utterance #%lu: %.2f s audio | VAD compute %.1f ms | endpoint wait %.0f ms (WAIT)",
      static_cast<unsigned long>(utterance_id_), dur_s, vad_compute_ms_, min_silence_ms_);
  }

  std::unique_ptr<eio::OnnxEngine> engine_;
  rclcpp::Publisher<wulun_demo::msg::VadResult>::SharedPtr pub_vad_;
  rclcpp::Publisher<wulun_demo::msg::AudioChunk>::SharedPtr pub_utt_;
  rclcpp::Subscription<wulun_demo::msg::AudioChunk>::SharedPtr sub_;

  std::vector<float> state_;
  std::vector<float> buffer_;
  std::vector<float> pre_roll_;
  double threshold_ = 0.5;
  double min_silence_ms_ = 800.0, min_speech_ms_ = 250.0, pre_roll_ms_ = 300.0;
  double silence_ms_ = 0.0, speech_ms_ = 0.0, vad_compute_ms_ = 0.0;
  int64_t utterance_start_ns_ = 0;
  uint32_t sample_rate_ = 16000;
  uint64_t utterance_id_ = 0;
  int consec_speech_ = 0;
  int min_consec_speech_ = 3;   // ~96 ms at 512-sample frames / 16 kHz
  bool in_speech_ = false, fired_this_frame_ = false;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<VadNode>());
  } catch (const std::exception & e) {
    RCLCPP_FATAL(rclcpp::get_logger("vad_node"), "fatal: %s", e.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
