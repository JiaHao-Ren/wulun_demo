// 打延迟瀑布图。订两条链路的尾巴,自己不发 topic。

#include <algorithm>
#include <iomanip>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"

#include "edge_inference_optimizer/latency_probe.hpp"
#include "edge_inference_optimizer/msg/asr_result.hpp"
#include "edge_inference_optimizer/msg/audio_chunk.hpp"
#include "edge_inference_optimizer/msg/latency_trace.hpp"
#include "edge_inference_optimizer/msg/servo_command.hpp"

namespace
{

constexpr int kBarWidth = 46;

struct StageAgg
{
  eio::LatencyStats stats;
  bool is_compute = true;
  double mean_start_ms = 0.0;
  int order = 0;
};

std::string bar(double start_ms, double dur_ms, double total_ms)
{
  if (total_ms <= 0.0) { return std::string(kBarWidth, ' '); }
  const double scale = static_cast<double>(kBarWidth) / total_ms;
  int lead = static_cast<int>(start_ms * scale);
  int len = static_cast<int>(dur_ms * scale);
  lead = std::clamp(lead, 0, kBarWidth);
  // 再短也占一格,空着会看起来像没耗时。
  len = std::clamp(len, dur_ms > 0.0 ? 1 : 0, kBarWidth - lead);
  std::string s(lead, ' ');
  s += std::string(static_cast<size_t>(len), '#');
  s += std::string(static_cast<size_t>(kBarWidth - lead - len), ' ');
  return s;
}

}  // namespace

class LatencyReporterNode : public rclcpp::Node
{
public:
  LatencyReporterNode()
  : Node("latency_reporter")
  {
    const auto period = declare_parameter<double>("report_period_s", 5.0);
    min_samples_ = declare_parameter<int>("min_samples", 10);

    sub_servo_ = create_subscription<edge_inference_optimizer::msg::ServoCommand>(
      "/servo_commands", 50,
      [this](edge_inference_optimizer::msg::ServoCommand::ConstSharedPtr m) {
        if (!m->trace.stages.empty()) { absorb("vision", m->trace); }
      });

    // 订对话链路尾巴。同时订 /asr_result 会把同一轮记两次。
    sub_tts_ = create_subscription<edge_inference_optimizer::msg::AudioChunk>(
      "/tts_audio", 20,
      [this](edge_inference_optimizer::msg::AudioChunk::ConstSharedPtr m) {
        if (!m->trace.stages.empty()) { absorb("dialogue", m->trace); }
      });

    // 没有 TTS 时可以只看 ASR。
    if (declare_parameter<bool>("report_asr_only", false)) {
      sub_asr_ = create_subscription<edge_inference_optimizer::msg::AsrResult>(
        "/asr_result", 50,
        [this](edge_inference_optimizer::msg::AsrResult::ConstSharedPtr m) {
          if (!m->trace.stages.empty()) { absorb("dialogue", m->trace); }
        });
    }

    timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(period > 0.0 ? period : 5.0)),
      std::bind(&LatencyReporterNode::report, this));

    RCLCPP_INFO(get_logger(), "reporting every %.1f s (min %d samples)", period, min_samples_);
  }

private:
  void absorb(const std::string & branch, const edge_inference_optimizer::msg::LatencyTrace & t)
  {
    auto & b = branches_[branch];
    for (const auto & s : t.stages) {
      auto it = b.stages.find(s.name);
      if (it == b.stages.end()) {
        StageAgg agg;
        agg.order = static_cast<int>(b.stages.size());
        it = b.stages.emplace(s.name, std::move(agg)).first;
      }
      it->second.stats.add(s.duration_ms);
      it->second.is_compute = s.is_compute;
      // 起点用滑动平均,柱子位置按均值画。
      const double n = static_cast<double>(it->second.stats.count());
      it->second.mean_start_ms += (s.start_ms - it->second.mean_start_ms) / n;
    }
    b.total.add(t.total_ms);
  }

  void report()
  {
    for (const auto & [name, b] : branches_) {
      if (static_cast<int>(b.total.count()) < min_samples_) { continue; }
      print_branch(name, b);
    }
  }

  struct Branch
  {
    std::map<std::string, StageAgg> stages;
    eio::LatencyStats total;
  };

  void print_branch(const std::string & name, const Branch & b)
  {
    std::vector<std::pair<std::string, const StageAgg *>> ordered;
    ordered.reserve(b.stages.size());
    for (const auto & [k, v] : b.stages) { ordered.emplace_back(k, &v); }
    std::sort(
      ordered.begin(), ordered.end(),
      [](const auto & a, const auto & c) { return a.second->mean_start_ms < c.second->mean_start_ms; });

    const double total = b.total.mean();
    double compute = 0.0, wait = 0.0;
    for (const auto & [k, v] : ordered) {
      (void)k;
      (v->is_compute ? compute : wait) += v->stats.mean();
    }

    std::ostringstream os;
    os << '\n'
       << "+----------------------------------------------------------------------------------+\n"
       << "| END-TO-END LATENCY WATERFALL  --  " << std::left << std::setw(47) << (name + " branch")
       << "|\n"
       << "| samples: " << std::left << std::setw(72) << b.total.count() << "|\n"
       << "+----------------------------------------------------------------------------------+\n";

    os << std::fixed << std::setprecision(2);
    for (const auto & [k, v] : ordered) {
      os << ' ' << std::left << std::setw(22) << k.substr(0, 22)
         << '|' << bar(v->mean_start_ms, v->stats.mean(), total) << '|'
         << std::right << std::setw(9) << std::setprecision(3) << v->stats.mean() << " ms"
         << std::setw(9) << v->stats.percentile(0.95) << " p95  " << std::setprecision(2)
         << (v->is_compute ? "compute" : "WAIT") << '\n';
    }

    os << " " << std::string(22, '-') << '+' << std::string(kBarWidth, '-') << "+\n"
       << ' ' << std::left << std::setw(22) << "TOTAL (end-to-end)"
       << ' ' << std::string(kBarWidth, ' ') << ' '
       << std::right << std::setw(8) << total << " ms"
       << std::setw(8) << b.total.percentile(0.95) << " p95\n";

    const double denom = (compute + wait) > 0.0 ? (compute + wait) : 1.0;
    os << '\n'
       << " compute : " << std::setw(8) << compute << " ms  ("
       << std::setprecision(1) << (100.0 * compute / denom) << "%)"
       << "   <- shrinks with a faster model or accelerator\n"
       << std::setprecision(2)
       << " waiting : " << std::setw(8) << wait << " ms  ("
       << std::setprecision(1) << (100.0 * wait / denom) << "%)"
       << "   <- does NOT; needs a protocol or threshold change\n";

    RCLCPP_INFO(get_logger(), "%s", os.str().c_str());
  }

  std::map<std::string, Branch> branches_;
  rclcpp::Subscription<edge_inference_optimizer::msg::ServoCommand>::SharedPtr sub_servo_;
  rclcpp::Subscription<edge_inference_optimizer::msg::AsrResult>::SharedPtr sub_asr_;
  rclcpp::Subscription<edge_inference_optimizer::msg::AudioChunk>::SharedPtr sub_tts_;
  rclcpp::TimerBase::SharedPtr timer_;
  int min_samples_ = 10;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LatencyReporterNode>());
  rclcpp::shutdown();
  return 0;
}
