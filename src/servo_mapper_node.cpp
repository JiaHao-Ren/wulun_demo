// 表情转舵机角度，按固定频率发。

#include <memory>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"

#include "wulun_demo/msg/emotion_result.hpp"
#include "wulun_demo/msg/servo_command.hpp"
#include "wulun_demo/ros_trace.hpp"
#include "wulun_demo/servo_mapper.hpp"

class ServoMapperNode : public rclcpp::Node
{
public:
  ServoMapperNode()
  : Node("servo_mapper")
  {
    auto joints = declare_parameter<std::vector<std::string>>(
      "joint_names",
      std::vector<std::string>{"left_brow", "right_brow", "left_mouth", "right_mouth", "eyelid"});
    auto lo = declare_parameter<std::vector<double>>(
      "limits_min", std::vector<double>{0, 0, 40, 40, 0});
    auto hi = declare_parameter<std::vector<double>>(
      "limits_max", std::vector<double>{90, 90, 140, 140, 100});
    const auto rate = declare_parameter<double>("rate_hz", 50.0);
    const auto speed = declare_parameter<double>("max_speed_deg_per_s", 120.0);

    std::vector<eio::JointLimit> limits(joints.size());
    for (size_t i = 0; i < joints.size(); ++i) {
      limits[i].min_deg = i < lo.size() ? lo[i] : 0.0;
      limits[i].max_deg = i < hi.size() ? hi[i] : 180.0;
    }
    mapper_.configure(joints, limits);
    mapper_.set_max_speed_deg_per_s(speed);

    // 姿态从 pose.<表情> 参数读,整张表放在 config/servo_mapping.yaml,改了不用重编译
    for (const auto & label : kLabels) {
      const auto key = "pose." + label;
      const auto angles = declare_parameter<std::vector<double>>(key, std::vector<double>{});
      if (angles.empty()) { continue; }
      if (!mapper_.set_pose(label, angles)) {
        RCLCPP_ERROR(
          get_logger(), "pose '%s' has %zu angles, expected %zu -- ignored",
          label.c_str(), angles.size(), mapper_.joint_count());
      }
    }
    for (const auto & label : kLabels) {
      if (!mapper_.has_pose(label)) {
        RCLCPP_WARN(get_logger(), "no pose for '%s'; it will fall back to neutral", label.c_str());
      }
    }
    if (mapper_.has_pose("neutral")) { mapper_.reset_to(mapper_.target_for("neutral")); }

    pub_ = create_publisher<wulun_demo::msg::ServoCommand>("/servo_commands", 10);
    sub_ = create_subscription<wulun_demo::msg::EmotionResult>(
      "/emotion_result", 10,
      [this](wulun_demo::msg::EmotionResult::ConstSharedPtr m) {
        // 传输延迟在消息到达时采,不要放到 50 Hz 循环里。
        latest_transport_ms_ = eio::transport_ms(rclcpp::Time(m->header.stamp), this->now());
        latest_ = m;
      });

    const auto period = std::chrono::duration<double>(rate > 0.0 ? 1.0 / rate : 0.02);
    last_tick_ = this->now();
    timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(period),
      std::bind(&ServoMapperNode::tick, this));

    RCLCPP_INFO(
      get_logger(), "%zu joints, %.0f Hz, max %.0f deg/s",
      mapper_.joint_count(), rate, speed);
  }

private:
  void tick()
  {
    const rclcpp::Time now = this->now();
    const double dt = (now - last_tick_).seconds();
    last_tick_ = now;

    auto snapshot = latest_;
    const std::string emotion = snapshot ? snapshot->emotion : std::string("neutral");

    const int64_t t0 = eio::steady_ns();
    const auto & angles = mapper_.step(emotion, dt);
    const double map_ms = eio::ns_to_ms(eio::steady_ns() - t0);

    wulun_demo::msg::ServoCommand cmd;
    cmd.header.stamp = now;
    cmd.header.frame_id = "face";
    cmd.joint_names = mapper_.joint_names();
    cmd.angles_deg.assign(angles.begin(), angles.end());
    cmd.source_emotion = emotion;
    cmd.blend_alpha = static_cast<float>(mapper_.blend_alpha());

    // 同一条检测 trace 只转发一次,50 Hz 循环里重复发会把均值拉高。
    if (snapshot && snapshot->trace.trace_id != last_traced_id_) {
      last_traced_id_ = snapshot->trace.trace_id;
      cmd.trace = snapshot->trace;
      eio::append_stage(cmd.trace, "dds_transport_emotion", cmd.trace.total_ms,
        latest_transport_ms_, /*is_compute=*/false);
      eio::append_stage(cmd.trace, "servo_map", cmd.trace.total_ms, map_ms, true);
    }

    pub_->publish(cmd);

    if ((++ticks_ % 100) == 1) {
      RCLCPP_INFO(
        get_logger(), "%-10s alpha=%.2f | %s",
        emotion.c_str(), mapper_.blend_alpha(), format(angles).c_str());
    }
  }

  static std::string format(const std::vector<double> & a)
  {
    std::ostringstream os;
    os << std::fixed << std::setprecision(1) << '[';
    for (size_t i = 0; i < a.size(); ++i) { if (i) { os << ", "; } os << a[i]; }
    os << ']';
    return os.str();
  }

  inline static const std::vector<std::string> kLabels = {
    "neutral", "happiness", "surprise", "sadness",
    "anger", "disgust", "fear", "contempt"};

  eio::ServoMapper mapper_;
  rclcpp::Publisher<wulun_demo::msg::ServoCommand>::SharedPtr pub_;
  rclcpp::Subscription<wulun_demo::msg::EmotionResult>::SharedPtr sub_;
  rclcpp::TimerBase::SharedPtr timer_;
  wulun_demo::msg::EmotionResult::ConstSharedPtr latest_;
  double latest_transport_ms_ = 0.0;
  uint64_t last_traced_id_ = 0;
  rclcpp::Time last_tick_;
  uint64_t ticks_ = 0;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ServoMapperNode>());
  rclcpp::shutdown();
  return 0;
}
