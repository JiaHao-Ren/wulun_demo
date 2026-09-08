// 表情到面部舵机角度。过渡按角速度限幅,不走指数混合。

#ifndef EDGE_INFERENCE_OPTIMIZER__SERVO_MAPPER_HPP_
#define EDGE_INFERENCE_OPTIMIZER__SERVO_MAPPER_HPP_

#include <map>
#include <string>
#include <vector>

namespace eio
{

struct JointLimit
{
  double min_deg = 0.0;
  double max_deg = 180.0;
};

class ServoMapper
{
public:
  void configure(std::vector<std::string> joint_names, std::vector<JointLimit> limits);

  /// 注册某个表情的目标姿态。超限角度在注册时就夹紧,
  /// 配置写错也无法在运行时穿到执行器。arity 不符返回 false。
  bool set_pose(const std::string & emotion, const std::vector<double> & angles);

  bool has_pose(const std::string & emotion) const;
  const std::vector<std::string> & joint_names() const { return names_; }
  const std::vector<JointLimit> & limits() const { return limits_; }
  size_t joint_count() const { return names_.size(); }

  /// 目标姿态,未知表情回落到 neutral
  std::vector<double> target_for(const std::string & emotion) const;

  void set_max_speed_deg_per_s(double v) { max_speed_ = v > 0.0 ? v : 0.0; }
  double max_speed_deg_per_s() const { return max_speed_; }

  /// 朝目标推进一步,每个关节最多移动 max_speed * dt 度。
  /// dt <= 0 时保持不动(时间戳重复不应导致跳变或倒退)。
  const std::vector<double> & step(const std::string & emotion, double dt_s);

  const std::vector<double> & current() const { return current_; }

  /// 过渡进度 [0, 1],1 表示已到位
  double blend_alpha() const { return alpha_; }

  void reset_to(const std::vector<double> & angles);

  static double clamp(double v, const JointLimit & l);

private:
  std::vector<std::string> names_;
  std::vector<JointLimit> limits_;
  std::map<std::string, std::vector<double>> poses_;
  std::vector<double> current_;
  std::vector<double> blend_from_;
  std::string active_emotion_;
  double max_speed_ = 180.0;
  double alpha_ = 1.0;
};

}  // namespace eio

#endif  // EDGE_INFERENCE_OPTIMIZER__SERVO_MAPPER_HPP_
