// 表情转成舵机角度，转动速度有上限。

#ifndef WULUN_DEMO__SERVO_MAPPER_HPP_
#define WULUN_DEMO__SERVO_MAPPER_HPP_

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

  /// 登记某个表情的角度。超出范围会夹住，个数不对返回 false。
  bool set_pose(const std::string & emotion, const std::vector<double> & angles);

  bool has_pose(const std::string & emotion) const;
  const std::vector<std::string> & joint_names() const { return names_; }
  const std::vector<JointLimit> & limits() const { return limits_; }
  size_t joint_count() const { return names_.size(); }

  /// 目标角度。不认识的表情当中性。
  std::vector<double> target_for(const std::string & emotion) const;

  void set_max_speed_deg_per_s(double v) { max_speed_ = v > 0.0 ? v : 0.0; }
  double max_speed_deg_per_s() const { return max_speed_; }

  /// 朝目标转一点，速度有上限。dt 小于等于 0 就不动。
  const std::vector<double> & step(const std::string & emotion, double dt_s);

  const std::vector<double> & current() const { return current_; }

  /// 0 还没到，1 到了。
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

#endif  // WULUN_DEMO__SERVO_MAPPER_HPP_
