#include "wulun_demo/servo_mapper.hpp"

#include <algorithm>
#include <cmath>

namespace eio
{

double ServoMapper::clamp(double v, const JointLimit & l)
{
  if (v < l.min_deg) { return l.min_deg; }
  if (v > l.max_deg) { return l.max_deg; }
  return v;
}

void ServoMapper::configure(std::vector<std::string> joint_names, std::vector<JointLimit> lim)
{
  names_ = std::move(joint_names);
  limits_ = std::move(lim);
  // 限位表比关节表短时补默认值,避免后面越界访问
  limits_.resize(names_.size(), JointLimit{});
  poses_.clear();
  current_.assign(names_.size(), 0.0);
  blend_from_ = current_;
  active_emotion_.clear();
  alpha_ = 1.0;
}

bool ServoMapper::set_pose(const std::string & emotion, const std::vector<double> & angles)
{
  if (angles.size() != names_.size()) { return false; }
  std::vector<double> clamped(angles.size());
  for (size_t i = 0; i < angles.size(); ++i) {
    clamped[i] = clamp(angles[i], limits_[i]);
  }
  poses_[emotion] = std::move(clamped);
  return true;
}

bool ServoMapper::has_pose(const std::string & emotion) const
{
  return poses_.find(emotion) != poses_.end();
}

std::vector<double> ServoMapper::target_for(const std::string & emotion) const
{
  auto it = poses_.find(emotion);
  if (it != poses_.end()) { return it->second; }
  // 未知表情回落到 neutral,再不行保持当前姿态。
  // 不能回落到全零:对面部机构而言那是把每个舵机拉到行程一端。
  auto neutral = poses_.find("neutral");
  if (neutral != poses_.end()) { return neutral->second; }
  return current_;
}

void ServoMapper::reset_to(const std::vector<double> & angles)
{
  current_.assign(names_.size(), 0.0);
  for (size_t i = 0; i < names_.size() && i < angles.size(); ++i) {
    current_[i] = clamp(angles[i], limits_[i]);
  }
  blend_from_ = current_;
  alpha_ = 1.0;
}

const std::vector<double> & ServoMapper::step(const std::string & emotion, double dt_s)
{
  if (names_.empty()) { return current_; }

  if (emotion != active_emotion_) {
    active_emotion_ = emotion;
    blend_from_ = current_;
    alpha_ = 0.0;
  }

  if (dt_s <= 0.0) { return current_; }

  const std::vector<double> target = target_for(emotion);
  const double max_delta = max_speed_ * dt_s;

  double total_span = 0.0;
  double remaining = 0.0;

  for (size_t i = 0; i < current_.size() && i < target.size(); ++i) {
    const double diff = target[i] - current_[i];
    const double stepped = std::abs(diff) <= max_delta
      ? target[i]
      : current_[i] + std::copysign(max_delta, diff);
    current_[i] = clamp(stepped, limits_[i]);

    total_span = std::max(total_span, std::abs(target[i] - blend_from_[i]));
    remaining = std::max(remaining, std::abs(target[i] - current_[i]));
  }

  alpha_ = (total_span <= 0.0) ? 1.0 : std::clamp(1.0 - remaining / total_span, 0.0, 1.0);
  return current_;
}

}  // namespace eio
