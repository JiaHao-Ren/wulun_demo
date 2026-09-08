// 延迟记录转成 ROS 消息。

#ifndef WULUN_DEMO__ROS_TRACE_HPP_
#define WULUN_DEMO__ROS_TRACE_HPP_

#include <string>
#include <vector>

#include "builtin_interfaces/msg/time.hpp"
#include "wulun_demo/latency_probe.hpp"
#include "wulun_demo/msg/latency_trace.hpp"
#include "wulun_demo/msg/stage_timing.hpp"
#include "rclcpp/rclcpp.hpp"

namespace eio
{

using LatencyTraceMsg = wulun_demo::msg::LatencyTrace;
using StageTimingMsg = wulun_demo::msg::StageTiming;

/// 转成 ROS 消息。
inline LatencyTraceMsg to_msg(const LatencyTrace & t, const rclcpp::Time & started)
{
  LatencyTraceMsg m;
  m.trace_id = t.id();
  m.trace_start = started;
  m.stages.reserve(t.stages().size());
  for (const auto & s : t.stages()) {
    StageTimingMsg sm;
    sm.name = s.name;
    sm.start_ms = ns_to_ms(s.start_ns - t.origin_ns());
    sm.duration_ms = s.duration_ms();
    sm.is_compute = s.is_compute;
    m.stages.push_back(sm);
  }
  m.total_ms = t.total_ms();
  return m;
}

/// 再加一段，单位毫秒。
inline void append_stage(
  LatencyTraceMsg & m, const std::string & name,
  double start_ms, double duration_ms, bool is_compute = true)
{
  StageTimingMsg sm;
  sm.name = name;
  sm.start_ms = start_ms;
  sm.duration_ms = duration_ms;
  sm.is_compute = is_compute;
  m.stages.push_back(sm);
  const double end = start_ms + duration_ms;
  if (end > m.total_ms) { m.total_ms = end; }
}

/// 从时间戳到现在过了多少毫秒。时间乱了就当 0。
inline double transport_ms(const rclcpp::Time & stamp, const rclcpp::Time & now)
{
  const double ms = (now - stamp).nanoseconds() / 1.0e6;
  return ms > 0.0 ? ms : 0.0;
}

}  // namespace eio

#endif  // WULUN_DEMO__ROS_TRACE_HPP_
