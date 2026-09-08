// LatencyTrace 和 ROS 消息的互转。

#ifndef EDGE_INFERENCE_OPTIMIZER__ROS_TRACE_HPP_
#define EDGE_INFERENCE_OPTIMIZER__ROS_TRACE_HPP_

#include <string>
#include <vector>

#include "builtin_interfaces/msg/time.hpp"
#include "edge_inference_optimizer/latency_probe.hpp"
#include "edge_inference_optimizer/msg/latency_trace.hpp"
#include "edge_inference_optimizer/msg/stage_timing.hpp"
#include "rclcpp/rclcpp.hpp"

namespace eio
{

using LatencyTraceMsg = edge_inference_optimizer::msg::LatencyTrace;
using StageTimingMsg = edge_inference_optimizer::msg::StageTiming;

/// 序列化成消息。各段偏移相对起点存储,跨进程后仍然可比。
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

/// 往已序列化的 trace 追加一段(偏移单位:毫秒)
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

/// header 戳到现在的毫秒。未来戳当 0,避免画出负传输时间。
inline double transport_ms(const rclcpp::Time & stamp, const rclcpp::Time & now)
{
  const double ms = (now - stamp).nanoseconds() / 1.0e6;
  return ms > 0.0 ? ms : 0.0;
}

}  // namespace eio

#endif  // EDGE_INFERENCE_OPTIMIZER__ROS_TRACE_HPP_
