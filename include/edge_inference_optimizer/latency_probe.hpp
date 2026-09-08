// 各节点共用的延迟计时。计算和等待分开记。

#ifndef EDGE_INFERENCE_OPTIMIZER__LATENCY_PROBE_HPP_
#define EDGE_INFERENCE_OPTIMIZER__LATENCY_PROBE_HPP_

#include <chrono>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace eio
{

inline int64_t steady_ns()
{
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
    std::chrono::steady_clock::now().time_since_epoch()).count();
}

inline double ns_to_ms(int64_t ns) { return static_cast<double>(ns) / 1.0e6; }

struct Stage
{
  std::string name;
  int64_t start_ns = 0;
  int64_t end_ns = 0;
  bool is_compute = true;

  double duration_ms() const { return ns_to_ms(end_ns - start_ns); }
};

/// 一次数据流经流水线时累积的各阶段耗时。
/// trace 随消息在节点间传递,终端节点即可还原完整时间线,无需跨节点对时。
class LatencyTrace
{
public:
  LatencyTrace() = default;

  explicit LatencyTrace(uint64_t id, int64_t origin_ns)
  : id_(id), origin_ns_(origin_ns) {}

  static LatencyTrace start(uint64_t id) { return LatencyTrace(id, steady_ns()); }

  uint64_t id() const { return id_; }
  int64_t origin_ns() const { return origin_ns_; }
  const std::vector<Stage> & stages() const { return stages_; }

  void add(std::string name, int64_t start_ns, int64_t end_ns, bool is_compute = true)
  {
    stages_.push_back(Stage{std::move(name), start_ns, end_ns, is_compute});
  }

  void add_stage(const Stage & s) { stages_.push_back(s); }

  /// 从起点到最后一段结束的墙钟跨度。
  /// 不是各段之和:段之间可能重叠(并行)或有空隙(排队),求和会把排队延迟藏掉。
  double total_ms() const
  {
    int64_t last = origin_ns_;
    for (const auto & s : stages_) {
      if (s.end_ns > last) { last = s.end_ns; }
    }
    return ns_to_ms(last - origin_ns_);
  }

  double compute_ms() const
  {
    double sum = 0.0;
    for (const auto & s : stages_) {
      if (s.is_compute) { sum += s.duration_ms(); }
    }
    return sum;
  }

  double wait_ms() const
  {
    double sum = 0.0;
    for (const auto & s : stages_) {
      if (!s.is_compute) { sum += s.duration_ms(); }
    }
    return sum;
  }

private:
  uint64_t id_ = 0;
  int64_t origin_ns_ = 0;
  std::vector<Stage> stages_;
};

/// RAII 计时器,析构时把该段写入 trace。
class ScopedStage
{
public:
  ScopedStage(LatencyTrace & trace, std::string name, bool is_compute = true)
  : trace_(trace), name_(std::move(name)), is_compute_(is_compute), start_ns_(steady_ns()) {}

  ~ScopedStage()
  {
    if (!committed_) { commit(); }
  }

  ScopedStage(const ScopedStage &) = delete;
  ScopedStage & operator=(const ScopedStage &) = delete;

  /// 提前结束该段并返回耗时,重复调用无副作用。
  double commit()
  {
    if (committed_) { return last_ms_; }
    const int64_t end = steady_ns();
    last_ms_ = ns_to_ms(end - start_ns_);
    trace_.add(name_, start_ns_, end, is_compute_);
    committed_ = true;
    return last_ms_;
  }

  double elapsed_ms() const { return ns_to_ms(steady_ns() - start_ns_); }

private:
  LatencyTrace & trace_;
  std::string name_;
  bool is_compute_;
  int64_t start_ns_;
  bool committed_ = false;
  double last_ms_ = 0.0;
};

/// benchmark 用的延迟样本统计。除均值外还给 P95/P99——
/// 均值好看但每秒卡一次的流水线,人感觉仍然是坏的。
class LatencyStats
{
public:
  void add(double ms) { samples_.push_back(ms); }
  size_t count() const { return samples_.size(); }
  const std::vector<double> & samples() const { return samples_; }
  void clear() { samples_.clear(); }

  double mean() const;
  double stddev() const;
  double min() const;
  double max() const;
  /// q 取 [0, 1],最近秩法
  double percentile(double q) const;

private:
  std::vector<double> samples_;
};

}  // namespace eio

#endif  // EDGE_INFERENCE_OPTIMIZER__LATENCY_PROBE_HPP_
