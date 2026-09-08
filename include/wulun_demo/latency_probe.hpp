// 记延迟。算的时间和等的时间分开。

#ifndef WULUN_DEMO__LATENCY_PROBE_HPP_
#define WULUN_DEMO__LATENCY_PROBE_HPP_

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

/// 各段延迟。跟着消息走。
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

  /// 从头到尾的时间，不是简单加起来。中间可能有重叠，也可能在排队。
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

/// 作用域结束时把这段时间记进去。
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

  /// 提前结束并记下耗时。
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

/// 延迟统计，带 P95/P99。
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
  /// 百分位，q 在 0 到 1。
  double percentile(double q) const;

private:
  std::vector<double> samples_;
};

}  // namespace eio

#endif  // WULUN_DEMO__LATENCY_PROBE_HPP_
