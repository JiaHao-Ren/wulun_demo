#include "edge_inference_optimizer/latency_probe.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace eio
{

double LatencyStats::mean() const
{
  if (samples_.empty()) { return 0.0; }
  return std::accumulate(samples_.begin(), samples_.end(), 0.0) /
         static_cast<double>(samples_.size());
}

double LatencyStats::stddev() const
{
  if (samples_.size() < 2) { return 0.0; }
  const double m = mean();
  double acc = 0.0;
  for (double v : samples_) { acc += (v - m) * (v - m); }
  // 样本标准差(N-1):这些是从过程中抽样,不是全体
  return std::sqrt(acc / static_cast<double>(samples_.size() - 1));
}

double LatencyStats::min() const
{
  if (samples_.empty()) { return 0.0; }
  return *std::min_element(samples_.begin(), samples_.end());
}

double LatencyStats::max() const
{
  if (samples_.empty()) { return 0.0; }
  return *std::max_element(samples_.begin(), samples_.end());
}

double LatencyStats::percentile(double q) const
{
  if (samples_.empty()) { return 0.0; }
  if (q <= 0.0) { return min(); }
  if (q >= 1.0) { return max(); }

  std::vector<double> sorted = samples_;
  std::sort(sorted.begin(), sorted.end());

  const auto n = static_cast<double>(sorted.size());
  auto rank = static_cast<size_t>(std::ceil(q * n));
  if (rank < 1) { rank = 1; }
  if (rank > sorted.size()) { rank = sorted.size(); }
  return sorted[rank - 1];
}

}  // namespace eio
