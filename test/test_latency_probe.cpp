// LatencyStats / LatencyTrace 的基本行为。

#include <gtest/gtest.h>

#include <cmath>
#include <thread>
#include <vector>

#include "edge_inference_optimizer/latency_probe.hpp"

using eio::LatencyStats;
using eio::LatencyTrace;
using eio::ScopedStage;

TEST(LatencyStats, EmptyIsZeroNotUndefined)
{
  LatencyStats s;
  EXPECT_EQ(s.count(), 0u);
  EXPECT_DOUBLE_EQ(s.mean(), 0.0);
  EXPECT_DOUBLE_EQ(s.stddev(), 0.0);
  EXPECT_DOUBLE_EQ(s.percentile(0.95), 0.0);
}

TEST(LatencyStats, MeanAndSampleStddev)
{
  LatencyStats s;
  for (double v : {2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0}) { s.add(v); }
  EXPECT_DOUBLE_EQ(s.mean(), 5.0);
  EXPECT_NEAR(s.stddev(), std::sqrt(32.0 / 7.0), 1e-12);
}

TEST(LatencyStats, NearestRankPercentiles)
{
  LatencyStats s;
  for (int i = 1; i <= 100; ++i) { s.add(static_cast<double>(i)); }
  EXPECT_DOUBLE_EQ(s.min(), 1.0);
  EXPECT_DOUBLE_EQ(s.max(), 100.0);
  EXPECT_DOUBLE_EQ(s.percentile(0.95), 95.0);
  EXPECT_DOUBLE_EQ(s.percentile(0.99), 99.0);
  EXPECT_DOUBLE_EQ(s.percentile(0.50), 50.0);
}

TEST(LatencyStats, PercentileIsOrderIndependent)
{
  LatencyStats a, b;
  const std::vector<double> v{9, 1, 8, 2, 7, 3, 6, 4, 5, 10};
  for (double x : v) { a.add(x); }
  for (auto it = v.rbegin(); it != v.rend(); ++it) { b.add(*it); }
  EXPECT_DOUBLE_EQ(a.percentile(0.9), b.percentile(0.9));
}

TEST(LatencyTrace, TotalIsWallClockSpanNotStageSum)
{
  LatencyTrace t(1, 0);
  t.add("a", 0, 10'000'000);          // 0 -> 10ms
  t.add("b", 0, 10'000'000);          // 0 -> 10ms, concurrent
  EXPECT_DOUBLE_EQ(t.total_ms(), 10.0);
  EXPECT_DOUBLE_EQ(t.compute_ms(), 20.0);
}

TEST(LatencyTrace, TotalIncludesInterStageGaps)
{
  LatencyTrace t(1, 0);
  t.add("a", 0, 5'000'000);            // 0 -> 5ms
  t.add("b", 20'000'000, 25'000'000);  // 20 -> 25ms
  EXPECT_DOUBLE_EQ(t.total_ms(), 25.0);
  EXPECT_DOUBLE_EQ(t.compute_ms(), 10.0);
}

TEST(LatencyTrace, ComputeAndWaitAreSeparated)
{
  LatencyTrace t(7, 0);
  t.add("silence_window", 0, 800'000'000, /*is_compute=*/false);
  t.add("asr", 800'000'000, 1'100'000'000, /*is_compute=*/true);
  EXPECT_DOUBLE_EQ(t.wait_ms(), 800.0);
  EXPECT_DOUBLE_EQ(t.compute_ms(), 300.0);
  EXPECT_DOUBLE_EQ(t.total_ms(), 1100.0);
}

TEST(ScopedStage, RecordsOnScopeExit)
{
  LatencyTrace t = LatencyTrace::start(42);
  ASSERT_EQ(t.stages().size(), 0u);
  {
    ScopedStage s(t, "work");
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  ASSERT_EQ(t.stages().size(), 1u);
  EXPECT_EQ(t.stages()[0].name, "work");
  EXPECT_GE(t.stages()[0].duration_ms(), 4.0);
}

TEST(ScopedStage, CommitIsIdempotent)
{
  LatencyTrace t = LatencyTrace::start(1);
  {
    ScopedStage s(t, "once");
    const double a = s.commit();
    const double b = s.commit();
    EXPECT_DOUBLE_EQ(a, b);
  }  // destructor must not append a second copy
  EXPECT_EQ(t.stages().size(), 1u);
}

int main(int argc, char ** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
