// 舵机角度、限位、速度。

#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <vector>

#include "wulun_demo/servo_mapper.hpp"

using eio::JointLimit;
using eio::ServoMapper;

namespace
{

const std::vector<std::string> kJoints = {
  "left_brow", "right_brow", "left_mouth", "right_mouth", "eyelid"};

const std::vector<JointLimit> kLimits = {
  {0, 90}, {0, 90}, {40, 140}, {40, 140}, {0, 100}};

const std::vector<std::string> kFerPlusLabels = {
  "neutral", "happiness", "surprise", "sadness",
  "anger", "disgust", "fear", "contempt"};

ServoMapper make_mapper()
{
  ServoMapper m;
  m.configure(kJoints, kLimits);
  m.set_pose("neutral",   {45, 45, 90, 90, 50});
  m.set_pose("happiness", {30, 30, 120, 120, 80});
  m.set_pose("surprise",  {20, 20, 110, 110, 95});
  m.set_pose("sadness",   {60, 60, 70, 70, 40});
  m.set_pose("anger",     {70, 70, 60, 60, 90});
  m.set_pose("disgust",   {65, 65, 65, 65, 60});
  m.set_pose("fear",      {25, 25, 100, 100, 95});
  m.set_pose("contempt",  {50, 40, 100, 75, 55});
  m.reset_to({45, 45, 90, 90, 50});
  m.set_max_speed_deg_per_s(180.0);
  return m;
}

}  // namespace

TEST(ServoMapping, EveryModelLabelHasAPose)
{
  auto m = make_mapper();
  for (const auto & label : kFerPlusLabels) {
    EXPECT_TRUE(m.has_pose(label)) << "no servo pose configured for '" << label << "'";
  }
}

TEST(ServoMapping, ConfiguredPosesAreInsideLimits)
{
  auto m = make_mapper();
  for (const auto & label : kFerPlusLabels) {
    const auto pose = m.target_for(label);
    ASSERT_EQ(pose.size(), kJoints.size()) << label;
    for (size_t i = 0; i < pose.size(); ++i) {
      EXPECT_GE(pose[i], kLimits[i].min_deg) << label << '/' << kJoints[i];
      EXPECT_LE(pose[i], kLimits[i].max_deg) << label << '/' << kJoints[i];
    }
  }
}

TEST(ServoMapping, OutOfRangeConfigIsClampedNotAccepted)
{
  auto m = make_mapper();
  ASSERT_TRUE(m.set_pose("bogus", {-40, 999, 0, 500, -1}));
  const auto p = m.target_for("bogus");
  for (size_t i = 0; i < p.size(); ++i) {
    EXPECT_GE(p[i], kLimits[i].min_deg);
    EXPECT_LE(p[i], kLimits[i].max_deg);
  }
}

TEST(ServoMapping, WrongArityPoseIsRejected)
{
  auto m = make_mapper();
  EXPECT_FALSE(m.set_pose("too_short", {1, 2, 3}));
  EXPECT_FALSE(m.has_pose("too_short"));
}

TEST(ServoMapping, TransitionRespectsMaxSpeed)
{
  auto m = make_mapper();
  m.set_max_speed_deg_per_s(90.0);
  const double dt = 0.02;                 // 50 Hz
  const double max_per_step = 90.0 * dt;  // 1.8 deg

  auto prev = m.current();
  for (int i = 0; i < 200; ++i) {
    const auto & cur = m.step("anger", dt);
    for (size_t j = 0; j < cur.size(); ++j) {
      EXPECT_LE(std::abs(cur[j] - prev[j]), max_per_step + 1e-9)
        << "joint " << kJoints[j] << " jumped at iteration " << i;
    }
    prev = cur;
  }
}

TEST(ServoMapping, TransitionConvergesExactlyAndStops)
{
  auto m = make_mapper();
  m.set_max_speed_deg_per_s(180.0);
  for (int i = 0; i < 500; ++i) { m.step("happiness", 0.02); }

  const auto target = m.target_for("happiness");
  const auto cur = m.current();
  for (size_t j = 0; j < cur.size(); ++j) {
    EXPECT_DOUBLE_EQ(cur[j], target[j]) << kJoints[j];
  }
  EXPECT_DOUBLE_EQ(m.blend_alpha(), 1.0);
}

TEST(ServoMapping, NeverLeavesLimitsDuringTransition)
{
  auto m = make_mapper();
  m.set_max_speed_deg_per_s(720.0);   // deliberately faster than sane
  for (const auto & label : kFerPlusLabels) {
    for (int i = 0; i < 40; ++i) {
      const auto & cur = m.step(label, 0.05);
      for (size_t j = 0; j < cur.size(); ++j) {
        EXPECT_GE(cur[j], kLimits[j].min_deg) << label << '/' << kJoints[j];
        EXPECT_LE(cur[j], kLimits[j].max_deg) << label << '/' << kJoints[j];
      }
    }
  }
}

TEST(ServoMapping, ZeroOrNegativeDtHoldsPosition)
{
  auto m = make_mapper();
  const auto before = m.current();
  m.step("anger", 0.0);
  EXPECT_EQ(m.current(), before);
  m.step("anger", -0.5);
  EXPECT_EQ(m.current(), before);
}

TEST(ServoMapping, UnknownEmotionFallsBackToNeutralNotZero)
{
  auto m = make_mapper();
  const auto t = m.target_for("no_such_emotion");
  EXPECT_EQ(t, m.target_for("neutral"));
  for (size_t i = 0; i < t.size(); ++i) {
    EXPECT_GE(t[i], kLimits[i].min_deg);
  }
}

TEST(ServoMapping, AlphaProgressesMonotonically)
{
  auto m = make_mapper();
  m.set_max_speed_deg_per_s(60.0);
  double last = -1.0;
  for (int i = 0; i < 100; ++i) {
    m.step("sadness", 0.02);
    const double a = m.blend_alpha();
    EXPECT_GE(a, last - 1e-12) << "alpha went backwards at " << i;
    EXPECT_GE(a, 0.0);
    EXPECT_LE(a, 1.0);
    last = a;
  }
}

TEST(ServoMapping, RetargetMidTransitionDoesNotJump)
{
  auto m = make_mapper();
  m.set_max_speed_deg_per_s(90.0);
  for (int i = 0; i < 10; ++i) { m.step("anger", 0.02); }
  const auto mid = m.current();

  const auto & after = m.step("happiness", 0.02);
  for (size_t j = 0; j < after.size(); ++j) {
    EXPECT_LE(std::abs(after[j] - mid[j]), 90.0 * 0.02 + 1e-9) << kJoints[j];
  }
}

int main(int argc, char ** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
