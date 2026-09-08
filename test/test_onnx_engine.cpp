// ONNX 测试。没有模型就跳过，不当失败。

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "wulun_demo/onnx_engine.hpp"

namespace
{

std::string model_dir()
{
  const char * env = std::getenv("EIO_MODEL_DIR");
  return env ? std::string(env) : std::string("/home/nb/edge_ws/models");
}

std::string emotion_model() { return model_dir() + "/emotion-ferplus-8.onnx"; }

bool exists(const std::string & p)
{
  std::ifstream f(p, std::ios::binary);
  return f.good();
}

eio::OnnxEngine::Options cpu_opts(const std::string & path)
{
  eio::OnnxEngine::Options o;
  o.model_path = path;
  o.backend = eio::Backend::kCpu;
  o.intra_op_threads = 1;
  o.inter_op_threads = 1;
  return o;
}

}  // namespace

TEST(TensorSpec, ShapeStringMarksDynamicAxes)
{
  eio::TensorSpec s;
  s.name = "x";
  s.shape = {-1, 3, 64, 64};
  EXPECT_EQ(s.shape_string(), "[?,3,64,64]");
  EXPECT_EQ(s.element_count(), 3 * 64 * 64);
  EXPECT_EQ(s.element_count(8), 8 * 3 * 64 * 64);
}

TEST(Backend, NameStringsMatchOrtProviderNames)
{
  EXPECT_EQ(eio::to_string(eio::Backend::kCpu), "CPUExecutionProvider");
  EXPECT_EQ(eio::to_string(eio::Backend::kCuda), "CUDAExecutionProvider");
}

TEST(OnnxEngine, LoadsModelAndReportsDeclaredShapes)
{
  if (!exists(emotion_model())) { GTEST_SKIP() << "model not present: " << emotion_model(); }

  eio::OnnxEngine e(cpu_opts(emotion_model()));

  ASSERT_EQ(e.inputs().size(), 1u);
  ASSERT_EQ(e.outputs().size(), 1u);
  EXPECT_EQ(e.inputs()[0].name, "Input3");
  EXPECT_EQ(e.inputs()[0].shape, (std::vector<int64_t>{1, 1, 64, 64}));
  EXPECT_EQ(e.outputs()[0].name, "Plus692_Output_0");
  EXPECT_EQ(e.outputs()[0].shape, (std::vector<int64_t>{1, 8}));
}

TEST(OnnxEngine, InferenceProducesExpectedOutputRank)
{
  if (!exists(emotion_model())) { GTEST_SKIP() << "model not present"; }

  eio::OnnxEngine e(cpu_opts(emotion_model()));
  std::vector<float> in(1 * 1 * 64 * 64, 0.5f);
  auto out = e.run(in, {1, 1, 64, 64});

  EXPECT_EQ(out.size(), 8u);
  for (float v : out) { EXPECT_TRUE(std::isfinite(v)); }
}

TEST(OnnxEngine, RepeatedInferenceIsDeterministic)
{
  if (!exists(emotion_model())) { GTEST_SKIP() << "model not present"; }

  eio::OnnxEngine e(cpu_opts(emotion_model()));
  std::vector<float> in(4096);
  for (size_t i = 0; i < in.size(); ++i) {
    in[i] = static_cast<float>(i % 255) / 255.0f;
  }

  auto a = e.run(in, {1, 1, 64, 64});
  auto b = e.run(in, {1, 1, 64, 64});
  auto c = e.run(in, {1, 1, 64, 64});

  ASSERT_EQ(a.size(), b.size());
  for (size_t i = 0; i < a.size(); ++i) {
    EXPECT_FLOAT_EQ(a[i], b[i]) << "index " << i;
    EXPECT_FLOAT_EQ(a[i], c[i]) << "index " << i;
  }
}

TEST(OnnxEngine, TimerMeasuresOnlyTheRunCall)
{
  if (!exists(emotion_model())) { GTEST_SKIP() << "model not present"; }

  eio::OnnxEngine e(cpu_opts(emotion_model()));
  std::vector<float> in(4096, 0.1f);

  e.run(in, {1, 1, 64, 64});           // warm-up: first call pays lazy init
  const double warm_before = e.last_inference_ms();
  e.run(in, {1, 1, 64, 64});
  const double warm_after = e.last_inference_ms();

  EXPECT_GT(warm_before, 0.0);
  EXPECT_GT(warm_after, 0.0);
  EXPECT_LT(warm_after, 5000.0);
}

TEST(OnnxEngine, RejectsMismatchedInputSize)
{
  if (!exists(emotion_model())) { GTEST_SKIP() << "model not present"; }

  eio::OnnxEngine e(cpu_opts(emotion_model()));
  std::vector<float> too_small(100, 0.0f);
  EXPECT_THROW(e.run(too_small, {1, 1, 64, 64}), std::runtime_error);
}

TEST(OnnxEngine, ReportsActualBackendNotRequestedOne)
{
  if (!exists(emotion_model())) { GTEST_SKIP() << "model not present"; }

  auto o = cpu_opts(emotion_model());
  eio::OnnxEngine e(o);
  EXPECT_EQ(e.backend(), eio::Backend::kCpu);
  EXPECT_TRUE(e.cuda_fallback_reason().empty());
  EXPECT_NE(e.describe().find("actual=CPUExecutionProvider"), std::string::npos);
}

int main(int argc, char ** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
