// 四档对比里的 C++ 侧(tier 3/4)。参数必须和 scripts/benchmark_python.py 一致。

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "edge_inference_optimizer/latency_probe.hpp"
#include "edge_inference_optimizer/onnx_engine.hpp"

namespace
{

struct Args
{
  std::string model;
  std::string label = "model";
  std::string out_json;
  int iters = 200;
  int warmup = 20;
  int threads = 1;
  bool cuda = false;
  /// 两次推理之间的空闲。用来模拟真实节点不是打满 CPU 的情况。
  int sleep_ms = 0;
  std::vector<int64_t> shape;
};

void usage(const char * prog)
{
  std::cerr
    << "usage: " << prog << " --model PATH [options]\n"
    << "  --model PATH      .onnx file (required)\n"
    << "  --shape a,b,c,d   input shape; default = model's declared shape\n"
    << "  --iters N         timed iterations (default 200)\n"
    << "  --warmup N        untimed iterations (default 20)\n"
    << "  --threads N       intra-op AND inter-op threads (default 1)\n"
    << "  --sleep-ms N      idle gap between iterations, to emulate a duty\n"
    << "                    cycle (0 = tight loop, the usual benchmark lie)\n"
    << "  --cuda            request the CUDA execution provider\n"
    << "  --label NAME      name recorded in the JSON output\n"
    << "  --out FILE        append one JSON record to FILE\n";
}

bool parse(int argc, char ** argv, Args & a)
{
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    auto next = [&](const char * name) -> std::string {
        if (i + 1 >= argc) {
          std::cerr << "missing value for " << name << '\n';
          std::exit(2);
        }
        return argv[++i];
      };
    if (k == "--model") { a.model = next("--model"); } else if (k == "--label") {
      a.label = next("--label");
    } else if (k == "--out") { a.out_json = next("--out"); } else if (k == "--iters") {
      a.iters = std::stoi(next("--iters"));
    } else if (k == "--warmup") { a.warmup = std::stoi(next("--warmup")); } else if (
      k == "--threads")
    {
      a.threads = std::stoi(next("--threads"));
    } else if (k == "--sleep-ms") {
      a.sleep_ms = std::stoi(next("--sleep-ms"));
    } else if (k == "--cuda") { a.cuda = true; } else if (k == "--shape") {
      std::stringstream ss(next("--shape"));
      std::string tok;
      while (std::getline(ss, tok, ',')) { a.shape.push_back(std::stoll(tok)); }
    } else if (k == "-h" || k == "--help") { usage(argv[0]); std::exit(0); } else {
      std::cerr << "unknown argument: " << k << '\n';
      usage(argv[0]);
      return false;
    }
  }
  if (a.model.empty()) { usage(argv[0]); return false; }
  return true;
}

std::string json_escape(const std::string & s)
{
  std::string o;
  for (char c : s) {
    if (c == '"' || c == '\\') { o += '\\'; o += c; } else if (c == '\n') { o += "\\n"; } else {
      o += c;
    }
  }
  return o;
}

}  // namespace

int main(int argc, char ** argv)
{
  Args args;
  if (!parse(argc, argv, args)) { return 2; }

  eio::OnnxEngine::Options opts;
  opts.model_path = args.model;
  opts.backend = args.cuda ? eio::Backend::kCuda : eio::Backend::kCpu;
  opts.intra_op_threads = args.threads;
  opts.inter_op_threads = args.threads;
  opts.opt_level = GraphOptimizationLevel::ORT_ENABLE_ALL;

  std::unique_ptr<eio::OnnxEngine> engine;
  try {
    engine = std::make_unique<eio::OnnxEngine>(opts);
  } catch (const std::exception & e) {
    std::cerr << "failed to load " << args.model << ": " << e.what() << '\n';
    return 1;
  }

  // --shape 优先,否则用模型声明的形状,动态轴按 1。
  std::vector<int64_t> shape = args.shape;
  if (shape.empty()) {
    shape = engine->inputs().at(0).shape;
    for (auto & d : shape) { if (d < 0) { d = 1; } }
  }
  int64_t n_elem = 1;
  for (int64_t d : shape) { n_elem *= d; }

  // 确定性填充,和 Python 侧同一套公式,不用 RNG。
  std::vector<float> input(static_cast<size_t>(n_elem));
  for (size_t i = 0; i < input.size(); ++i) {
    input[i] = static_cast<float>(i % 251) / 251.0f;
  }

  std::cout << engine->describe() << "\n\n";
  if (args.cuda && engine->backend() != eio::Backend::kCuda) {
    std::cerr << "WARNING: CUDA was requested but the session fell back to CPU ("
              << engine->cuda_fallback_reason() << ").\n"
              << "         This record is CPU data and is labelled as such.\n\n";
  }

  for (int i = 0; i < args.warmup; ++i) { engine->run(input, shape); }

  eio::LatencyStats pure;      // time inside Session::Run only
  eio::LatencyStats wall;      // plus tensor construction and output copy
  for (int i = 0; i < args.iters; ++i) {
    if (args.sleep_ms > 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(args.sleep_ms));
    }
    const int64_t t0 = eio::steady_ns();
    engine->run(input, shape);
    wall.add(eio::ns_to_ms(eio::steady_ns() - t0));
    pure.add(engine->last_inference_ms());
  }

  const std::string backend_name = eio::to_string(engine->backend());
  std::cout << std::fixed << std::setprecision(3)
            << "label      : " << args.label << '\n'
            << "backend    : " << backend_name << '\n'
            << "threads    : " << args.threads << '\n'
            << "iters      : " << args.iters << " (warmup " << args.warmup << ")\n"
            << "run() mean : " << pure.mean() << " ms  (sd " << pure.stddev() << ")\n"
            << "run() p50  : " << pure.percentile(0.50) << " ms\n"
            << "run() p95  : " << pure.percentile(0.95) << " ms\n"
            << "run() p99  : " << pure.percentile(0.99) << " ms\n"
            << "run() min  : " << pure.min() << " ms\n"
            << "wall mean  : " << wall.mean() << " ms\n";

  if (!args.out_json.empty()) {
    std::ofstream f(args.out_json, std::ios::app);
    if (!f) {
      std::cerr << "cannot append to " << args.out_json << '\n';
      return 1;
    }
    f << std::fixed << std::setprecision(6)
      << "{\"tier\":\"" << (engine->backend() == eio::Backend::kCuda ? "ort_cpp_cuda" : "ort_cpp")
      << "\",\"label\":\"" << json_escape(args.label)
      << "\",\"model\":\"" << json_escape(args.model)
      << "\",\"backend\":\"" << backend_name
      << "\",\"threads\":" << args.threads
      << ",\"iters\":" << args.iters
      << ",\"warmup\":" << args.warmup
      << ",\"ort_version\":\"" << Ort::GetVersionString() << '"'
      << ",\"mean_ms\":" << pure.mean()
      << ",\"sd_ms\":" << pure.stddev()
      << ",\"p50_ms\":" << pure.percentile(0.50)
      << ",\"p95_ms\":" << pure.percentile(0.95)
      << ",\"p99_ms\":" << pure.percentile(0.99)
      << ",\"min_ms\":" << pure.min()
      << ",\"wall_mean_ms\":" << wall.mean()
      << "}\n";
    std::cout << "appended JSON record to " << args.out_json << '\n';
  }
  return 0;
}
