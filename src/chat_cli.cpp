// 单独测对话，不走 ROS。

#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "wulun_demo/qwen_chat.hpp"

int main(int argc, char ** argv)
{
  std::string model_dir, prompt;
  std::vector<std::string> turns;
  int threads = 4, max_new = 48;
  bool cuda = false;

  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    auto nx = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
    if (k == "--model-dir") { model_dir = nx(); } else if (k == "--say") {
      turns.push_back(nx());
    } else if (k == "--threads") { threads = std::stoi(nx()); } else if (k == "--max-new") {
      max_new = std::stoi(nx());
    } else if (k == "--cuda") { cuda = true; }
  }
  if (model_dir.empty() || turns.empty()) {
    std::cerr << "usage: chat_cli --model-dir DIR --say \"你好\" [--say \"...\"] "
              << "[--threads N] [--cuda]\n";
    return 2;
  }

  eio::QwenChat::Options o;
  o.model_path = model_dir + "/onnx/model_q4f16.onnx";
  o.vocab_path = model_dir + "/qwen_vocab.bin";
  o.merges_path = model_dir + "/qwen_merges.bin";
  o.backend = cuda ? eio::Backend::kCuda : eio::Backend::kCpu;
  o.threads = threads;
  o.max_new_tokens = max_new;

  try {
    eio::QwenChat chat(std::move(o));
    std::cout << chat.engine().describe().substr(0, 160) << "...\n\n";

    for (const auto & t : turns) {
      std::cout << "用户: " << t << '\n';
      const auto r = chat.chat(t);
      std::cout << "机器人: " << r.text << '\n'
                << std::fixed << std::setprecision(1)
                << "  [prompt " << r.prompt_tokens << " tok"
                << " | tokenize " << r.tokenize_ms << " ms"
                << " | prefill " << r.prefill_ms << " ms"
                << " | decode " << r.decode_ms << " ms / " << r.generated_tokens << " tok"
                << " = " << r.ms_per_token << " ms/tok]\n\n";
    }
  } catch (const std::exception & e) {
    std::cerr << "fatal: " << e.what() << '\n';
    return 1;
  }
  return 0;
}
