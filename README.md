# 对话延时测试并加速

给无论科技做的 demo。展会上客户与机器人对话时会卡几秒或者响应不灵敏

这个项目用 ROS 2 + C++ 把语音和表情推理跑在本机，并把每一个部分的延时拆开

开发平台：ROS 2 Humble，ONNX Runtime 1.28，RTX 3060。

演示视频：[docs/ceshi_demo.mp4](docs/ceshi_demo.mp4)

## 功能

两条链路:

- 声音：麦克风或 wav → VAD → Whisper 转写 → Qwen 回复 → piper 合成语音
- 画面：摄像头或视频 → 人脸 → 表情分类 → 五路面部舵机角度（眉、嘴、眼皮）


## 运行命令

```bash
conda activate ros2demo
cd <workspace>
source install/setup.bash

# 回放测试视频
ros2 launch wulun_demo full_demo.launch.py \
  models_dir:=/path/to/models \
  source:=file wav_path:=audio.wav \
  camera:=file video_path:=video.mp4 \
  backend:=cuda play_audio:=false loop:=false

# 使用麦克风和摄像头实时对话
ros2 launch wulun_demo full_demo.launch.py \
  models_dir:=/path/to/models \
  source:=mic camera:=device backend:=cuda
```

编译和测试：

```bash
colcon build --packages-select wulun_demo --cmake-args -DCMAKE_BUILD_TYPE=Release
colcon test --packages-select wulun_demo
```

## 测试数据

同一段中文录音、视频，优化前后：

| | 优化前 | 优化后 |
|---|---:|---:|
| 人说完到机器人开口 | 3119 ms | 1128 ms |
| Whisper 解码 | 2462 ms | 677 ms |
| 等待占比 | 48.9% | 65.8% |

对话链路每一段（优化后，单位 ms）：

| 阶段 | 耗时 | 类型 |
|---|---:|---|
| 等你说完（静音 500 ms） | 500.0 | 等待 |
| 节点传输 | 0.4 | 等待 |
| 音频预处理 | 25.0 | 计算 |
| Whisper encoder | 56.9 | 计算 |
| Whisper decoder | 43.1 | 计算 |
| ASR 之后排队 | 125.4 | 等待 |
| Qwen prefill | 63.0 | 计算 |
| Qwen decode | 81.8 | 计算 |
| LLM 之后排队 | 116.1 | 等待 |
| TTS 音素 | 0.2 | 计算 |
| TTS 合成 | 115.7 | 计算 |
| **合计** | **1128** | |

1128 ms 里，模型计算大约 386 ms，剩下 742 ms 是在等：等你说完、以及节点之间排队。所以再换一块更快的显卡，最多只能把那 386 ms 再压下去。等的那部分不会跟着变快。

Whisper 解码从 2462 ms 降到 677 ms，是因为加了 KV cache，输出和原来一样。

表情链路（和对话同时跑）：

| 阶段 | 耗时 |
|---|---:|
| 传图 | 16.0 ms |
| 人脸检测 | 48.1 ms |
| 预处理 | 2.0 ms |
| 表情模型 | 1.7 ms |
| 传到舵机映射 | 51.9 ms |
| 舵机映射 | 0.001 ms |
| 合计 | 120 ms |

人脸检测比表情模型慢很多。把表情模型再优化，对话也不会更快。

测试 36 / 36。中文能听懂并回复。因为没有脸部模型或者相关的硬件，表情部分只有单纯的输出，并没有办法去做验证。
