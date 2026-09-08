# 一段单声道 PCM，float32，范围 [-1, 1]。

std_msgs/Header header
float32[]        samples
uint32           sample_rate
uint64           trace_id
int64            capture_ns

# 只有 /tts_audio 带完整 trace，给延迟报告用。
LatencyTrace     trace
