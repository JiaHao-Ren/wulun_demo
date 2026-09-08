# 一段音频。

std_msgs/Header header
float32[]        samples
uint32           sample_rate
uint64           trace_id
int64            capture_ns

# 合成语音这边带上延迟记录。
LatencyTrace     trace
