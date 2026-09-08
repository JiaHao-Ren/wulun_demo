# 一帧音频的语音活动。utterance_end 表示判定人说完了。

std_msgs/Header header
bool             is_speech
float32          speech_prob
bool             utterance_end
float32          inference_ms
float32          trailing_silence_ms
uint64           trace_id
