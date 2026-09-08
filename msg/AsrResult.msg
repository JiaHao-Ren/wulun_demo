# 一句语音的识别结果。

std_msgs/Header header
string           text
float32          audio_duration_s
float32          preprocess_ms
float32          encoder_ms
float32          decoder_ms
int32            decoded_tokens
float32          inference_ms
float32          rtf
LatencyTrace     trace
