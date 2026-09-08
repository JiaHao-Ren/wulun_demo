# 跟着数据走的延迟记录，每个节点往 stages 里追加一段。

uint64                   trace_id
builtin_interfaces/Time  trace_start
StageTiming[]            stages
float64                  total_ms
