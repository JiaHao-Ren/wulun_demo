# 延迟记录，每个节点往后面加一段。

uint64                   trace_id
builtin_interfaces/Time  trace_start
StageTiming[]            stages
float64                  total_ms
