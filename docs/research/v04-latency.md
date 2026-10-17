# v04 int8 靶机延迟构成（threads=4，Windows，本机 CPU）

口径：`$WORK\export\v04-int8-ml256`（int8-gather，单图），`Predictor(session.run)` 单次前向耗时，
batch=1，精确长度。这不是靶机实测，是训练机实测，靶机只会更慢。

**同一个 session**：本文的三组数字（端到端、state 预算扫描、batching）出自**一次进程、
一个 session**。这一点是硬要求：同一个包在不同 session 之间的端到端 p50 实测能差 2.5 倍
（见 [RUNS.md](RUNS.md) 2026-10-17 条），比模型侧任何效应都大。本次 session 的机器状态读数
（2000x2000 numpy matmul 的 p50）：**62.0ms**（跑完再量一次是 62.5ms）。
物理内存：开始时空闲 20320MB / 总 30413MB，
结束时 20669MB —— 内存不是这次波动的原因。

端到端（`bench_input.jsonl` 里的 150 条真实工单，warmup 8，
与 `python -m mai_jev.deploy` 同款协议）：
p50 **361.55ms**、p95 **462.32ms**、mean **367.43ms**，
加载后常驻 **814.8MB**、峰值 **833.1MB**、加载 3.69s。

## 一句话结论

**p50 < 50ms 这条验收线在 322M 编码器 + 4 线程下不可能达到，且不是量化方案的问题。**
模型**完全不读工单正文**（state 预算设 0）时，三原语合计已经要 243.6ms。

## 延迟由问句主导，正文几乎不参与

部署口径是「精确长度」：state 只截**不补**。真实工单正文 p50 只有几十 token，
所以预算一旦超过该工单正文的长度，几档就截到**同一条序列** —— 各档实际序列长度见
`--out-json` 的 `sweep.by_question.*.seq_len`，右几格的小差异是噪声，不是「预算越大越慢」。
每格是该档 3 条工单 × 8 轮（共 24 次）前向的 p50（ms），
档位之间轮转取样以抵消机器状态漂移。

| question | head 长度 | 0 | 32 | 64 | 96 | 满 |
---|---|---|---|---|---|---|
| route_team | 109 | **101.4** | 127.5 | 138.2 | 135.0 | 134.3 |
| urgency | 110 | **100.4** | 125.2 | 139.4 | 134.4 | 139.0 |
| needs_refund_or_escalation | 37 | **41.8** | 65.7 | 70.8 | 72.5 | 71.1 |

三原语合计：state_room=0 → **243.6ms**（完全不给正文）；满预算 → **344.4ms**（真实工单的完整长度）。

## batching：合并成一次前向更慢

同一批工单、同一个 session 里对比：三原语合成 batch=3 一次前向 p50 **375.9ms**，反而比三次
batch=1 的 **359.5ms** 更慢 —— 编码器无论如何都要逐行跑，而三个问句产生的序列确实不同。
（batch=3 右 padding 还会让短的 `noul` 行返回非有限值，这是硬约束，见
[quantization-and-deployment.md](quantization-and-deployment.md) §7。NaN / argmax 那两问由
`scripts/probe_batch_merge.py` 负责，本文只报耗时。）

## 复现

```powershell
python scripts/probe_latency_session.py `
  --model-dir <导出目录> --input <工单 jsonl> `
  --threads 4 --repeats 150 --warmup 8 `
  --out-json <session.json> --out-md docs/research/v04-latency.md
```

**跨 session 不可比**：本文的数字只在上面那个 session 的状态下成立。要横向对比两个变体，
必须在同一个 session 里交替测（见 `docs/research/RUNS.md` 2026-10-17 条）。
