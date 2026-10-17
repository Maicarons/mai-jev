# v03 int8 靶机延迟构成（threads=4，Windows，本机 CPU）

口径：`$WORK\export\v03-int8-ml256`（int8-gather，666.0 MB），
`Predictor(session.run)` 单次前向耗时，batch=1，精确长度，40 条 test 工单取 p50。
这不是靶机实测，是训练机实测，靶机只会更慢。

**注意**：本文的 383.6ms 是「只算 session 前向、固定把 state 预算填满」的数字。
发布口径用**端到端**（含 tokenizer 与后处理）：同一变体 150 条真实工单实测
p50 421–499ms、p95 862–976ms，见 [RUNS.md](RUNS.md) 2026-10-17 条。两个数字不矛盾，
但它们不是同一件事，对外引用要用端到端那个。

**跨 session 不可比**：后来在 2026-10-17 查到，同一台机、同一个包、同一条命令，
不同 session 之间端到端 p50 能差 **2.5 倍**（同一个 v04 包实测 313ms 与 806ms 都有），
比模型侧任何效应都大。本文这张表只是**当时那个 session** 的记录，不要和别的 session
的数字混着引用；v04 包在同一 session 内量完的那一份见 [v04-latency.md](v04-latency.md)。

## 一句话结论

**p50 < 50ms 这条验收线在 322M 编码器 + 4 线程下不可能达到，且不是量化方案的问题。**
模型**完全不读工单正文**（state 预算设 0）时，三原语合计已经要 283ms。

## 延迟 ≈ 序列长度 × ~1ms/token

| question | head 长度 | state_room=0 | 32 | 64 | 96 | 146（满） |
|---|---|---|---|---|---|---|
| route_team | 109 | **104.7** | 136.9 | 137.7 | 143.0 | 139.8 |
| urgency | 110 | **132.3** | 132.9 | 144.6 | 145.1 | 143.9 |
| needs_refund_or_escalation | 37 | **46.1** | 71.6 | 82.9 | 83.6 | 99.9 |

三原语合计：state_room=0 → **283.1ms**；满预算 → **383.6ms**（与 #9 的 403.7ms 端到端口径一致）。

## 三个关键事实

1. **head 长度是主要成本，不是工单正文。** route_team / urgency 的 head 各 109 / 110 token，
   而 test 工单正文 p50 只有 47.5 token、max 82。把 state 从 146 砍到 0 只省 35ms。
2. **head 之所以长，是 `questions.py` 的 criteria 描述本身很长**（每个选项一句中文
   「一般售后咨询：保修、维修、换货、使用问题」这类）。而 head 文本必须与训练时一致，
   不能为了提速擅自截短——那会让选项语义在推理侧与训练侧不一致。
3. **加线程已吃满**：t1 835.5 → t4 378.7 → t8 332.0 → t16 368.7 ms（#9 实测）。
   fp32 不比 int8 快，说明瓶颈是 CPU 算力不是内存带宽。

## 「把三原语合并成一次前向」不成立

实测 batch=3（右 padding）：

- **非有限值 60/180** —— 复现 #8 报告 1.5 的 NaN bug；
- batch=3 一次前向 p50 **824.9ms**，反而比三次 batch=1 的 **748.0ms** 更慢。

原因是 `DecisionModel.forward`（`laya/common.py:609`）是逐行过 encoder 的，
batch 维度不会省掉编码；而三个原语各带一段**不同**的 question head，
输入序列根本不同，没有可共享的编码结果。

## 复现

```
python -c "<sweep：固定 head，改变 state 预算，量 session.run 耗时>"
python $WORK\export\_probe_merge.py $WORK\export\v03-int8-ml256 60 4
```

## 建议（需用户拍板，见票 #9）

把 Destination 的延迟口径从「单问题 p50 < 50ms」改成与实测一致的口径，例如
「一条工单跑完三原语 p50 < 500ms（4 线程，322M 量化模型）」，
并把 p50 < 50ms 作为**未来小模型档位**（见 map fog 里「第二阶段蒸馏到 ≤500MB」那条）的验收线。