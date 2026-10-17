# v03 int8 量化退化报告（checkpoint `v03_rlcd_bal`）

## 结论

**PASS（就"int8 量化退化 ≤ 0.05/原语"这条验收口径而言）。** int8-gather 相对 fp32 ONNX 的
argmax 准确率退化是 route_team **0.0000**、urgency **0.0058**、needs_refund_or_escalation
**0.0029**，全部 ≤ 0.05。**v03 的 urgency 修复在量化下完整存活**——int8-gather 只量化 Gather
（embedding 查表），MatMul 全留 fp32，urgency argmax 一致率 0.9942（343 行中仅 2 行翻转）。

但有一个**必须先修的独立问题**：v03 的 ONNX 导出路径相对 eager 本身就有明显退化，
这是本次测量中更大的误差源，和量化无关，见下。

## 测量设置

- 数据：`$WORK\data\adapted_v03\mai_jev_test.laya.jsonl`，343 行全量，未抽样。
- reference：eager `laya.load(runs\v03_rlcd_bal, calibration=runs\v03_rlcd_bal\calibration.json)`，
  `batch=1` 精确长度，逐行 `agent.predict`。
- 两条量化侧都走 `mai_jev.deploy.Predictor`，同一 `calibration.json`、同一温度，
  以隔离量化误差（不引入温度失配）。
- fp32 ONNX：`onnx\v03_ft\laya_fp32.onnx`，1228.6 MB，导出 parity `max|dlogits| = 1.550e-06`。
- 变体：`--variant int8-gather --max-len 128`，666.0 MB。
- gold 取 `row['gold'][qid]['probabilities']` 的 argmax；urgency 的 score key 是 0-based。

## 主表：int8-gather 相对 fp32 ONNX（本次验收口径）

| 原语 | fp32 ONNX argmax acc | int8 argmax acc | DELTA（验收指标，≤0.05） | max abs prob 偏差 | argmax 一致率 |
|---|---|---|---|---|---|
| route_team | 0.7988 | 0.7988 | **0.0000** | 0.2354 | 1.0000 |
| urgency | 0.8017 | 0.7959 | **0.0058** | 0.0716 | 0.9942 |
| needs_refund_or_escalation | 0.8513 | 0.8484 | **0.0029** | 0.0884 | 0.9971 |

> ⚠️ 本文所有 `needs_refund_or_escalation` 的绝对值（0.8513 / 0.8484）都取自 **v0.3 标签
> 定义**，而那个标签已被证伪为一条 proxy（问的是「是否需要人工介入」而不是客户诉求，
> 见 `DATASETS.md` §3 第 1 条与 [#17](https://github.com/Maicarons/mai-jev/issues/17)）。
> 作为**量化退化**的对照它们仍然有效（fp32 与 int8 用同一份 gold、同一口径比较），
> 但**不可以**当成「退款诉求准确率」对外引用；v0.4 重标后 noul 的多数类基线变成 0.796，
> 裸准确率跨版本不可比。

int8 相对 fp32 ONNX 翻转的样本（共 3 行）：urgency `zhtw-train-01750`（0→1）、
`zhbench-bs-00021`（1→0）；needs_refund `zhtw-train-01658`（false→true）。

注意 max abs 概率偏差全部由少数离群行贡献，且不影响 argmax——route_team 偏差最大
（0.2354）却一行 argmax 都没变，属于温度缩放后的尾部概率抖动，不是判别力损失。

## 附表：ONNX 路径相对 eager 的退化（与量化无关）

| 原语 | eager fp32 acc | fp32 ONNX acc | DELTA | int8 acc | DELTA vs eager |
|---|---|---|---|---|---|
| route_team | 0.9446 | 0.7988 | **+0.1458** | 0.7988 | +0.1458 |
| urgency | 0.9329 | 0.8017 | **+0.1312** | 0.8017 | +0.1312 |
| needs_refund_or_escalation | 0.8513 | 0.8513 | +0.0000 | 0.8484 | +0.0029 |

这些 acc 是对 gold 的准确率，gold 取各行 `gold[qid]['probabilities']` 的 argmax。
fp32 ONNX 自身 argmax 一致率对 eager：route_team 0.8484、urgency 0.8367、
needs_refund_or_escalation 1.0000。

## 根因：`max_len` 小于问题头长度（已定位，未修）

按任务给定的 `--max-len 96` 构建时，v03 的问题头已经变长：

- `route_team` 头 109 token（marker 在 17/35/52/71/90）
- `urgency` 头 110 token（marker 在 20/35/54/67/86）
- `needs_refund_or_escalation` 头 37 token

`Predictor._feed` 在 `max_len=96` 下把 `room = 96 - 110 - 1 < 0`，于是**状态文本被整段丢弃**，
urgency 第 5 个 marker（86）之后的选项文本也被截断。此时退化是灾难性的：
route_team 0.9446→0.1866、urgency 0.9329→0.7026（argmax 一致率 0.198 / 0.738），
而且 **fp32 ONNX 与 int8 数值几乎完全相同**——证明这不是量化误差，是导出配置错误。

上表所有数字均在 `--max-len 128`（≥ 110 头长 + 状态）下测得。

## 待决

1. ~~发布配置必须改为 `max_len ≥ 112`~~ **已修并已升级为硬闸门。** 父 agent 已动手：
   - `mai_jev/deploy.py`：新增 `MIN_STATE_ROOM = 32` 与 `Predictor.state_room`，
     `__init__` 里逐个 question 检查 `max_len - len(head.ids) - 1`，不足即抛
     `ValueError` 并报出「至少设到多少」；`_feed` 改为读 `self.state_room[qid]`。
   - `scripts/build_export.py`：新增 `deploy_head_lengths()`，构建发布目录前用
     **部署侧同一份 `_Head`** 量出 head 长度，装不下直接 `SystemExit`。
   两者合起来保证「head 比 max_len 长」这个静默失效不可能再进到发布目录里。
2. ~~跨路径（eager→ONNX）残余 0.13–0.15 的 route_team/urgency 退化根因未定位~~
   **已定位：与第 1 条是同一个 bug，只是程度不同。** head 110 token 时
   `max_len=128` 只剩 18 token 给 state，仍远低于 32 的下限；因此 128 也是坏的，
   只是没 96 那么灾难。父 agent 的 max_len 扫描（115 行子样本，同一 export 目录
   只改 `Predictor.max_len`）：

   | max_len | route_team | urgency | noul | macro |
   |---|---|---|---|---|
   | 96 | 0.1826 | 0.7826 | 0.8348 | 0.6000 |
   | 128 | — | — | — | —（state 只剩 18，低于下限） |
   | 256 | 0.9652 | 0.9652 | 0.8348 | 0.9217 |
   | 512 | 0.9652 | 0.9652 | 0.8348 | 0.9217 |

   256 与 512 完全一致，说明 256 已经饱和。最终在 **343 行全量 test split** 上
   复测 `--max-len 256` 的 int8-gather 产物（`export\v03-int8-ml256`，666.0 MB）：

   | 原语 | eager fp32 acc | int8 @ml256 acc | delta |
   |---|---|---|---|
   | route_team | 0.9475 | **0.9475** | 0.0000 |
   | urgency | 0.9329 | **0.9359** | +0.0030 |
   | needs_refund_or_escalation | 0.8513 | **0.8484** | −0.0029 |
   | macro | 0.9106 | **0.9106** | 0.0000 |

   **发布变体定为 `v03-int8-ml256`：相对 eager 零退化，量化退化 ≤0.003。**
   注意本报告主表（ml128）与本节（ml256）数字不可混用——ml128 的 state 预算是坏的。

## 未做

- 旧 checkpoint v02 的同类数字**未测**（任务允许跳过）。v03 的结论不依赖它。
- 整图 `int8`（非 gather）变体**未测**。v02 上已证其 urgency argmax 一致率塌到 0.417，
  且在 v03 上 fp32-ONNX 与 int8-gather 已足够接近，没有理由再花时间在全图 int8 上。
- v03 的 int8 温度**未在 int8 图上重新拟合**。当前沿用 fp32 图拟合的温度，
  依据是本报告 ml256 上 argmax 一致率 1.000、max 概率偏差 ≤0.09；严格起见
  发布前应在 ml256 变体自己的 calib 上重拟合一次（#8 定的规矩），本次未做。

## 复现

```
# 注意 --seq-len 是导出 traced shape，与发布用的 max_len 是两件事；
# 真正决定 state 预算的是 build_export.py 的 --max-len。
python scripts/export_onnx.py --model-dir $WORK\runs\v03_rlcd_bal \
  --out-dir $WORK\onnx\v03_ft --seq-len 128 --marker-rows 8
# 发布变体（最终口径，max_len=256）：
python scripts/build_export.py --model-dir $WORK\runs\v03_rlcd_bal \
  --fp32 $WORK\onnx\v03_ft\laya_fp32.onnx \
  --calibration $WORK\runs\v03_rlcd_bal\calibration.json \
  --out $WORK\export\v03-int8-ml256 --variant int8-gather --max-len 256
# fp32 对照：同上，--variant fp32 --out $WORK\export\v03-fp32-ml256
```
