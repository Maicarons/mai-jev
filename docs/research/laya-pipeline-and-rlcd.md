# 研究：Laya 微调管线与 RLCD 实现（mai-jev #3）

> 结论先行：**`laya-train` 就是我们要的训练入口，且它随 pip 包一起装在我们的 venv 里**
> （`$WORK\venv\Lib\site-packages\laya\train.py`）。我们不需要自己写训练循环，
> 需要写的是**数据适配层**（把我们的三原语 jsonl 变成 laya 的 `{state, questions, expected}` 行）
> 和**评测层**（准确率 + ECE + soft accuracy + 延迟）。下面每条都带 `file:line` 出处，
> 仓库 clone 在 `$WORK\src\laya\`（0.4.3，与 site-packages 逐字节等价，仅换行符不同）。

## TL;DR

1. 训练入口：`laya-train`（= `laya train`，`laya/cli.py:394-396`），CLI 定义在 `laya/train_cli.py:18`。
2. 数据格式：JSONL，每行 `{state, questions, expected}`（硬标签）或 `{state, questions, gold}`（教师概率），见 `docs/finetune.md:50-70`。**一个 case 可以同时问三个问题**，这正是我们要的。
3. 目标函数默认 `--loss rlcd`：策略梯度项（GRPO 组基线）+ 全权重 soft cross-entropy，`laya/train.py:499-518`。
4. 奖励是严格适当评分规则：log score + 0.75×spherical，score 问题再减 RPS，`laya/common.py:719-745`。
5. 探索噪声按 epoch 从 0.4 退火到 0.1，`sigma_at` 在 `laya/train.py:521`。
6. **温度标定是 finetune 的一部分**：训练前切出 calib 切片（≤400 条或 10%），训完在切片上 LBFGS 拟合每类型一个温度，写进 `rl_agent_config.json`（`laya/train.py:989` finetune 主体）。
7. 标定必须用**没训过的数据**。`docs/finetune.md:280-284`：在已训练项上拟合"测量的是拟合而不是标定"，优化器没东西可软化，会返回退化尺度。
8. 优化器 AdamW 双学习率（encoder 2.5e-5 / head 1e-4）+ cosine，`laya/train.py:671-681`。**无 LoRA**，只有 `--freeze-encoder`。
9. 序列预算默认取 checkpoint 自己的配置（multilingual 是 `max_len`/`head_max_len` 各自的值），notebook 用 1024/256（`docs/finetune.md:169,265`）。
10. 参考成绩（typed-decisions，单 T4）：base 0.362 acc / 0.174 ECE → `rlcd` 0.7715 / 0.150（`docs/finetune.md:183-190`）。**注意 ECE 0.150 > 我们 0.10 的验收线**，所以 #6 必须自己再拟合并报数。
11. **选项数 >20 会明显掉准确率**（laya issue #394），我们 5 类 choice 完全安全。
12. ONNX 导出脚本 `scripts/export_onnx.py` **不在 pip 包里**（`pyproject.toml:37` packages 不含 scripts），要从 GitHub 树取。INT8 是 weight-only `quantize_dynamic`，`per_channel` 默认关（#790 会塌）。
13. Windows 无额外训练坑；`laya-train` 支持 CUDA/MPS/CPU（`docs/finetune.md:23`）。唯一 Windows 坑是 stdout 编码，`laya/cli.py:398-402` 要求 `sys.stdout.reconfigure(encoding="utf-8")`。
14. 本机 8 GB 显存够跑默认配置：文档里 16 GB 卡的配方是 fp16 autocast + gradient checkpointing，默认 `amp` 在 CUDA 上自动开、`gradient_checkpointing` 跟随 `amp`（`laya/train.py:86`）。
15. 每次 epoch 后重写 `checkpoint_latest/`，崩溃最多损失一个 epoch（`docs/finetune.md:345`）。

## 1. 训练入口与完整 CLI

`laya-train`（或 `laya train`）——`laya/train_cli.py:18 build_parser()`：

```
laya-train --data DATA.jsonl --out ./my-checkpoint
           [--eval/--eval-data FILE]            # 前后对比评测集
           [--base multilingual]                # 目录 / 内置名 english|multilingual|typed-decisions / Hub id
           [--text-column text] [--label-column label]   # 仅 CSV 用
           [--question-id label] [--instructions STR]    # 仅 CSV 用
           [--epochs 4] [--micro-batch 8] [--grad-accum 8]
           [--encoder-lr 2.5e-5] [--head-lr 1e-4]
           [--loss rlcd|soft-ce] [--label-smoothing 0.0]
           [--shuffle-options] [--freeze-encoder]
           [--target-error 0.10] [--min-abstain-n 10]
           [--device auto] [--seed 0] [--max-len N] [--head-max-len N]
           [--dry-run]
```

- `--out` 在非 `--dry-run` 时必填（`laya/train_cli.py:175-176`：`parser.error("--out is required unless --dry-run is set")`）。
- `--dry-run` 只建 items、不加载权重不训练，打印行数 / items 数 / 跳过原因 / 优化器更新预算（`docs/finetune.md:73-88`）——**这是我们放全量前的第一道闸**。
- 更新预算公式 `updates = epochs * ceil(ceil(train_items / micro_batch) / grad_accum)`（`laya/train.py:607 optimizer_updates`）。
- 没有 torch 时打印 `laya-train: error: PyTorch is required for training. Please install torch.` 并返回 1（`laya/train_cli.py` main）。

## 2. 数据格式（关键：我们的 jsonl 要长什么样）

`docs/finetune.md:50-70` 定义的 JSONL 行，**一个 case 可带多个不同类型的 question**：

```json
{"state": "客户：上周买的耳机一边没声音，申请退款，急！",
 "questions": {"route_team": {"type": "choice", "instructions": "...",
                              "criteria": {"售后": "...", "退款处理": "...", "技术支持": "...",
                                           "账务": "...", "投诉升级": "..."}},
               "urgency": {"type": "score", "instructions": "...",
                           "criteria": ["1 级描述", "2 级描述", "3 级描述", "4 级描述", "5 级描述"]},
               "needs_refund_or_escalation": {"type": "noul", "instructions": "..."}},
 "expected": {"route_team": "退款处理", "urgency": 3, "needs_refund_or_escalation": true}}
```

- `expected` 是硬标签：choice 给 label 字符串，noul 给 bool（也收 `true`/`false` 字符串），score 给**整数 level 或 level 字符串**（`laya/train.py:196 target_from_expected`）。小数 level（如 1.5）会把权重分给相邻两级。
- `gold` 是教师概率：choice 按 label 键、noul 按 `"false"/"true"`、score 按 **level index 字符串**（`laya/train.py:167 target_from_gold`）。零和 → 均匀分布；负数/非有限 → `ValueError("probabilities must be finite and non-negative, got %r")`。
- `--label-smoothing 0.1` 把 10% 权重匀给其他选项（`laya/train.py:206`）。
- state 只编码一次、同行所有 question 共享（`laya/train.py:27-29`）；`serialize_state`（`laya/common.py:94`）对 str 直通、dict/list 走 `json.dumps(..., ensure_ascii=False)`，并把 `[MASK]` 替换成空格（`laya/train.py:287-289`）——**工单正文里不能出现 `[MASK]`**。
- 头部格式 `[CLS] <type> instructions [SEP] [MASK] opt0 [MASK] opt1 ... [SEP]`，每个选项 token 化时 `max_length=48`；`markers` 记录各 `[MASK]` 位置（`laya/common.py:225 build_head`）。
- 跳过原因有名字可查：`empty_text` / `empty_label` / `invalid_question` / `invalid_target` / `options_collapsed` / `options_beyond_max_len`（`docs/finetune.md:82-87`）。
- CSV 路径（`laya/train.py:365 rows_from_csv`）只造**一个** choice question，选项 = label 列去重值按首现排序，写 `questions.json` 到 checkpoint 旁。我们不用 CSV，用 JSONL。

## 3. RLCD 目标函数与奖励（逐行）

`rlcd_loss(logits, target, mask, qtype, sigma, samples=4, w_sph=0.75, w_rps=1.0)` —— `laya/train.py:499-518`：

```python
k = mask.sum(-1, keepdim=True).float(); fmask = mask.float()
eps = torch.randn((samples,) + logits.shape, device=logits.device) * sigma * fmask
eps = (eps - eps.sum(-1, keepdim=True) / k) * fmask     # 投影成该行选项上的零均值噪声
z = logits.detach().unsqueeze(0) + eps
q = torch.softmax(z.masked_fill(~mask, _MASKED), -1)
with torch.no_grad():
    r = proper_reward(q, target.unsqueeze(0), qtype, fmask, w_sph=w_sph, w_rps=w_rps)
    adv = r - r.mean(0, keepdim=True)                   # GRPO 式组基线
    adv = adv / (adv.std() + 1e-6)
logp = -(((z - logits.unsqueeze(0)) ** 2) * fmask).sum(-1) / (2 * sigma ** 2)
return -(adv * logp).mean() + soft_ce_loss(logits, target, mask)
```

`proper_reward(q, target, qtype, mask, w_sph=0.5, w_rps=1.0, log_floor=-9.21)` —— `laya/common.py:719-745`：

```python
q = q * mask
logq = torch.log(q.clamp_min(1e-12)).clamp_min(log_floor)
log_score = (target * logq).sum(-1)
sph = (target * q).sum(-1) / q.norm(dim=-1).clamp_min(1e-9)
r = log_score + w_sph * sph
is_score = (qtype == QTYPES["score"]).float()
if is_score.any():
    k = mask.sum(-1).clamp(min=2).float()
    cdf_q = torch.cumsum(q, -1); cdf_t = torch.cumsum(target, -1)
    rps = (((cdf_q - cdf_t) ** 2) * mask).sum(-1) / (k - 1)
    r = r - w_rps * rps * is_score
```

- log score 与 spherical 是**最大化**，RPS 是 CDF 平方误差被**减去**。
- `soft_ce_loss`（`laya/train.py:493-496`）= `-(target * log_softmax(logits) * mask).sum(-1).mean()`，`_MASKED = -1e4`（`laya/train.py:74`）。
- 训练默认 `w_sph=0.75`（`TrainConfig.w_sph`，`laya/train.py:103`），与函数签名默认 0.5 不同，以 TrainConfig 为准。
- 噪声退火：`sigma_at(epoch, epochs, start, end) = start + (end-start)*(epoch/max(1,epochs-1))`，默认 0.4→0.1（`laya/train.py:521,101-102`）。
- 为什么不用交叉熵：`laya/train.py:70-72` 的记录是"#963 里训练 CE 卡在 ln 4 是坍缩信号；上报的 training loss 不是 CE，默认 rlcd 目标就不是交叉熵"。（仓库里没有更深的"为什么 RLCD 优于 CE"的理论论证；`train.py:7-10` 只说 `soft-ce` 可用且 #741 测得 typed-decisions 上该项无增益。）

## 4. 温度标定与持久化

`laya/calibrate.py`（612 行），常量：`MIN_BUCKET_N=2000`、`MIN_TYPE_N=10`、`ECE_HOLDOUT_FRAC=0.2`、`CALIBRATION_VERSION=2`、`MIN_ABSTAIN_BUCKET_N=100`、`MIN_BINNING_BUCKET_N=200`。

可用函数（签名逐字）：

- `fit_one_temperature(pairs: Sequence, min_n: Optional[int] = None) -> float`（calibrate.py:101）
  LBFGS 拟合 log 温度：`log_t = torch.zeros(1, requires_grad=True)`，`LBFGS([log_t], lr=0.1, max_iter=100)`，
  闭包 loss `-(t_mat * torch.log_softmax(z_mat / log_t.exp(), -1)).sum(-1).mean()`；
  返回 `clamp_temperature(..., TEMP_MIN, TEMP_MAX)`；桶内不足 `min_n` 返回 `1.0`。
- `fit_temperature_map(records: Iterable, compute_ece: bool = False, seed: int = 0) -> Dict[str, Any]`（calibrate.py:250）
  返回 `{'temperature': [3 floats], 'temperature_by_options': {bucket: float}, 'n_by_bucket': {...}}`，
  `compute_ece=True` 时另有 `'report'`（含 `ece_before`/`ece_after`/`n`/`n_eval`/`buckets_excluded_from_ece`）。
  类型标量按 `MIN_TYPE_N`、桶按 `MIN_BUCKET_N`。`fit_temperatures` 是它的别名（calibrate.py:297）。
- `fit_abstention_thresholds(records, temperature, temperature_by_options, *, binning_map=None, target_error=0.10, min_bucket_n=MIN_ABSTAIN_BUCKET_N, conservative=True) -> Dict[str, float]`（calibrate.py:338）
- `fit_binning_map(records, temperature, temperature_by_options, *, bins=15, min_bucket_n=MIN_BINNING_BUCKET_N)`（calibrate.py:404）
- `apply_binning_map(confidence, bucket, binning_map)`（calibrate.py:449）
- `records_from_labeled(agent, pairs)`（calibrate.py:465，`@torch.no_grad()`）从 `(state, questions, targets)` 造 `(qtype, logits, target, k)` 记录 —— **我们自己拟温度的入口就是它**。
- `calibration_payload(...)`（calibrate.py:506）与 `apply_calibration_payload(obj, payload)`（calibrate.py:607）分别是 `Agent.save_calibration` 写的 JSON 体与装载函数。

持久化位置：`rl_agent_config.json` 的 `temperature`（3 个 float，顺序 = `QTYPES` 即 choice/score/noul）+ 可选 `temperature_by_options`，桶键形如 `"choice:2"`、`"choice:3-5"`、`"choice:6-10"`、`"choice:11+"`（`laya/common.py:815-817`）。

**代码里的温度钳制是 `TEMP_MIN = 0.5`、`TEMP_MAX = 5.0`（`laya/common.py:824-825`）**，而 `docs/finetune.md:287` 说 notebook 用 `[0.1, 10]`。仓库代码与文档不一致，**以代码为准**：温度不可能低于 0.5。出厂 multilingual 检查点的温度超界被 laya 钳到 0.5 并告警，就是这条路径。

`finetune` 保存时：`out_cfg = dict(cfg, max_len=..., head_max_len=..., fine_tuned=True, temperature=fitted['temperature'])`，**删掉继承来的 `temperature_by_options`**，只有本次拟合产出桶值时才重新写回（`laya/train.py` finetune 主体；理由见 `docs/finetune.md:288-290`：旧桶值在推理时优先，会悄悄掩盖新拟合）。回归测试 `python tests/test_calibration_persistence.py`。

## 5. 显存 / 默认值 / 优化器

- **无 LoRA**。全量微调；唯一Encoder选项是 `--freeze-encoder`（只训决策头，`docs/finetune.md:166` 明说"能塞进小显卡或 CPU，但收益远小于全量微调"）。
- 优化器：`torch.optim.AdamW(groups, weight_decay=0.01)`，两组参数 —— encoder 用 `encoder_lr`，非 `encoder.*` 的用 `head_lr`（`laya/train.py:671-676`）；`CosineAnnealingLR(T_max=optimizer_updates(...), eta_min=min_lr)`（678-681）；梯度范数裁剪 1.0；`GradScaler` 仅 cuda+amp。
- 默认：epochs 4 / micro_batch 8 / grad_accum 8（有效 batch 64，`laya/train.py:81-82`）、encoder_lr 2.5e-5、head_lr 1e-4、min_lr 1e-6、weight_decay 0.01、loss `rlcd`、rl_samples 4、sigma 0.4→0.1、calib_max 400、calib_frac 0.1、calib_seed 20260922、seed 0、amp 自动（仅 CUDA 开）、gradient_checkpointing 跟随 amp。
- 16 GB 卡配方（`docs/finetune.md:257-265`）：epochs 4、有效 batch 64、encoder 2.5e-5 / head 1e-4 AdamW cosine、fp16 autocast + encoder 和 head 都开 gradient checkpointing、grad clip 1.0、`max_len` 1024、`head_max_len` 256、`max_tokens_per_batch` 4096。
- 运行时参考（2×T4）：demo 6,000 decisions 约 4–6 分钟；~30k questions × 4 epoch 约 4–5 小时（`docs/finetune.md:267-268`）。我们 8 GB 单卡，按有效 batch 64 + gradient checkpointing 应该能跑，**先用 3k 样本 smoke 验证显存与时长**。
- 参考成绩（typed-decisions 6,000 train / 2,000 test，单 T4，`docs/finetune.md:183-190`）：

  | run | accuracy | ECE | 选项反转后答案改变率 |
  | --- | --- | --- | --- |
  | base checkpoint | 0.362 | 0.174 | 0.357 |
  | `--loss rlcd`（默认） | 0.7715 | 0.150 | 0.065 |
  | `--loss soft-ce` | 0.790 | 0.157 | 0.078 |
  | `--loss soft-ce --shuffle-options` | 0.7875 | 0.159 | 0.062 |

  20 选项的 MASSIVE 上 base 0.730 → 0.936（无 shuffle）/ 0.941（shuffle）（`docs/finetune.md:192-199`）。
  参考行（`docs/finetune.md:309-314`）：TypeSafe Jev 1.13.0 0.727/0.144、ModernBERT-base(149M) 0.646/0.179、教师自一致上限 0.735、Laya 出厂微调版 0.766。
  **ECE 0.144–0.157 全部高于我们 0.10 的验收线** → #6 必须自己拟合并报 ECE，不能引用这些数字交差。
- 5 选项远小于 #394 报告的 >20 掉准确率门槛，`--shuffle-options` 可不开（`docs/finetune.md:201-204`：4 选项时微调本身就让答案对重排稳定，shuffle 无可测收益）。

## 6. ONNX 导出与量化（repo-only，必须从 GitHub 树取）

入口 `scripts/export_onnx.py`（148 行，**不在 pip 包内**，`pyproject.toml:37` 的 packages 列表不含 scripts）：

- `export_to_onnx(model_id_or_path: str, output_path: str)`（line 57）：`Agent(model_id_or_path, compile=False, device="cpu")`，用 dummy `batch, seq_len, num_markers = 2, 17, 3`（刻意两两不同，避免把 batch=1 烤进去，#695）trace，`opset_version=18`、`do_constant_folding=True`、`dynamic_axes=...`，`input_names=["input_ids","attention_mask","marker_pos","marker_mask","qtype"]`，`output_names=["logits","act_logits"]`。
- `quantize_model(model_path, output_path, per_channel=False) -> str`（line 8）：INT8 weight-only `quantize_dynamic`，`op_types_to_quantize=["MatMul"]`，`weight_type=QuantType.QInt8`；`per_channel` 默认 False，因为 per-channel 会塌模型（#790）。
- `int8_output_path(output_path)`（line 51）：`laya.onnx` → `laya.int8.onnx`。
- CLI：`python scripts/export_onnx.py [--model convaiinnovations/laya] [--output laya.onnx] [--quantize] [--per-channel]`。
- 运行时侧是 `laya/onnx_agent.py`（在 pip 包里）。extra：`laya[onnx]` = `["onnx","onnxruntime","onnxscript", ...]`（`pyproject.toml:70-71`）。
- **#9 的行动项**：把 `scripts/export_onnx.py` 复制进我们的 `scripts/`（保留上游署名），不要 `pip install -e .` 整个仓库。

## 7. Windows / Python 版本

- `pyproject.toml:10 requires-python = ">=3.10"`，classifiers 列 3.10–3.13；依赖 `torch>=2.0.0`、`transformers>=4.48.0`、`safetensors>=0.4.0`、`huggingface_hub>=0.20.0`、`numpy>=1.20.0`。我们的 3.12.12 + torch 2.6.0+cu124 + transformers 5.19.0 全部满足。
- CI 有专门的 `test-windows` job（`.github/workflows/ci.yml`，Python 3.11，`pip install torch` 走默认 PyPI，注释说 Windows 上默认 PyPI torch 就是 CPU wheel）。
- README Windows 安装路径：`py -3.11 -m venv .venv` → `\.venv\Scripts\python.exe -m pip install laya`（README.md:172-178）。
- Windows + Python 3.14 的模型构建崩溃（#123）已在 Laya 0.3.7 修复（README.md:244-248）—— 我们用 3.12，无关。
- 唯一的 Windows 输出坑：`laya/cli.py:398-402` 说 stdout cp1252 编不动非拉丁字符，要 `sys.stdout.reconfigure(encoding="utf-8")`。**我们的脚本统一在 `__main__` 里 reconfigure**。
- 训练本身无 Windows 特例：`docs/finetune.md:23` "runs on one device: CUDA, MPS or CPU"。

## 8. 安装包 vs 仓库

site-packages 与 clone 的 27 个 `.py` 文件逐行数一致（train.py 1177、train_cli.py 224、calibrate.py 612、common.py 824、agent.py 1962、router.py 1656…），仅换行符不同（clone CRLF / install LF），归一化 `\r\n`→`\n` 后逐字节相同。

- **训练、奖励、温度拟合、ONNX 运行时都在 pip 包里**：`laya/train.py`、`laya/train_cli.py`、`laya/calibrate.py`、`laya/common.py`、`laya/onnx_agent.py`。
- **只有 ONNX 导出/量化脚本在仓库里**：`scripts/export_onnx.py`。
- `research/scripts/finetune_single_device.py`、`research/scripts/step4_benchmark/*`、Kaggle notebook 都是同一循环的副本/评测装置，不是 `laya-train` 的依赖。

## 9. 对 mai-jev 管线的直接影响

1. **我们不需要写训练循环**。写 `scripts/train_rlcd.py` 做三件事：把我们的三件套 jsonl 转成 laya 的 `{state, questions, expected}` 行（含 split 与 calib 隔离）→ 调 `laya-train`（或直接 `from laya.train import TrainConfig, finetune`）→ 汇总报告。
2. **数据侧的关键约束**：state 里不能有 `[MASK]`；`expected.urgency` 用整数 level（0-based 还是 1-based 见 `target_from_expected`：整数 level 直接当 index，所以**我们的 1..5 要减 1 变成 0..4**，否则 5 越界会 `ValueError: expected score 5 out of bounds [0, 4]`）。
3. **calib 切片 laya 自己会切**（`calib_max=400`、`calib_frac=0.1`、`calib_seed=20260922`，`laya/train.py:1017 split_calibration`）。我们仍然要在数据层保留自己的 calib/test 集，因为 laya 的切片只有 400 条、按 `docs/finetune.md:353` 自己都说"够拟合三个标量，不够验证"。
4. **ECE 验收必须自己算**：用 `laya.calibrate.records_from_labeled` 在我们的 calib 集上造记录，`fit_temperature_map(..., compute_ece=True)` 拿 `ece_before`/`ece_after`；温度钳在 [0.5, 5.0]，如果拟合后 ECE 仍 >0.10，需要上报而不是引用上游数字。
5. **`--freeze-encoder` 是 8 GB 显存的退路**，但收益小；先试全量 + gradient checkpointing。
6. 放全量前必须跑 `--dry-run` 看 items 数与跳过原因，再跑 3k smoke 验证 ≤1 小时。
