# mai-jev 运行笔记（Runs）

每条都记：跑了什么、结果数字、失败原因、下一步。新记录追加在文件末尾。

---

## 2026-10-12 冒烟管线（#12）

**目的**：在放全量前先验证 `make_smoke_data → build_dataset → train_rlcd convert → laya-train` 这条链路在 8GB 单卡上能跑通，并记录 laya-train 的真实行为。

### 环境

- venv：`$WORK\venv`（Python 3.12.12）
- GPU：consumer GPU (8GB VRAM)，torch 2.6.0+cu124
- 数据：`$WORK\data\`（`v0.1_*`，300 条合成冒烟工单）

### 踩坑（3 个，都已修）

1. `prepare_smoke.py` 里 venv 路径写错：`REPO.parent / "mai-jev-work"` 解析成 `$WORK\...` 之外的东西。
   改成 `Path("$WORK") / "venv" / "Scripts" / "python.exe"`。
2. `subprocess.run(["laya-train", ...])` → `FileNotFoundError [WinError 2]`：venv 的
   `Scripts\laya-train.exe` 不在 PATH。
3. `python -m laya train` → `No module named laya.__main__`：laya 包没有 `__main__.py`。
   **最终方案**：`[sys.executable, "-c", "import sys;from laya.cli import main;sys.exit(main(['train'] + sys.argv[1:]))"]`
   —— 和 `laya-train.exe` 走同一个 `main()`（`laya/cli.py:394-396` 转发到 `train_cli.main`）。

### dry-run（--max-samples 200）

```
dry-run: eval 30 rows read, 90 valid items, skipped: {}
dry-run: 240 rows read, 720 valid items, skipped: {}
dry-run: optimizer budget: 648 train item(s), effective batch up to 64 (micro_batch 8 x grad_accum 8), 11 optimizer updates over 1 epoch(s)
```

720 = 240 行 × 3 问题；`items` 是 question 级不是 case 级 —— **估时长要用 items 数，不是行数**。

### 真实跑（1 epoch，648 train items，72 calib，90 eval）

```
epoch 1/1 mean loss 0.5949
Accuracy               0.9778         0.6111         -0.3667
Loss                   0.3006         1.6561         +1.3555
ECE                    0.2145         0.2050         -0.0095
Brier (multi-class)    0.1477         0.5481         +0.4004
Brier (top-1)          0.0864         0.1649         +0.0785
Mean Confidence        0.7633         0.8091         +0.0458
```

checkpoint 落在 `$WORK\runs\smoke\`（`model.safetensors` 643MB，含 `checkpoint_latest/`）。

### 关键观察

- **精度大幅下降是预期的**：11 个 optimizer update 训不动 322M 模型，且 encoder_lr 2.5e-5 在
  648 items 上等于没训。这条链路验证的是"能跑通"，不是"能学好"。
- **三个类型温度全部被钳到 0.5**：calib 只有 72 items（choice 24 / score 22 / noul 26），
  `MIN_TYPE_N=10` 够，但 LBFGS 拟合出来就在下界。温度标定必须用更大的 calib 集。
- `noul` 的 `mean_confidence` 到 1.0、ECE 0.0：合成数据里 noul 完全由 route_team 决定，
  模型很快过拟合到这个捷径。**真实数据不会这么干净**。
- 告警 `Detected call of lr_scheduler.step() before optimizer.step()` 来自 laya 自己
  （`train.py` 的调用顺序），不是我们的问题，忽略。

### 自己的评测脚本（`scripts/eval_checkpoint.py`）

按 question id 聚合（不是按 choice/score/noul），并对 urgency 报 off-by-one 容差。
同一份 test 上（30 条）：

| 指标 | 规则基线 | 冒烟 checkpoint |
| --- | --- | --- |
| route_team | 0.867 | 0.900 |
| urgency | 0.300（off-by-one 0.433） | 0.000（off-by-one 0.400） |
| needs_refund_or_escalation | 1.000 | 0.000 |

冒烟 checkpoint 的 urgency/noul 崩了 —— 又一次印证 11 个 update 不够，需要全量 + 更多 epoch。

### 下一步

1. 等 #4 数据集审计结果，确定真实中文工单数据来源。
2. 全量跑之前先把 calib 集做大（目标 ≥2000 items 让 `MIN_BUCKET_N=2000` 也能满足）。
3. 全量配置按 `docs/finetune.md:267-268` 的 2×T4 是 4–6 小时，我们单卡 8GB 必须
   `--epochs 2 --grad-accum 4` 或 `--freeze-encoder` 才可能压进 4 小时预算。

---

## 2026-10-12 laya 仓库 / 安装包审计结论（3 个子代理，互相印证）

- **训练代码就在 pip 包里**（`laya/train.py` 1331 行 + `train_cli.py` 239 行，与仓库逐行一致，仅换行符不同）。
  之前"训练入口在仓库里"的说法作废。
- **"GRPO" 只是标签，不是实现**：`rlcd_loss`（`laya/train.py:499-518`）是 REINFORCE +
  **单 item 自身的 4 个噪声样本**做基线（`adv = r - r.mean(0)`），没有 prompt group、
  没有组归一化、没有 PPO clip/ratio/KL。→ CONTEXT.md 的术语表要改口：说"带样本基线的
  策略梯度 + 严格适当评分规则奖励"，不要写"GRPO 组基线"。
- **[0.5, 5.0] 温度钳制是权威代码**（`laya/common.py:824-825`），`docs/finetune.md:287` 的
  `[0.1, 10]` 是过期文档。
- `answer_confidence = max(p)`（标定量，ECE 测的就是它）；`confidence_from_probs = 1 - H(p)/log(k)`
  （未标定）—— **两者不能用同一个阈值门控**。
- `load_calibration` 读一个没有 `binning_map` 的文件会**清掉**已拟合的 binning：文件是全量状态不是补丁。
- ONNX 导出脚本确实只在仓库里（`scripts/export_onnx.py`），opset 18，dummy batch/seq/markers = 2/17/3；
  INT8 dynamic MatMul 只能在 CPU 上跑（CUDAExecutionProvider 没有 INT8 MatMul kernel）；
  `per_channel` 必须保持 False（#790 会让模型塌）。
- Windows GPU 微调无硬阻塞：CI 有 windows-latest lane；要避开的是
  `HF_HUB_DISABLE_SYMLINKS=1`、`USE_TF=0`、`TOKENIZERS_PARALLELISM=false`、cp1252 stdout。
- **≤4h / 30k 预算是真实风险**：文档实测 2×T4 要 4–6 小时，单卡 8GB 要 `--epochs 2 --grad-accum 4` 或 `--freeze-encoder`。
- 许可证 Apache-2.0，无 NOTICE 文件，署名用 "Convai Innovations"。
- 第三方基准数据要单独跟许可证走：`research/benchmarks/feishu_zh/LICENSE` 是 MIT / (c) 2026 Adkid-Zephyr。

---

## 2026-10-17 发布变体的延迟复测（复核 model card / #5 的 380ms）

**目的**：核查对外发布页上「三原语合计 p50 = 380ms」的口径来源 —— 它和论文
`paper/generated/numbers.tex` 里 `\LatPFiveZero{498.65}`（来源 `export/v03_bench_final.json`）
相差 31%，必有一个是错的。

**结论：380ms 出自一个已经作废的变体，不能用于发布变体。**

- 380ms 的来源是 `export/bench_report.json` 的 `thread_scaling_int8_gather_max_len_96`
  表（threads=4 → p50 378.69ms），那是 **`mai-jev-int8-ml96`**。该变体的 `max_len=96`
  装不下 v03 的 question head（109/110 token），整条工单正文被丢弃 —— 同一目录下
  `v03c.log`/`v03e.log` 实测 `route_team` 0.945 → **0.1826**。用它测延迟等于测一个
  不读工单的模型。
- 发布变体是 **`v03-int8-ml256`**（int8-gather，666.0MB），端到端复测（本机 desktop-class CPU，
  4 线程，batch=1，精确长度，150 条真实工单，一条工单跑三原语）：

  | 运行 | p50 | p95 | mean | 常驻 RSS | load |
  | --- | --- | --- | --- | --- | --- |
  | `v03_bench_final.json`（2026-10-13） | 498.65 | 975.58 | 583.81 | 795.4 | 3.87s |
  | 本次复测（2026-10-17） | **420.93** | 862.24 | 530.17 | 796.2 | 3.85s |

  复现命令（`--input`/`--repeats` 就是 `deploy.py` CLI 自己那段计时，口径与验收线同一处代码）：
  `python -m mai_jev.deploy --model-dir $WORK\export\v03-int8-ml256
   --input $WORK\export\bench_input.jsonl --threads 4 --repeats 150 --warmup 8`

**两条要记住的**：

1. 「三原语合计 p50 < 500ms」**只有 0–16% 余量且逐次运行漂移**（421 ↔ 499ms，同一台机、同一输入），
   p95 是 1.7–2.0× 超线。对外只能报区间 + 「训练机上」这个限定，不能报单点、不能称稳定达标。
2. 发布页（`docs/model-card.md`，即 ModelScope 的 README）当时报的「p50 380ms / 1029 条工单」
   两头都不对：变体错（ml96）、样本数错（复测是 n=150；`bench_report.json` 的 workload 是
   200 条 / repeats 40）。已在 2026-10-17 的提交里改成发布变体自己的数字并写明来源。

---

## 2026-10-17 两地发布完成 + 发布页数字更正（#5）

**HF 那个「缺 token」的阻塞消失了，而且原因不明 —— 如实记下来。**

- `.env` 里的 `hf-key`（`hf_y…`）**现在对 HF 有效**：`HfApi.whoami()` → `Maicarons`，
  `auth.accessToken.displayName = kaggle`、`role = write`。
- 但 2026-10-14 10:21 那次 `create_repo` 确实报了 401「Invalid username or password」，
  而 `.env` 的 mtime 是 **2026-10-14 10:00:54**（之后未改动）—— 同一份凭证，一次失败一次成功，
  **401 现在无法复现**。可能是一次性的服务侧/网络故障，不是凭证问题。结论：**不要**再把
  「ModelScope token 当成 HF token」当成已证实的解释，那只是当时的猜测。
- 发布后 HF 增量重传 13.4s、首传 179.6s（1.31 GB / 21 文件）。

**发布路径进了仓库。** 以前上传靠工作目录里的两个一次性脚本（`publish_mai_jev.py` /
`publish_ms.py`），它们不在版本库里 —— 也就是说「仓库自带可复现的发布」当时并不成立。
现在 `scripts/publish_model.py`（`--source hf|modelscope|both`、`--dry-run`、按 source 取默认
repo、token 只从进程环境或 `--env-file` 读且不回显）配合已有的
`scripts/verify_published_model.py` 构成 组装→上传→验收 三步，彼此分开。
顺手修了 `verify_published_model.py` 的一个真实缺陷：它的 `--repo` 默认值是 **HF 的 owner**，
所以单独跑 `--source modelscope` 会去 `Maicarons/...` 找模型并拿到 404 —— 现在默认值按 source 分。

**误传的非产物文件**：`.ms_upload_cache`（ModelScope 上传器的缓存）被推到了 HF，发布页上多出
一个和模型无关的文件。已加 `IGNORE_PATTERNS` / `STRAY_PATTERNS`，HF 侧显式 `delete_file` 删掉、
MS 侧用 `delete_files(delete_patterns=...)` 清掉 —— 两地现在都是**恰好 21 个文件**。

**更正了两处「数字来自已作废变体」的发布页事实错误（这才是本次最要紧的事）：**

1. **int8 精度行** 原为 0.9096 / 0.4606 / 0.8397 —— 那是**整图 int8**（max|dlogit| 2.31，
   #8/#9 已弃用）在 v0.2 数据上的数。发布变体 `int8-gather` 在 v0.3 test 上是
   0.9475 / 0.9359 / 0.8484，量化退化 ≤0.0029。已改。
2. **延迟** 原为「p50 380ms / 1029 条工单」—— 见上一节，出自作废变体 ml96，且样本数也不对。
   改为发布变体自己的端到端区间 p50 421–499ms / p95 862–976ms + 「训练机上」限定。

产物侧同步：staging 的 `README.md`（= `docs/model-card.md`）与 `onnx/deploy_config.json` 的
`parity_note`（原文说「不重拟合温度」，与同目录 `calibration.json` 里 int8 refit 的实际值矛盾）
已更正，`MANIFEST.json` 里该文件的 sha256 同步重算（`model.onnx` 的 sha256 **未变**，图的字节
没动），两地重传。`deploy_config.json` 的三条口径声明已提升为 `build_export.py` 的模块级常量，
以后修正发布声明直接 import，不靠手抄。

**验收（`scripts/verify_published_model.py`，两地 exit 0）**

| 源 | repo | 文件 | 校验 |
| --- | --- | --- | --- |
| HF | `Maicarons/mai-jev-zh-support` | 21 / 1.31 GB | safetensors 614.0MB / 170 × F16；`laya.load` 成功；三原语合法；ONNX session 输入五项 |
| ModelScope | `Mai2026/mai-jev-zh-support` | 21 / 1.31 GB | 同上 |

⇒ **发布这步完成了**（#5 可闭）。但挂在上面的延迟问题没有随之解决：Destination 的
「三原语合计 p50 < 500ms」在发布变体上只有 0–16% 余量、p95 是 1.7–2.0× 超线，且**只在训练机
上测过**。真实靶机复测之前，这条验收线只能报区间 + 限定，不能报达标。

