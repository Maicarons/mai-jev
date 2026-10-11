# mai-jev 运行笔记（Runs）

每条都记：跑了什么、结果数字、失败原因、下一步。新记录追加在文件末尾。

---

## 2026-10-12 冒烟管线（#12）

**目的**：在放全量前先验证 `make_smoke_data → build_dataset → train_rlcd convert → laya-train` 这条链路在 8GB 单卡上能跑通，并记录 laya-train 的真实行为。

### 环境

- venv：`G:\mai-jev-work\venv`（Python 3.12.12）
- GPU：RTX 4060 Laptop 8GB，torch 2.6.0+cu124
- 数据：`G:\mai-jev-work\data\`（`v0.1_*`，300 条合成冒烟工单）

### 踩坑（3 个，都已修）

1. `prepare_smoke.py` 里 venv 路径写错：`REPO.parent / "mai-jev-work"` 解析成 `G:\mai-jev-work\...` 之外的东西。
   改成 `Path("G:/mai-jev-work") / "venv" / "Scripts" / "python.exe"`。
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

checkpoint 落在 `G:\mai-jev-work\runs\smoke\`（`model.safetensors` 643MB，含 `checkpoint_latest/`）。

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
