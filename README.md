# mai-jev — 面向低端设备的中文优先 System One 决策模型

> 基座：[`convaiinnovations/laya-multilingual`](https://huggingface.co/convaiinnovations/laya)（322M，mmBERT-base 编码器 + 决策头，Apache-2.0）
> 训练：RLCD（Reinforcement Learning for Calibrated Decisions）+ 温度标定
> 目标靶机：x86 CPU-only 老笔记本 / 中端手机 —— **常驻内存 ≤1GB、4 线程、单问题 p50 < 50ms**
> 发布：Hugging Face `Maicarons/mai-jev-*` 与 ModelScope 同名仓库（URL 待回填，见 #5）

## 这是什么

一个中文优先的 **System One 决策模型**：输入一段中文客服工单（State），同时回答三个具名问题（Question）：

| 原语 | 问题 | 输出 |
| --- | --- | --- |
| `choice` | `route_team` 该路由到哪个团队 | 售后 / 退款处理 / 技术支持 / 账务 / 投诉升级 + 概率分布 |
| `score` | `urgency` 紧急程度 | 1–5 有序等级 + 分布 |
| `noul` | `needs_refund_or_escalation` 是否要求退款或升级 | [0,1] 概率 |

每个回答都带经温度标定的 Confidence，可直接按门限分流（≥0.9 自动执行 / 0.6–0.9 复核 / <0.6 转人工）。术语见 [CONTEXT.md](CONTEXT.md)。

**为什么不是调云端 API**：Jev（TypeSafe AI）是闭源云服务，不支持微调、权重不开放、据报道未对中国大陆开放。我们训练的是同构的开源"Jev 式"决策模型，权重自持、可离线部署在低端设备上。对比研究见 [docs/research/](docs/research/)。

## 快速开始

```bash
# 1. 建环境（torch 必须来自 cu124 索引，PyPI 上的 torch 是 CPU 构建）
uv venv --python 3.12 $WORK\venv
uv pip install --python python -r requirements.txt \
    --index-url https://download.pytorch.org/whl/cu124

# 2. 推理（3 行）
python -c "import laya; r=laya.Router(preload=True); print(r.predict(
    state='用户称上周买的耳机一边没声音，要求退款，语气很急。',
    questions=[{'name':'route_team','type':'choice','instructions':'这条工单该给哪个团队？',
                'criteria':{'售后':'一般售后咨询','退款处理':'要求退款','技术支持':'产品故障排查',
                            '账务':'发票与账单','投诉升级':'要求投诉或升级'}}],
    model='multilingual'))"
```

跑通冒烟与零样本基线（原版权重在中文上接近随机，这是微调的出发点）：

```bash
python tools/smoke_laya.py
```

## 复现训练

```bash
# ① 数据：原始数据集 → 三原语 jsonl（train / calib / test）
python scripts/build_dataset.py --out $WORK\data

# ② 阶段一：≤3k 样本 1 小时内跑通「数据 → RLCD → checkpoint」
python scripts/train_rlcd.py --config runs/smoke/train_config.yaml

# ③ 阶段二：全量（约 3–5 万问题 × 4 epoch，单轮 ≤4 小时）
python scripts/train_rlcd.py --config runs/v0.1/train_config.yaml

# ④ 标定 + 评测：在 calib 留出集上拟合温度，报 ECE / 准确率 / soft accuracy
python scripts/calibrate_and_eval.py --ckpt $WORK\ckpt\v0.1 --data $WORK\data

# ⑤ 导出 + 靶机基准
python scripts/export_onnx.py --ckpt $WORK\ckpt\v0.1 --out $WORK\export
python tools/bench_target.py --model $WORK\export\v0.1-int8.onnx
```

> 脚本状态：`tools/smoke_laya.py` 已可用；`scripts/*` 随 #7 / #12 / #6 / #9 逐步落地，未落地前会明确说明。

## 当前进度

| 票 | 内容 | 状态 |
| --- | --- | --- |
| #2 | 训练环境（G 盘 venv + consumer GPU cu124 校验） | ✅ closed |
| #4 | 训练数据来源与许可证盘点 | 🔬 research 中 |
| #3 | Laya 微调管线与 RLCD 实现拆解 | 🔬 research 中 |
| #8 | 低端设备部署与量化路径 | 🔬 research 中 |
| #11 | HF / ModelScope 发布流程与合规 | 🔬 research 中 |
| #7 | 构建三原语训练数据集 | ⛔ 等 #4 |
| #12 | RLCD 微调（先小样本后全量） | ⛔ 等 #3 #7 |
| #6 | 温度标定与评测（ECE ≤0.10） | ⛔ 等 #12 |
| #9 | 量化导出与靶机基准 | ⛔ 等 #6 #8 |
| #5 | 发布到 HF 与 ModelScope（HITL：需用户 token） | ⛔ 等 #9 #11 |
| #10 | 仓库基建：CONTEXT.md / ADR | 🚧 CONTEXT.md 已写 |

## 验收线

- 准确率超过同规模基线（多数类基线 + 原版 `laya-multilingual`）
- **ECE ≤ 0.10**（在 calib 留出集上温度标定后；出厂权重 ECE ≈ 0.466）
- **中文场景优于原版 `laya-multilingual`**
- 靶机：常驻内存 ≤1GB、单问题 p50 < 50ms

## 环境要点（Windows / 双盘）

- Python 用 uv 的 **3.12.12**（系统默认 3.14 不用；Laya 要求 ≥3.10）。
- torch 必须用 `https://download.pytorch.org/whl/cu124` 装 **2.6.0+cu124**；默认 PyPI 的 `torch==2.14.1` 是 `+cpu` 构建，装上会报 `OSError: [WinError 127] ... torch_cuda.dll`。
- **系统盘空间紧张**：venv、数据、checkpoint、导出全部在 `$WORK\`；仓库只放代码与文档。
- 已知告警（原厂 multilingual 检查点自带温度超界，置信度视为未标定）：`laya: this checkpoint ships invalid temperatures or values outside [0.5, 5]; using choice:11+=0.1006 -> 0.5`。
- GPU 上零样本单问题延迟 22.7–24.5ms（consumer GPU）；靶机 CPU 数字见 #9。

## 许可证与致谢

本项目代码 Apache-2.0；衍生自 [Laya](https://github.com/NandhaKishorM/laya)（作者 Nandakishor M / Convai Innovations，Apache-2.0）。模型卡将保留原作者署名、修改声明与 NOTICE，训练数据出处见 [DATASETS.md](DATASETS.md)（待 #7 落地）。

## 参考

- [docs/research/jev-vs-laya.md](docs/research/) — Jev 与 Laya 的架构、RLCD、基准对比
- <https://wilsonwu.me/blog/2026/jev-vs-laya/>
- <https://typesafe.ai/blog/introducing-system-one-models-and-jev>
- <https://github.com/NandhaKishorM/laya>
- <https://github.com/datawhalechina/jev-cookbook>
