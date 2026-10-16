# mai-jev — 面向低端设备的中文优先 System One 决策模型

> 基座：[`convaiinnovations/laya-multilingual`](https://huggingface.co/convaiinnovations/laya)（322M，mmBERT-base 编码器 + 决策头，Apache-2.0）
> 训练：RLCD（Reinforcement Learning for Calibrated Decisions）+ 温度标定
> 目标靶机：x86 CPU-only 老笔记本 / 中端手机 —— **常驻内存 ≤1GB、4 线程、三原语合计 p50 < 500ms**
> 发布：ModelScope [`Mai2026/mai-jev-zh-support`](https://www.modelscope.cn/models/Mai2026/mai-jev-zh-support) · Hugging Face [`Maicarons/mai-jev-zh-support`](https://huggingface.co/Maicarons/mai-jev-zh-support)

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

每一步的实测输出与踩坑记在 [docs/research/RUNS.md](docs/research/RUNS.md)；论文里的每个数字都能由这些脚本再生（`scripts/make_paper_tables.py` + `scripts/check_numbers.py`）。以 v0.3 为例：

```bash
$env:PYTHONIOENCODING='utf-8'   # Windows 控制台必须，否则中文乱码

# ① 数据：原始数据集 → 三原语 jsonl（train / calib / test）
python scripts/build_dataset.py --raw <原始数据目录> --out $WORK\data --write

# ② 紧急度重标 + 重平衡（#16）：规则证据分级器给 score 合成语义正确的标签
python scripts/label_urgency.py  --in-dir $WORK\data --out-dir $WORK\data\adapted_v03
python scripts/balance_urgency.py --in-dir $WORK\data\adapted_v03 \
    --out $WORK\data\adapted_v03\mai_jev_train_balanced.laya.jsonl

# ③ RLCD 微调：run 结束会把超参 + 数据 sha256 + 显存峰值写进 runs/<id>/train_config.yaml
python scripts/train_rlcd.py run \
    --data $WORK\data\adapted_v03\mai_jev_train_balanced.laya.jsonl \
    --eval-data $WORK\data\adapted_v03\mai_jev_test.laya.jsonl \
    --out $WORK\runs\v03_rlcd_bal --base multilingual --loss rlcd \
    --epochs 2 --micro-batch 8 --grad-accum 4 --seed 20261013

# ④ 标定 + 评测：calib 上拟合温度，test 上报准确率 / ECE / 置信度三档
python scripts/calibrate_eval.py --checkpoint $WORK\runs\v03_rlcd_bal \
    --calib-data $WORK\data\adapted_v03\mai_jev_calib.laya.jsonl \
    --test-data  $WORK\data\adapted_v03\mai_jev_test.laya.jsonl \
    --baseline-checkpoint convaiinnovations/laya-multilingual \
    --out $WORK\runs\v03_rlcd_bal

# ⑤ 导出 → 量化 → 组装发布目录（head 装不下 max_len 会直接报错退出，不是静默降级）
python scripts/export_onnx.py --model-dir $WORK\runs\v03_rlcd_bal --out-dir $WORK\onnx
python scripts/calibrate_export.py --model-dir <量化变体目录> --calib <calib jsonl> --write
python scripts/build_export.py --model-dir $WORK\runs\v03_rlcd_bal --fp32 <fp32.onnx> \
    --calibration <该变体 calibration.json> --out $WORK\export\v03-int8-ml256 \
    --variant int8-gather --max-len 256

# ⑥ 验收口径（与靶机验收线同一口径）：一条工单跑完三原语的 p50/p95 + 常驻内存
python -m mai_jev.deploy --model-dir $WORK\export\v03-int8-ml256 ^
    --input $WORK\export\bench_input.jsonl --threads 4 --repeats 150 --warmup 8

# ⑦ 发布 + 发布后验收（发布需用户 token，见 #5）
python scripts/publish_model.py --source modelscope --stage <staging 目录>
python scripts/verify_published_model.py --source modelscope
```

> **温度是「权重 × 推理后端」的联合属性**：换量化变体必须用 `scripts/calibrate_export.py` 在**该变体自己的 calib** 上重拟合，并把 `calibration.json` 与 `.onnx` 放在同一目录 —— 跨变体复用标定会让 ECE 悄悄退回未标定水平。

## 当前进度

| 票 | 内容 | 状态 |
| --- | --- | --- |
| #1 | 地图：Destination / Notes / Decisions / Fog | 🗺️ map |
| #2 | 训练环境（G 盘 venv + consumer GPU cu124 校验） | ✅ closed |
| #3 | Laya 微调管线与 RLCD 实现拆解 | ✅ closed |
| #4 | 训练数据来源与许可证盘点 | ✅ closed |
| #7 | 构建三原语训练数据集 | ✅ closed（v0.2 3,209 行） |
| #12 | RLCD 微调（先小样本后全量） | ✅ closed（v0.2 全量轮） |
| #6 | 温度标定与评测（ECE ≤0.10） | ✅ closed（v0.2：宏平均 0.7823，最差 ECE 0.0950） |
| #16 | 补齐语义正确的紧急度标注并重训 score 原语 | ✅ closed（v0.3：宏平均 0.9106，最差 ECE 0.0871） |
| #8 | 低端设备部署与量化路径 | ✅ closed（int8-gather 666MB / 常驻 796MB） |
| #9 | 量化导出与靶机基准 | ✅ closed（发布变体 v03-int8-ml256；延迟口径改为三原语合计 p50<500ms） |
| #11 | HF / ModelScope 发布流程与合规 | ✅ closed |
| #15 | 撰写 arXiv 格式论文（数字全部可再生） | ✅ closed（`paper/`） |
| #10 | 仓库基建：CONTEXT.md / ADR / README / DATASETS.md | ✅ closed |
| #5 | 发布到 HF 与 ModelScope | ✅ closed（两地均上线，`verify_published_model.py` 双源通过） |

## 验收线

- 准确率：held-out test（343 行，从未参与训练或标定）宏平均 **0.9106** = route_team 0.9475 / urgency 0.9329 / noul 0.8513。
- **ECE ≤ 0.10**：每个变体在**自己的 calib 切分**上重拟合温度后，torch 最差 0.0871、int8 最差 0.0927。
- **中文场景优于原版**：宏平均 0.4052（原版 `laya-multilingual` 零样本）→ 0.9106；规则基线 0.4373/0.5918/0.4111、多数类 0.3586/0.7026/0.6356。
- 靶机：常驻内存 ≤1GB ✅（int8-gather 796MB）；延迟口径为「一条工单跑完三原语合计 **p50 < 500ms**」——训练机上端到端实测 **p50 421–499ms / p95 862–976ms**（n=150，两次复测）。**余量很小，且真实靶机只会更慢，复测前不作为稳定达标宣称**。原「单问题 p50 < 50ms」口径已由 #9 作废，保留给未来 ≤500MB 蒸馏档位。

## 环境要点（Windows / 双盘）

- Python 用 uv 的 **3.12.12**（系统默认 3.14 不用；Laya 要求 ≥3.10）。
- torch 必须用 `https://download.pytorch.org/whl/cu124` 装 **2.6.0+cu124**；默认 PyPI 的 `torch==2.14.1` 是 `+cpu` 构建，装上会报 `OSError: [WinError 127] ... torch_cuda.dll`。
- **系统盘空间紧张**：venv、数据、checkpoint、导出全部在 `$WORK\`；仓库只放代码与文档。
- 已知告警（原厂 multilingual 检查点自带温度超界，置信度视为未标定）：`laya: this checkpoint ships invalid temperatures or values outside [0.5, 5]; using choice:11+=0.1006 -> 0.5`。
- GPU 上零样本单问题延迟 22.7–24.5ms（consumer GPU）；靶机 CPU 数字见 #9。

## 许可证与致谢

本项目代码 Apache-2.0；衍生自 [Laya](https://github.com/NandhaKishorM/laya)（作者 Nandakishor M / Convai Innovations，Apache-2.0）。模型卡将保留原作者署名、修改声明与 NOTICE，训练数据出处、许可义务与署名声明见 [DATASETS.md](DATASETS.md)。

## 参考

- [docs/research/jev-vs-laya.md](docs/research/) — Jev 与 Laya 的架构、RLCD、基准对比
- <https://wilsonwu.me/blog/2026/jev-vs-laya/>
- <https://typesafe.ai/blog/introducing-system-one-models-and-jev>
- <https://github.com/NandhaKishorM/laya>
- <https://github.com/datawhalechina/jev-cookbook>
