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

## 2026-10-17 第二个 proxy 标签：`noul` 重标 + v0.4 重训（#17）

### 为什么：旧 `noul` 头是第二个 urgency 头

上游 zh-tw 的 `needs_human` 与 zh-bench 的 `escalate` 问的都是「**是否需要人工介入**」，
而我们的问句是「客户是否提出**退款 / 赔付 / 升级处理**的诉求」。三个 split 的正例率**都是 ~64%**
（不是旧文档写的「train 65% / test 47%」—— 那个 47% 是 zh-bench 那 34 行），而正例里约 **70%
连退款 / 赔付 / 升级的字眼都没有**，典型是「我訂的帽子卡在海關，急死我了，請快幫我查清楚！」。
⇒ 旧的 `noul` 头学到的是「客户是不是在着急」，`0.8513` 是这个意思的数字，不是「退款诉求准确率」。

用户拍板的语义口径（写进 `CONTEXT.md`，标注 / 评测 / 发布文案一律照它）：明说、**升级威胁**、
**问资格**（`定金能退吗`）都算 `true`（宽松口径）；纯流程 / 规则问句、只有情绪催促算 `false`；
客户明说「随便问问 / 东西没问题」也算 `false`（那是**没有诉求**，不是问资格）。

### 怎么做：规则标注器 + 人写反例集硬门禁（不用 teacher）

`scripts/label_noul.py`：分级证据阶梯（明说 → 独立诉求 → 追讨已申请 / 已承诺的退款 →
无名词资格询问 → 择一诉求 → 对象窗口 → 礼貌诉求 → 规则问句 → 无证据）+ **37 条人工反例集**，
rubric 与反例集不一致就**直接 abort**，不产出数据。关键词基线（`baseline_rules.py`）在这 37 条上
只答对 **12/37**，说明 rubric 不是同一套关键词。

眼验 + 探针共抓出 7 类真 bug（都固化成反例）：

| # | 形态 | 例子 | 后果 / 修法 |
| --- | --- | --- | --- |
| 1 | 订单号被当消协热线 | `123456789` / `#A12345` | 热线的字母数字边界 |
| 2 | 单字 `查` 误杀 | `我要马上查明跟赔偿` | `LEARN_CUE` 加窗口 |
| 3 | 疑问形式成片漏报 | `還能退貨嗎` / `可以辦理退貨退款嗎` | 资格窗口含对象本身 |
| 4 | `我要的是换货` 误判 | 换货 / 補寄 不是我们口径的对象 | 只作择一诉求的配对词 |
| 5 | 裸 `是要` 误伤 | `退货时赠品…是要一起退还还是直接送我？` | 移出 WILL_CUE |
| 6 | 追讨已申请退款漏报 | `退款怎么还没到账` / `说好的补偿没消息` | 新增 `REFUND_PENDING` + 学习动词守卫 |
| 7 | 裸「能退吗」漏报 | `定金能退吗` | 新增 `STANDALONE_ELIGIBILITY` + ±12 字窗口 |

**重标后的分布**（`label_noul.py --in-dir data\adapted_v03 --out-dir data\adapted_v04`）：

| split | 正例率 前 → 后 | true / false | 旧标签多数类 → 新标签多数类 |
| --- | --- | --- | --- |
| train 2581 | 0.655 → **0.1999** | 516 / 2065 | 0.655 → **0.8001** |
| calib 285 | 0.6351 → **0.214** | — | — |
| test 343 | 0.6356 → **0.2041** | 70 / 273 | 0.6356 → **0.7959** |

三个 split 互相一致 ⇒ 不是 split artifact。**多数类被翻转**（现在「一律答 false」拿 0.796），
所以 `noul` 的裸准确率从此不可作判据，一律报 balanced accuracy。

### 单变量验证：与 v0.3 只差一个标签字段

`data\adapted_v04\mai_jev_train_balanced.jsonl`（9425 行）与 v0.3 的**逐行 id 顺序、state 文本、
route_team gold、urgency gold 完全相同**，只有 **4337 行**的 `noul` gold 不同
（3926 true→false、411 false→true）；`questions` 里的 `instructions` 文本两版相同
（刻意**只改标签、不改 prompt**）。均衡文件里的 `noul` 正例率是 0.4157（urgency 均衡重采样后的比例，
不是 0.20）—— 引用时要注意是哪个文件。

⇒ 训练超参与 v0.3 **逐项相同**（`train_config.yaml` 只有路径与 sha 不同），可归因到标签：

```
python scripts/train_rlcd.py run \
  --data  $WORK\data\adapted_v04\mai_jev_train_balanced.laya.jsonl   # sha e6e81e3cf2525e05
  --eval-data $WORK\data\adapted_v04\mai_jev_test.laya.jsonl          # sha 24b9ccfe67f504c4
  --out $WORK\runs\v04_rlcd_noul --base multilingual --loss rlcd \
  --epochs 2 --micro-batch 8 --grad-accum 4 --encoder-lr 2.5e-05 --head-lr 0.0001 \
  --max-len 0 --seed 20261013 --device auto
```

（v0.3 的对应 sha：train `ccd2ca6cf71b0ec9`、test `e0508bd4dc4c5968`。）

论文里的数字来自这三份评测：

```powershell
$py='python'; $d='$WORK\data\adapted_v04'

# 1) v0.4 自己：主表 + 置信度分档 + 规则基线 / 多数类 / 原版零样本对照
& $py scripts\calibrate_eval.py --checkpoint $WORK\runs\v04_rlcd_noul `
  --calib-data "$d\mai_jev_calib.jsonl" --test-data "$d\mai_jev_test.jsonl" `
  --baseline-checkpoint convaiinnovations/laya-multilingual `
  --out $WORK\runs\v04_rlcd_noul\paper_eval

# 2) 旧 v0.3 头 × v0.4 的新标签：量化「旧 0.8513 里有多少是标签定义撑的」
& $py scripts\calibrate_eval.py --checkpoint $WORK\runs\v03_rlcd_bal `
  --calib-data "$d\mai_jev_calib.jsonl" --test-data "$d\mai_jev_test.jsonl" `
  --out $WORK\runs\v03_on_v04

# 3) 37 条人工反例集：只带 noul 一个问题，通用脚本吃不下（缺另外两个原语的 gold）
& $py scripts\eval_noul_antibody.py --checkpoint $WORK\runs\v04_rlcd_noul `
  --antibody "$d\mai_jev_noul_antibody.jsonl" `
  --calibration $WORK\runs\v04_rlcd_noul\paper_eval\calibration.json `
  --out $WORK\runs\v04_antibody\summary.json
```

### 结果（test 343 行，标定后；`runs\v04_rlcd_noul\paper_eval\`）

| 原语 | v0.3（旧 noul 标签） | v0.4（新 noul 标签） |
| --- | --- | --- |
| route_team acc / bal | 0.9475 / 0.9561 | **0.9300 / 0.9378** |
| urgency acc / bal | 0.9329 / 0.8009 | **0.9475 / 0.8555** |
| noul acc / bal | 0.8513 / 0.8506（**旧口径，不可比**） | **0.9300 / 0.8923** |
| 宏平均 acc | 0.9106 | **0.9359** |
| 最差单原语 ECE | 0.0871 | **0.1061（超线，见下）** |

新标签下同一批 343 行的基线（`baseline_rules.py` 现在也报 `balanced_accuracy`）：

| 对照 | acc | balanced |
| --- | --- | --- |
| 微调后（v0.4） | 0.9300 | **0.8923** |
| 规则基线关键词表 | 0.8484 | 0.6711 |
| 多数类（一律 false） | 0.7959 | 0.5000（构造上） |
| 原版 laya-multilingual 零样本 | 0.7055 | 0.7407 |

⇒ **判据满足**：balanced accuracy 0.8923 显著高于关键词表的 0.6711（+0.22），也高于原版零样本的 0.7407。
但裸准确率只比关键词表高 0.082 —— 这个原语比旧口径「词汇化」得多，**必须报 balanced**。

### 旧 v0.3 头 × 新标签：0.8513 里有多少是标签定义

`runs\v03_on_v04\`（同一个 v0.3 checkpoint，在 v0.4 的 calib 上重拟合温度、v0.4 的 test 上评）：

| 原语 | acc | balanced |
| --- | --- | --- |
| route_team | 0.9475 | 0.9561 |
| urgency | 0.9329 | 0.8009 |
| **noul** | **0.4898** | **0.5733** |

⇒ 旧头在新标签上 **0.4898 < 多数类 0.7959**，balanced 0.5733 也低于关键词表 0.6711。
**旧 0.8513 主要是标签定义撑起来的**：同一个权重，换成诚实的问句就跌到常数基线以下。
这也解释了为什么「重标 + 重训」是必需的 —— 只重标不重训，等于把一个答错问题的头留在线上。

### 反例集上量模型：一个必须记住的负面结果

`scripts/eval_noul_antibody.py`（37 行手写反例**不在任何 split 里**，`0/37` 出现在
train / calib / test / balanced 中，是**分布外**的对抗切片）：

| 评分方 | acc | balanced |
| --- | --- | --- |
| v0.4 头 | 0.4865 | **0.4583** |
| v0.3 头（旧 proxy） | 0.5946 | 0.5387 |
| 规则基线关键词表 | 0.3243 | 0.3080 |
| 多数类（一律 false） | 0.5676 | 0.5000 |

⇒ 模型在自然 test 上 balanced 0.8923（true 类召回 ~0.8），在**把表面词与语义分开**的 37 行上
balanced 0.4583（true 类召回 0.25），**低于常数基线**。诚实结论：**标签修好了，学到的概念只修好一半** ——
2,581 行训练数据足以让头复述新标签的分布，不足以让它学通口径里那些「有字面词却是 false、
没字面词却是 true」的边界。要真正解决需要更多这样的对抗样本进训练集（或做数据增广），
这是 #17 之后仍然开着的口子，已写进 model card 的 Limitations。

### ECE 退线：0.1061 > 0.10，且诊断为小样本噪声

| 原语 | 标定前 ECE | 标定后 ECE | 拟合温度 |
| --- | --- | --- | --- |
| route_team | 0.0466 | **0.1061** | 1.1057 |
| urgency | 0.0514 | 0.0237 | 3.1359 |
| noul | 0.0330 | 0.0707 | 1.2252 |

choice 的 NLL-最优温度把 test 上的平均置信度从 0.898 压到 0.864，而准确率仍是 0.930 ⇒
ECE 从 0.0466 涨到 0.1061。v0.3 是**同一个方向、几乎同一个温度**（1.1131，0.0527 → 0.0871）：
285 行 calib 与 343 行 test 的准确率差出几个点，温度迁移不好，**这条「≤0.10」在 n=343 上是噪声级别**，
v0.4 只是把噪声落在了线的另一侧 0.006。我们**没有**按 test ECE 选温度（那是拿 test 调参）。
要真正解决得加大 calib 切分或按 ECE 而非 NLL 选温度 —— 那是标定方法学（#6）的改动，不在本票范围内。

### 环境坑：GPU 争用（两次失败的真正原因）

- 第一次训练直接 `CUDA error: out of memory`：8 GB 显存里 **7 GB 被同机 ComfyUI 占着**。
- 释放后跑到 step 200 又 `CUDA error: unspecified launch failure` —— 用户当时正在跑 1024×1024 生成任务。
- 用 ComfyUI 自己的 `POST /free`（`{"unload_models":true,"free_memory":true}`，
  可逆）把占用降到 **180 MiB** 后，训练与导出全部正常。
- ⇒ **动手训练/导出前先看 `nvidia-smi`**：显存占用 > 2.5 GB 就先谈，不要硬跑。

### 打包链路（v0.4 的 int8 包）没跑通：不是代码问题，是机器内存

- `export_onnx.py` 成功：`laya_fp32.onnx` **1228.60 MB**，eager parity `max|dlogits| 3.3e-06`，
  且与 v0.3 的导出**字节数完全相同** ⇒ 图结构与参数规模一致，只有 `noul` 头权重变了。
- `build_export.py` 失败在 onnxruntime 的 `load_model_with_shape_infer` →
  `onnx.shape_inference.infer_shapes_path`：在 1.2 GB 的图上做形状推理要数 GB 空闲内存，
  而当时本机**空闲物理内存只剩几个 GB、commit 已接近上限**，它报
  `MemoryError: bad allocation`，写出半截模型，紧接着 `get_opset_version` 报
  `Failed to find proper ai.onnx domain`。**那不是 opset 版本问题**（单独跑 `infer_shapes_path`
  在内存充足时 opset_import 是正常的 `[('', 17)]`），是内存不足的次生症状。
- 当时的机器状态：ComfyUI 的 python + 一批 `blender-mcp` 进程（另一件活儿）把 commit 占满，
  PowerShell 自己都起不来（`800705af` = ERROR_COMMITMENT_LIMIT）。
- ⇒ **动手训练 / 导出 / 评测前先看 `FreePhysicalMemory`，低于 ~6 GB 就先谈**，不要硬跑。
- 因此论文的打包 / 量化 / 延迟数字**仍标注为 v0.3 包**（论文 §5.2 加了 Version note 说明
  v0.4 只换 `noul` 头、导出图大小相同），v0.4 的 int8 变体等机器有空闲内存时再补测。

### 评测产物的一次修补（记录在案）

- `baseline_rules.evaluate` 在 #17 才补上 urgency 的 balanced accuracy，而当时已有的两份
  `calibration_eval.json` 是补之前跑的。规则基线是**纯 CPU、无模型**的确定性函数，所以在同一批
  test 行上用 `calibrate_eval.gold_label` 复算、只替换 JSON 里 `baseline_rules` 这一个键 ——
  复算出的 choice / noul 与原有值逐位相同（脚本据此自校验）才写入。
- 模型侧的评测**没有重跑**（撞上上面那堵内存墙），但 `baseline_rules` 与模型无关，这一步是等价的。
  顺带量到：urgency 的规则基线均衡准确率只有 **0.2630**，而它的 accuracy（0.5918）比多数类（0.7026）还低 ——
  关键词表在这个原语上其实很弱，论文 §5.1 把这个反过来的偏斜也写了出来。

### 产物

- 数据：`data\adapted_v04\`（三 split + balanced + `mai_jev_noul_antibody.jsonl` +
  `relabel_report_noul.json` + `baseline_on_new_labels.txt` + `zhbench_noul_review.md` + `miss_classes_probe.md`）。
- 训练：`runs\v04_rlcd_noul\`（`train_config.yaml` / `train_report.json` / `model.safetensors`）。
- 评测：`runs\v04_rlcd_noul\paper_eval\`（论文数字；命令见上「重跑评测」）、`runs\v03_on_v04\`
  （旧头 × 新标签）、`runs\v04_antibody\summary.json`（反例集，`eval_noul_antibody.py --out`）。
- 论文：`paper/mai-jev.tex` + `paper/generated/`（宏由 `make_paper_tables.py` 从上面这些 JSON 生成；
  v0.4 需要 `--noul-json` / `--noul-oldhead-json` / `--antibody-json` 三个新入参）。


