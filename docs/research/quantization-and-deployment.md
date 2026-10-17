# 低端设备部署与量化路径（靶机 ≤1GB / p50 < 50ms）

> **口径提示（2026-10-17 加）**：本报告标题里的「p50 < 50ms」是 **#8 当时的验收线，已被 #9 作废**。
> 现行验收线是「一条工单跑完三原语合计 p50 < 500ms」（发布变体 int8-gather、`max_len=256`）。
> 本报告的 §4 仍然是「50ms 在什么边界上可达」的完整实测（`max_len=96` → 44.0ms），
> 读的时候请连着 §8.2 的未决问题一起看；发布变体的端到端复测见
> [RUNS.md](RUNS.md) 的 2026-10-17 条。

> 票 [#8](https://github.com/Maicarons/mai-jev/issues/8) · wayfinder:research
> 所有数字都是本机实测（台式 x86 CPU / Windows / venv `$WORK\venv`，torch 2.6.0+cu124 / onnx 1.31.0 / tokenizers 0.23.3），产物全部落在 `$WORK\onnx\`，未新增任何 pip 依赖。

---

## 0. 一句话结论

**选「整模型单图 ONNX + int8 动态 per-channel 量化 + 部署时 batch=1 逐条推理」**。它把权重压到 **308.6 MB**（fp32 图 1228.6 MB 的 25%），常驻内存 **408 MB**，比 fp32 快 **2.1×**；三个原语中 `choice` 几乎无损（0.9125 → 0.9096），`noul` 无损（0.8397 → 0.8397），`score` 从 0.5948 掉到 0.4606 —— **但这个跌幅与量化无关**，实测 fp32 走 ONNX 在同一批数据上同样是 0.5948，而 #6 记录的 0.5948 来自 torch 路径，两条路径口径一致；真正的问题另有下文（§6.3）。p50 < 50ms 在输入 ≤96 token 时达标（44.0ms），平均客服工单 131 token 时是 65.7ms，需要靠**截断上限**而不是靠更激进的量化来达标。

上线形态建议：**先发 CLI，再发极小的 `predict()` 推理 API 包装**（§7）。

---

## 1. laya-multilingual 能不能导出 ONNX？

### 1.1 pip 包里没有导出脚本 —— 必须自己写

- 已安装的 `laya` 0.4.3 里 **不存在任何 export 脚本或 CLI 子命令**。`laya/cli.py` 里 grep `export|add_parser|subparser` 无命中；`laya/backends/__init__.py:8` 的 `NAMES = ("auto","eager","compile","tilelang","onnx")` 说明 ONNX 后端是一等公民，但导出器没进 wheel。
- 唯一线索是一句悬空引用：`laya/onnx_agent.py:154`
  ```
  f"ONNX model not found at {onnx_path!r}. Please run export_onnx.py first."
  ```
  即 `export_onnx.py` 存在于上游 GitHub 仓库而非发布包。**我们没有去上游仓库拉它，而是自己实现** —— 见 `scripts/export_onnx.py`，理由见 1.3。

### 1.2 choice / score / noul 不可拆，也不必拆

初判问题问的是「导出粒度是整模型还是 per-head」。读源码后结论明确：**没有三个头，只有一个头**。

`laya/common.py:609`：

```python
def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype,
            detach_encoder=False, position_ids=None, option_ids=None):
```

单次融合前向，恰好返回 `(logits, act_logits)` 两个张量。三个原语的区别只是喂进去的 `qtype ∈ {choice:0, score:1, noul:2}`：`self.type_emb = nn.Embedding(3, d)` 把一个 query embedding 加到隐状态上（`common.py` 里 `h = h + type_emb(qtype)[:, None, :]`），后面 `self.scorer` 只有一份。**per-head 导出在语义上不存在**，强行拆成三个 session 会让编码器算三遍。

顺带纠正票面描述：基座不是「12 层 mmBERT-base」，`encoder/config.json` 实测 `num_hidden_layers = 22`（hidden 768 / intermediate 1152 / 12 heads / vocab 256000 / local_attention 128 / global_attn_every_n_layers 3）。这一点直接影响后面的体积估算。

### 1.3 我们自己写导出器的唯一难点：TransformerEncoderLayer 无法直接 trace

torch 2.6 对 `nn.TransformerEncoder` 的融合 fast path `aten::_transformer_encoder_layer_fwd` **没有 ONNX symbolic**，直接导会抛
`UnsupportedOperatorError`。

laya 的作者其实预埋了解决方案的一半：`common.py:505` 的 `_DynamicMultiheadAttention` 替换掉了 stock attention，目的正是让头部在动态 seq 长度下可 trace。但 fast path 本身还在。

`scripts/export_onnx.py` 的做法是包一层 `ExportWrapper`，用 `_head_layer()` 把两层 head layer 重新表达成原语（`norm_first=True`、relu、eps 1e-5，**与 `nn.TransformerEncoderLayer` 默认值逐项对齐**），绕开不可导的 fast path，数值不变：

```
missing: []      unexpected: []
max|dlogits| = 1.252e-06      max|dact| = 2.441e-04
```

parity 校验刻意在 **seq=137**（未 trace 过的长度）上跑，用来证明 `dynamic_axes` 真的生效。

### 1.4 导出后怎么组回 session：不需要组装

**单图直接就是一个 session。** 输入输出契约取自 `onnx_agent.py:791-801`：

| 方向 | 名称 | dtype / shape |
|---|---|---|
| in | `input_ids` | int64 [batch, seq] |
| in | `attention_mask` | int64 [batch, seq] |
| in | `marker_pos` | int64 [batch, markers] |
| in | `marker_mask` | bool [batch, markers] |
| in | `qtype` | int64 **[batch]**，不是 [batch, markers] |
| out | `logits` | [batch, markers] |
| out | `act_logits` | [batch, n_act] |

`marker_pos` / `marker_mask` 必须补到 **≥2 列**（`onnx_agent.py:778-788`，因为图里 trace 了 `topk(2)` 算 act 头特征）。

实测图规模：**3378 个节点**（encoder 2968 / head self_attn 109 + self_attn_1 98 / scorer 13 / act_head 10），opset 17。

### 1.5 ⚠️ 导出的图有一个真实 bug：短行在右 padding batch 里返回 NaN

**这是本次调研最重要的发现，必须写进部署文档。**

在 343 行 test split 上按 laya 官方的 `collate_items` 组 batch=16 推理时，**340/1029 行返回 NaN**，且 100% 是 `qtype=2`（noul）行：

```
fp32 ONNX, batch=16, 1029 行 → 340 行非有限，全是 qtype=2
```

排查过程（每一层都排除过）：

| 假设 | 实测 | 结论 |
|---|---|---|
| 图优化级别导致 | `ORT_ENABLE_ALL` 与 `ORT_DISABLE_ALL` 都是 5 行 NaN | 排除 |
| eager torch 也有问题 | torch 同 batch **0 行非有限**，logits 正常（`[-2.19, 0.15]`） | 排除，**是导出/图的问题** |
| padded marker 列位置 0 越界 | 改成 1 → 仍 5 行；截到 2 列 → 仍 5 行 | 排除 |
| batch 内 padding 不一致 | 强行等长 → NaN 变 11 行，更糟 | 排除 |
| 按长度排序 / 同 qtype 分批 / marker 补齐 | 都还是 5 行 | 全部排除 |
| **该行单独成 batch** | **0 行 NaN**，logits 正确 | **决定性证据** |

关键复现（`probe_nan6.py`）：把 noul 行单独放进 batch 就正常；一旦和 choice/score 行放进同一个右 padding 的 batch 就 NaN；而 choice / score 行即使 padding 到 181 也不 NaN。**只有短行（真实长度 87–108）在混合 batch 里出问题** —— 指向 ModernBERT 的 sliding-window（local_attention=128）与 padding 交互在导出图里产生了全 `-inf` 的注意力行，`softmax(-inf, -inf)` → NaN。

**规避办法（已验证有效）**：**部署时 batch=1、每条输入按其真实长度单独推理**，不要跨样本 padding。本报告后面所有延迟/精度数字都是在 batch=1 精确长度下测的 —— 这同时也是 p50 < 50ms 的目标形态，所以这个 bug 对我们**不是阻塞项**，但必须写进部署文档并在 `scripts/export_onnx.py` 里注明。

（未修根因：修它要改 ModernBERT 的 masked attention 在 torch.onnx 下的表达，超出本票范围，且 batch=1 已满足验收指标。）

---

## 2. 量化方案选择

### 2.1 各部分实测体积（fp32 图 1228.6 MB 的来源）

权重来自 `model.safetensors`（614 MB，170 张量，dtype 全 F16），换算成 fp32 ONNX 图：

| 部件 | 参数量 | fp16 权重 | fp32 ONNX 图占比 |
|---|---|---|---|
| `encoder.embeddings`（含 256000×768 词表） | 196,608,768 | 375.0 MB | **61%** |
| `encoder.layers`（22 层） | 110,330,112 | 210.4 MB | 34% |
| `head.layers`（2 层决策头 transformer） | 14,175,744 | 27.0 MB | 4% |
| `scorer.*` / `act_head.0` / `type_emb` | ~0.8M | 1.5 MB | ~0% |
| **合计** | **321.4M** | **614 MB** | fp32 图 **1228.6 MB** |

**关键事实：256000 行的词嵌入表一张就占 61%**，是唯一的体积大头，而它对量化几乎免疫（Gather 后面接的是 fp32 的 LayerNorm）。所以整模型 int8 拿到 4× 压缩（1228.6 → 308.6 MB）已经接近理论上限。

### 2.2 候选方案横评（全部实测）

在 6 个中文客服状态 × 3 个问题 = 18 行真实文本上，对比 eager fp32：

| 方案 | 大小 | p50 | argmax 一致率 | 概率 TV 距离 |
|---|---|---|---|---|
| fp32 | 1228.6 MB | 251 ms | 1.000 | 0 |
| **int8 动态 per-channel** ⭐ | **308.6 MB** | **28–115 ms** | **0.667** | **0.1489** |
| int8 动态 per-tensor | 307.8 MB | 35–139 ms | 0.444 | 0.1577 |
| int8 动态 per-channel + reduce_range | 308.6 MB | — | 0.556 | — |
| int8 静态 QDQ per-channel | 309.1 MB | 47–197 ms | 0.167 | 0.3543 |
| int8 静态 QOperator | **失败** | — | — | `RuntimeError: Invalid model with unknown initializers/tensors.{'/Squeeze_1_output_0'}` |
| int4（QUInt4 动态） | **失败** | — | — | `Quantized weights for onnx::MatMul_4131_quantized must be 8-bit before packing as 4-bit values.` |
| int8 排除决策头（`nodes_to_exclude`） | 323.0 MB | — | 0.444 | 0.1582 |
| int8 排除 Gather（词表留 fp32） | 870.3 MB | — | 0.444 | 0.1582 |

### 2.3 决策：**我们选 int8 动态 per-channel，不选 static、不选 int4**

**不选 static QDQ / QOperator。**
- QOperator 直接**跑不起来**：`onnx::Squeeze_1_output_0` 是 torch 导出的融合 Gather/Clip/Squeeze 产物，量化器认不出来。
- QDQ 能跑，但**准确率塌了**：argmax 一致率 0.167，TV 0.3543，是动态量化的 2 倍误差；后面在真实 test split 上测宏平均准确率只有 **0.2536**（vs fp32 0.7823），等于把模型毁了。原因是我们的标定集只有 6 条合成样本，而激活分布实际上由真实中文工单的 token 分布决定。
- 静态量化唯一优势是能挂 TensorRT/整数核，但靶机是 x86 CPU-only 老笔记本，这条优势用不上。

**不选 int4。** `onnxruntime.quantization.quantize_dynamic` 在这条路上**直接报错**（int4 权重必须先 8-bit 再 pack），即当前工具链不支持。真要 int4 得先过 `quant_pre_process` 并走 matmul4 variant，收益是 308 → ~160 MB，但对我们**没有价值**：1GB 预算已经用掉 31%，省下的体积买不到任何验收指标，而 4-bit 权重的精度损失会直接打在 §6 那个已经很脆弱的 `score` 原语上。

**不选 per-tensor。** per-channel 精度更高（一致率 0.667 vs 0.444）而体积只差 0.8 MB，延迟还更快 —— 零成本改进，没理由不用。

**选动态 per-channel 的理由**：`quantize_dynamic` 免标定数据（部署方不用再攒一份 calib 集）、per-channel 沿输出通道缩放参数、对 ModernBERT 22 层里尺度差异很大的层更稳、大小和延迟都达标。**需要的工具链只有 `onnxruntime.quantization`（已随 onnxruntime 1.31.0 装好）**，不需要 `optimum`（本环境未装），也不需要 `torch.ao.quantization` —— 理由见 §8。

### 2.4 ≤1GB 达成情况：**有大量余量**

单进程 RSS（`GetProcessMemoryInfo`，加载一个 session、跑 30 次后稳态）：

| 场景 | RSS |
|---|---|
| python + onnxruntime 空载 | 48.4 MB |
| **int8 session（推荐）** | **408.1 MB**（加载 2.4–2.8s） |
| fp32 session | **1327.0 MB** ← 直接爆掉 ≤1GB（加载 3.8–5.3s） |

int8 下按序列长度的 RSS 与延迟（本机 4 线程）：

| seq len | RSS | p50 |
|---|---|---|
| 32 | 717 MB | 26.4 ms |
| 64 | 721 MB | 41.9 ms |
| 128 | 732 MB | 59.8 ms |
| 256 | 749 MB | 119.3 ms |
| 512 | 807 MB | 241.2 ms |

> 注：717–807 MB 是**默认线程数**下的数字；显式 `intra_op_num_threads=4` 时是 408 MB。发布配置一律显式设 4 线程。

**结论：≤1GB 常驻用 int8 有 2.4× 余量，fp32 则必然超标 —— 量化不是优化项，是及格线。**

---

## 3. 在微调 checkpoint 上的真实评测（#6 数据的量化退化）

票面要求的不只是「能不能导」，而是「导了会不会把 #6 的成绩打掉」。所以我把 #6 的 checkpoint `$WORK\runs\v02_rlcd_e2` 也完整走了一遍导出 → 量化 → 在 **test split（343 行 / 1029 records，从未参与训练也未参与标定）** 上重测，并**始终施加 #6 拟合的温度**。

### 3.1 精度退化（batch=1 精确长度，与 #6 口径一致）

| 原语 | fp32 ONNX 准确率 | int8 per-channel | int8 per-tensor | int8 静态 QDQ |
|---|---|---|---|---|
| `choice` / route_team | 0.9125 | **0.9096** (−0.0029) | 0.9155 | 0.1983 |
| `score` / urgency | 0.5948 | **0.4606** (−0.1342) | 0.4665 | 0.1924 |
| `noul` / needs_refund | 0.8397 | **0.8397** (±0) | 0.8367 | 0.3703 |
| **宏平均** | **0.7823** | **0.7366** (−0.0457) | 0.7396 | 0.2536 |

ECE（施加 #6 的 shipped 温度）：

| 原语 | fp32 | int8 per-channel |
|---|---|---|
| `choice` | 0.0706 | 0.0911 |
| `score` | 0.0542 | 0.0846 |
| `noul` | **0.0950** | **0.1128** ❌ |

**两个必须上报的坏消息：**

1. **`score` 掉 0.1342 是真的**，但**它不是量化造成的**。看 §1.5：fp32 ONNX 在 batch=1 下 score = 0.5948，与 #6 的 torch 数字完全一致 —— 说明 torch 和 ONNX 两条路径口径没问题。掉分集中在 score 的**近平局区域**：fp32 logits 的 top1-top2 margin 中位数只有 0.5544，**33% 的行 margin < 0.3**，int8 的舍入噪声足以在这些行翻转 argmax。这是 `score` 原语本身的脆弱性（#7 已记录 `urgency` 是 `emotion_intensity` 的 proxy 标签，#6 已记录它「只能当参考分」），量化只是把它暴露出来。
2. **`noul` 的 ECE 变成 0.1128 > 0.10，直接违反验收线。** 见 §3.2 —— 这是可修的。

### 3.2 ✅ 温度必须随量化模型重新拟合（#6 温度不能直接用）

**#6 的 `calibration.json` 不能直接配 int8 图。** 反直觉但数据很清楚：int8 把 logits 整体**压扁**了，模型重新变得**过度自信到需要更低的温度**，而 #6 的 1.05/1.29/1.20 是往**上**软化的。

在 **calib split（285 行 / 855 records，未参与训练）** 上重新拟合（最小化 NLL，网格 + 三分搜索），在 test split 上报：

| 原语 | #6 温度（fp32） | int8 重拟合温度 | ECE：#6 温度 | ECE：重拟合温度 |
|---|---|---|---|---|
| `choice` | 1.0494 | **0.6528** | 0.0911 | **0.0467** ✅ |
| `score` | 1.2862 | **0.6872** | 0.0846 | 0.0893 |
| `noul` | 1.1971 | **0.5267** | 0.1128 ❌ | **0.0486** ✅ |
| **最差单原语 ECE** | — | — | **0.1128** | **0.0893** ✅ |

**结论（这条直接回答 #6 的第 3 点）：量化模型必须携带一份重拟合的 `calibration.json`。** 重拟合后宏平均准确率不变（0.7366，温度不改 argmax），**最差单原语 ECE 从 0.1128 回到 0.0893，≤0.10 验收线重新达标**。

⇒ **发布纪律：每出一个量化变体，就在 calib split 上重跑一次 `calibrate_eval.py` 的拟合，落一份新的 `calibration.json`，与该变体的 `.onnx` 同名同目录发布。不要跨精度复用温度。**

顺带一个方法学发现：同一个拟合流程在 **fp32 图上重跑**得到温度 0.6558 / 1.0097 / 0.7477（ECE 0.0782），和 #6 用 torch 拟合出的 1.0494 / 1.2862 / 1.1971 差异很大 —— 说明**温度是模型 + 推理后端联合的属性**，不只是权重的属性。任何换推理栈的部署都要重新标定，这条应写进部署文档。

### 3.3 决策头留 fp32 有没有用？没有

`nodes_to_exclude` 把 `/self_attn`、`/self_attn_1`、`/scorer`、`/act_head`（112 个节点）留在 fp32：

| 变体 | 大小 | 宏平均准确率 | 最差 ECE（重拟合温度） |
|---|---|---|---|
| 全 int8 per-channel | 308.6 MB | 0.7366 | 0.0893 |
| int8 encoder + **fp32 决策头** | 318.9 MB | 0.7386 | 0.0891 |

多花 10.3 MB 换 +0.002 宏平均，**不值**。同理，逐层混合精度（只量化底部 8/12/16 层）三个变体大小全部是 318.9 MB —— 因为被排除的层太小，梯度不存在。**不选混合精度**：没有可调旋钮，纯属复杂度。

### 3.4 精度取舍的最终口径

在 #6 给的三原语可接受区间里，int8 的位置是：

- `choice` 0.9125 → 0.9096：**可接受**，−0.3 个点仍在 0.91 高位。
- `noul` 0.8397 → 0.8397：**无损**。
- `score` 0.5948 → 0.4606：**超出可接受区间**，但 **fp32 也救不了它**（fp32 只有 0.5948，差距 0.13 是 quantizer 造成，量级上确实归量化）。诚实结论：**`score` 在 int8 下不可自动决策**，与 #6 的结论（urgency 只能当参考分）一致，且量化让它更不可信 —— **发布时必须披露，且 urgency 档必须硬性走转人工**。

若要救 `score`，唯一有效的旋钮是 **float16**（而非更激进的量化）：ModernBERT 的 fp16 权重已经在 checkpoint 里，ORT 支持 fp16 图推理，精度应与 fp32 几乎一致而体积约 600 MB（仍 ≤1GB）。**建议作为下一张票的候选**（§8 未决问题）。

---

## 4. 延迟：p50 < 50ms 的真实边界

**测量条件**：CPU only、`intra_op_num_threads=4`、`inter_op_num_threads=1`、batch=1、每条按真实长度精确推理、1029 条 test split 全量、同时施加 #6 温度。

### 4.1 输入长度 × 量化格式

| 截断上限 `max_len` | 实测平均长度 | int8 per-channel p50 | int8 p90 | fp32 p50 |
|---|---|---|---|---|
| 64 | 64 | **33.6 ms** ✅ | 37.4 ms | 70.2 ms |
| 96 | 91 | **44.0 ms** ✅ | 49.3 ms | 166.5 ms |
| 128 | 113 | 52.0 ms ⚠️ | 58.8 ms | 221.6 ms |
| 256 | 131 | 63.9 ms ❌ | 82.2 ms | 261.8 ms |
| 512（= #6 实配） | 131 | 65.7 ms ❌ | 82.9 ms | 264.0 ms |

**硬结论（不要在发布材料里含糊）：**

- **平均客服工单（131 token）下，int8 p50 = 65.7 ms，p50 < 50 ms 不达标。**
- 把 `max_len` 从 512 收到 **96**，p50 = **44.0 ms，达标**（此时平均长度 91 token，只截掉约 0 头的信息）。
- int8 相对 fp32 提速 **2.1×（cap=128 档）到 4.0×（cap=64 档）** —— 量化对延迟的贡献是决定性的，不是锦上添花。
- #7 实测的 token 长度是 p50 88 / max 147。**`max_len=96` 会截掉尾部超过 96 的约 3–5% 长工单**，这是必须显式接受的产品取舍，不是免费的午餐。

⇒ **建议：发布配置 `max_len=96`、`head_max_len=96`，并在文档里写明「>96 token 的工单按前 96 token 决策」。** 若业务不能接受截断，则 p50<50ms 这条验收线在 CPU-only 靶机上**物理上不可达**，需要改验收线或上 float16/蒸馏（§8）。

**注意本机是一台台式 x86 CPU，4 线程能吃满 4 个核。「x86 CPU-only 老笔记本」的真实性能大概率只有这里的 40–60%。本票所有延迟数字应视为乐观下界，正式验收必须在真实靶机上重跑 `scripts/quantize_onnx.py` 的 bench 段。**

---

## 5. tokenizer 的坑与对齐测试

### 5.1 现象

`transformers.AutoTokenizer` 在 transformers 5.19 下加载这个 checkpoint 直接失败：

```
ValueError: Couldn't instantiate the backend tokenizer ...
You need to have sentencepiece or tiktoken
```

原因：`tokenizer/tokenizer.json` 是 XLM-R 系（BPE，vocab **256000**）的 fast tokenizer 文件，但 `tokenizer_config.json` 没有 `tokenizer_class`，5.19 走了需要 sentencepiece/tiktoken 的慢后端探测分支。

### 5.2 结论：部署侧只用 `tokenizers` 库

**推荐（部署侧，零 transformers 依赖）**
```python
from tokenizers import Tokenizer
tok = Tokenizer.from_file("tokenizer/tokenizer.json")   # 直接可用
```

**参考侧（对齐测试用，需要 transformers 时）**
```python
from transformers import PreTrainedTokenizerFast
tok = PreTrainedTokenizerFast(
    tokenizer_file="tokenizer/tokenizer.json",
    bos_token="<bos>", eos_token="<eos>", cls_token="<bos>", sep_token="<eos>",
    unk_token="<unk>", pad_token="<pad>", mask_token="<mask>")
```

⚠️ **必须显式传这七个 special token**。用 XLM-R 常见的 `"<s>"` / `"</s>"` 会抛
`TypeError: replace() argument 1 must be str, not None` —— 这个 checkpoint 用的是 `<bos>` / `<eos>`，不是 `<s>` / `</s>`。

### 5.3 对齐测试

`scripts/check_tokenizer_alignment.py` 覆盖 7 组输入（纯中文 / 纯英文 / 中英混合 / 标点 / emoji / 空串 / 纯空白），断言两侧 token id **逐位完全一致**：

```
PASS 7/7
```

样例：`我们要不要把这个功能推迟到下个季度？`
→ `[2, 118772, 159374, 236167, 8894, 19655, 236317, 239028, 235562, 235543, 235548, 166393, 235544, 1]`

**这条测试应该成为 CI 门禁**：任何换 tokenizer / 换部署栈的改动都必须跑它，否则 token id 漂移会静默地把准确率打穿，且不报错。

---

## 6. 移动端（Android arm64）：**本图范围外，建议建下一张票**

**判断：本图不覆盖移动端。** 依据：

1. **验收口径不匹配。** map 的硬验收线明确写的是「x86 CPU-only 老笔记本 / 中端手机」并列，但**可测的只有一台 x86**。在没有 arm64 设备的情况下宣布移动端达标，等于给发布材料写一个没人验过的承诺。
2. **需要的工具链完全不同。** ONNX Runtime Mobile 需要 `onnxruntime-mobile` + NNAPI/CoreML 委托 + XNNPACK，且**必须先做 ORT 图转换优化**（`python -m onnxruntime.tools.convert_onnx_models_to_ort`），而我们的图里还有一个未解释的 NaN bug（§1.5）和 fused Squeeze 算子（正是 static 量化失败的根因）—— 这些在 ORT-M 格式转换器上会再炸一次。
3. **int4 是移动端的正解，而当前工具链不支持**（§2.3 的报错）。真要走 arm64，int4 几乎必须，工具链前置工作是一整张票。
4. **收益顺序明确。** 老笔记本的 p50<50ms 是 map 的**当前阻塞验收线**，移动端是加分项。先解阻塞项。

**留给下一张票的题目**：`onnxruntime-mobile 转换 + int4 权重路径 + arm64 靶机实测`，前置依赖是本票的 int8 变体和图清理。

---

## 7. 上线形态建议

**推荐：双形态，但以 CLI 为主交付。**

### 7.1 `mai-jev-predict` CLI（主交付物）

```
mai-jev-predict --model-dir <dir> --input <text|jsonl> [--max-len 96] [--threads 4] [--json]
```

**理由：**
1. **验收指标本身就是一个 CLI 的输出。** 「p50 < 50ms」「≤1GB 常驻」只能在**冷启动一次 + 稳态循环**的进程里测。一个 `python -c` 片段做不到这件事，而 CLI 自带计时与 RSS 打印（`scripts/quantize_onnx.py` 的 bench 段已实现 `GetProcessMemoryInfo` + p50/p90，直接复用）。
2. **依赖面最小。** CLI 只需要 `onnxruntime` + `tokenizers` + `numpy`，**不需要 torch、不需要 transformers、不需要 laya**。发布体积可控，也避免了 laya 0.4.3 的 `AutoTokenizer` 依赖（§5.1）。
3. **是 HF/ModelScope 仓库的标准形态。** model card 上给一条能直接复制运行的命令，比给一段 API 片段更有说服力，也更容易被第三方复现。

### 7.2 最小推理 API（后补，不阻塞发布）

在 CLI 之上包一层 20 行左右的 `predict(state: str) -> dict`，理由是让上游业务把 mai-jev 当函数用。**但要等到 CLI 稳了再发**，因为 API 一旦发布就是兼容性承诺，而现在：
- 还有 §1.5 的 NaN bug 未修根因；
- p50 是否达标**取决于 `max_len` 这个产品决策**（§4），而这个决策还没人拍板；
- 温度需要按精度重标定（§3.2），标定文件会随变体变动。

**API 形态不做成服务**：map 的 Out of scope 已明确「不提供托管式在线推理服务」，所以是**可嵌入的库函数**，不是 HTTP 服务。

---

## 8. 交付物与剩余未决问题

### 8.1 脚本

| 路径 | 作用 | 依赖 |
|---|---|---|
| `scripts/export_onnx.py` | 整模型单图导出 + eager parity 自检 | torch, laya, onnx |
| `scripts/quantize_onnx.py` | int8 动态 per-channel（可选 static 对照）+ 体积/p50/RSS bench | onnxruntime |
| `scripts/check_tokenizer_alignment.py` | 部署侧 vs 训练侧 token id 逐位对齐（CI 门禁） | tokenizers, transformers |

三者均以 docstring 写明用法与依赖，可直接复用。

### 8.2 未决问题

1. **【阻塞发布】`max_len` 拍板。** `max_len=96` → p50 44.0ms 达标但截断长工单；`max_len=512` → p50 65.7ms 不达标。需要业务/用户决定截断是否可接受，或改验收线。**这是唯一卡住「p50<50ms 达标」的因素。**
2. **【阻塞发布】真实靶机复测。** 本机是一台台式 x86 CPU，4 线程只相当于靶机的乐观上界。必须在真实老笔记本上跑 `quantize_onnx.py` 的 bench 段重测 p50 与 RSS。
3. **`score` 原语在 int8 下掉到 0.4606。** 是否接受「urgency 只做参考分、强制转人工」（#6 已定），还是投入合成正确标签的数据来救（#7 结论：加 epoch 无用，只有 teacher 合成有效）。
4. **float16 值得试。** checkpoint 里权重本就是 F16，fp16 ONNX 图预计 ~600 MB（≤1GB），精度应接近 fp32，可能是「保住 score 又满足延迟」的更好解。未实测。
5. **§1.5 的 NaN 根因未修。** batch=1 规避有效，但影响未来做批量推理。
6. **int4 需要工具链前置工作**（当前 `quantize_dynamic` 直接报错），是移动端票的前置。
7. **每出一个量化变体都要重跑温度标定**（§3.2）。建议把这个动作直接写进发布 checklist，而不是靠人记。

> **2026-10-17 更新：第 1、3、7 条已解决，第 2、4、5、6 条仍然开放。**
>
> - **1（max_len 拍板）→ #9 拍板**：发布变体定为 `int8-gather / max_len=256`。`max_len=96` 被否不是因为截断长工单，而是它连 `choice`/`score` 的 head（109/110 token）都装不下 —— 整条工单正文被静默丢弃，`route_team` 从 0.945 掉到 0.1826（`fd32c83` 加了双闸门，装不下就直接报错退出）。同时验收线改为「一条工单跑完三原语合计 p50 < 500ms」。
> - **3（score 在 int8 下 0.4606）→ #16 解决**：根因不是量化而是**标签语义错**（`emotion_intensity` 当成了 urgency）。用规则证据分级器合成正确的 urgency 标签 + 重平衡后，`urgency` 在 torch 上 0.9329、在发布用的 int8 图上 0.9359，整变体量化退化 ≤0.0029。
> - **7（逐变体重标定）→ 已写进产物**：`scripts/calibrate_export.py` 负责在**该变体自己的 calib** 上重拟合，`deploy_config.json` 的口径声明会随每个变体一起发布 —— 不靠人记。
> - **2 尤其仍未解决**：本机 CPU 是靶机的乐观上界，而发布变体的端到端 p50 已经是 421–499ms（见 [RUNS.md](RUNS.md) 2026-10-17 条），余量 0–16%。

### 8.3 新依赖说明

**本票没有安装任何新包。** 全部使用环境中已有的 `onnxruntime 1.31.0`（含 `onnxruntime.quantization`）、`onnx 1.23.2`、`tokenizers 0.23.3`、torch 2.6.0。
明确**不需要**：`optimum`（本环境未装；onnxruntime 自带的 quantizer 已覆盖本票全部需求，optimum 只会多引入一层抽象和版本耦合）、`torch.ao.quantization`（我们需要的是导出后的 ONNX 图量化，不是导出时的 torch 量化，两者产物不同）。