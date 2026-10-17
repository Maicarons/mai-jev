# CONTEXT — mai-jev 领域术语表

mai-jev 是一个**面向低端设备的中文优先 System One 决策模型**：以 `convaiinnovations/laya-multilingual`（322M，mmBERT-base 编码器 + 决策头）为基座，用 RLCD 微调，经温度标定使 ECE ≤ 0.10，量化后部署在 x86 CPU-only 靶机上。本文件是唯一的术语来源；写代码、测试名、issue 标题时一律用这里的词，不要换成同义词。

## State（输入状态）

模型一次决策所看到的全部输入文本。对客服工单分诊来说，State 就是一条工单的正文（可含标题、渠道、历史摘要）。不是"prompt"、不是"input string"、不是"context window"。

约束：Laya 上下文默认 512/1024 token，留给 state 约 768 token；超过部分按 `tools/` 里的截断策略处理，不要静默丢掉尾部。

## Question（具名问题）

一次推理携带的、带名字的具名问题，例如 `route_team` / `urgency` / `needs_refund_or_escalation`。每个 Question 有 `instructions` 和 `criteria`：

- `choice` 的 `criteria`：dict（label → 描述）或 label 列表。**不能写成 `options`**（laya 会直接报 `ValueError: question 'x': a choice question takes 'criteria' as a dict of label -> description, or a list of labels`）。
- `score` 的 `criteria`：**等级描述列表，index 0 = 最低分**（不是 `min`/`max`）。
- `noul` 的 `criteria` 可选，且只允许 `{'true','false'}` 两个键。

不是"prompt"、不是"task"、不是"label"。

## 原语（primitives）：choice / score / noul

三种问题类型，对应 laya 的 `QTYPES = {'choice': 0, 'score': 1, 'noul': 2}`：

| 原语 | 输出 | 本项目的旗舰域含义 |
| --- | --- | --- |
| `choice` | 离散类别 + 概率分布 | 路由到哪个团队（售后 / 退款处理 / 技术支持 / 账务 / 投诉升级） |
| `score` | 1..5 有序等级 + 分布 | 工单紧急程度 |
| `noul` | [0,1] 概率 | 是否要求退款、赔付或升级处理 |

**`noul` 的语义口径（标注器 / 评测 / 发布文案一律以这条为准）**：

- `true` = 客户提出了退款 / 赔付 / 升级处理的**诉求**，三种形态都算：
  ① **明说**：`我要退款`、`麻烦快点理賠`、`把钱退回我的卡里`；
  ② **升级要求或威胁**：`我要找你们主管`、`再拖我就去消保协会投诉`、`发到网上曝光`；
  ③ **问资格或意向**（**宽松口径**，2026-10-17 用户拍板）：`我能不能退款？`、`还能退货吗`、
  `定金能退吗`、`有没有什么补偿方案`。
- `false` = 纯问流程 / 规则（`退貨流程怎么走`、`退運費谁负担`、`退款政策几天`）、
  只有情绪或催促（`急死我了，快帮我查清楚`）、或者客户明说只是随便问问
  （`东西没问题，我就是随便问问` —— 那是**没有诉求**，不是问资格）。
- 标签由 `scripts/label_noul.py` 生成（规则分级器 + 37 条人工反例集**硬门禁**，
  [#17](https://github.com/Maicarons/mai-jev/issues/17)），**不再**用上游
  `needs_human` / `gold.escalate` —— 那两个问的是「是否需要人工介入」而非客户诉求。
- 重标后正例只剩 ~20%（多数类基线 0.796），所以 `noul` 的**裸准确率不可作判据**，
  一律报 balanced accuracy；且新旧 `noul` 数字**不可跨版本比较**（标签定义变了）。

> 已知张力（留给后续拍板）：`questions.py` 里模型可见的 `instructions` 仍是
> 「客户是否**明确要求**退款、赔付或升级处理？」，而口径第 ③ 条把「问资格」也算 true。
> v0.4 刻意**只改标签、不改 prompt**（保持单变量，且不动已发布的 v0.3 产物口径）；
> 是否把宽松口径写进 `instructions` 是另一个决定，见 #17 的后续项。

不要说"分类任务 / 回归任务 / 二分类"来指代它们——那是别的模型的词。

## Confidence（置信度）

模型对自己这一票的确信度，来自概率分布经**温度缩放**后的统计量（`answer_confidence(p, k)` / `confidence_from_probs(p, k)`）。不是"softmax 最大值"本身，也不是"概率"。

## ECE（Expected Calibration Error）

期望校准误差，按置信度分箱统计"置信度 vs 实际命中率"的差距。本项目的验收线是 **ECE ≤ 0.10（温度标定后）**。注意：`convaiinnovations/laya-multilingual` 出厂 ECE 约 0.466，且自带温度超界会被 laya 钳到 0.5（告警：`laya: this checkpoint ships invalid temperatures or values outside [0.5, 5]; ... Treat confidence from the affected entries as uncalibrated.`），所以原版权重输出的 Confidence 一律视为未标定。

## 温度标定（temperature fitting）

在**留出集 calib**（不参与训练）上为每个 Question 拟合一个温度 T，把 logits 除以 T 后再 softmax。本项目用 `laya.fit_one_temperature` / `fit_temperature_map`，不要在训练集上拟合。目标：ECE 从 0.466 量级降到 ≤0.10。

## RLCD（Reinforcement Learning for Calibrated Decisions）

Laya 的训练范式：用**严格适当评分规则**（对数 / 球面 / 排序概率评分）组成奖励，配 GRPO 风格的组基线策略梯度更新，**不用交叉熵**。之所以不用 CE：CE 只逼模型把正确类推高，不约束置信度校准；RLCD 直接优化"概率分布 vs 真值的评分规则差距"，校准与准确率一起学。本项目对应 `scripts/train_rlcd.py`。

不要写成"RLHF"（那是偏好对齐）、不要写成"DPO"。

## 靶机（target machine）

x86 CPU-only 的老笔记本 / 中端手机这一档：**常驻内存 ≤1GB、4 线程、一条工单跑完三原语合计 p50 < 500ms**（发布变体 int8-gather、`max_len=256`）。这是部署验收口径，不是训练机（训练机是一台消费级台式 x86 CPU + 单张 8GB 显存的消费级 GPU）。说"低端设备"时，指的就是靶机。

口径沿革：原口径是「单问题 p50 < 50ms」（2026-10-13 之前），#9 实测作废——延迟 ≈ 序列长度 × ~1ms/token，而序列被 question head 主导（head 109/110 token vs 工单正文 p50 47.5），322M + 4 线程下物理上不可达。现行口径的**余量很小**：发布变体在训练机上端到端实测 p50 420.9–498.7ms、p95 862–976ms（n=150，两次复测，见 `docs/research/RUNS.md` 2026-10-17 条）。「单问题 p50 < 50ms」保留给未来的 ≤500MB 蒸馏档位；真实靶机复测之前，不得对外声称延迟稳定达标。

**2026-10-17 追加（重要）**：这台训练机的延迟会**跨 session 大幅波动**——同一个二进制、同一条命令、同一份输入，7 次 session 的端到端 p50 实测落在 **317.60ms–806.44ms**（2.54 倍；A/B 交替对照里两个包各自的两次也跨满整个区间，所以这不是模型差异）。内存一直充裕（空闲 16–20GB / 总 30.4GB），常住内存在 7 次 session 里都落在 797–815MB 这条窄带上，CPU 频率计数器也没给出单调解释，机制未查清。因此「p50 < 500ms」这条线**在快的 session 里达标、在慢的 session 里不达标**：对外引用必须同时给「训练机」与「哪一次 session / 跨 session 区间」两个限定，不能写成稳定达标。证据见 `docs/research/v04-latency.md` 与 `docs/research/RUNS.md` 2026-10-17 条。

## 决策门（Act / Escalate）

按代价矩阵自动执行或升级：判对 +1、判错 −3、升级 −0.5 → 只有正确率 > 62.5% 才值得自动执行。落到工程上就是按 Confidence 分档：≥0.9 自动执行 / 0.6–0.9 复核 / <0.6 转人工。

## Act / Escalate 头

Laya 决策头里基于 `[CLS]` 隐状态与分布统计量的二级输出，用来判断"这一票该不该自动执行"。是否为本项目训练，是地图里的一条 fog（当前倾向：先只做温度阈值分流，不训这个头）。

## 刻意不用的同义词

- prompt / instruction tuning（用 Question）
- input / context（用 State）
- classification / regression（用 choice / score）
- accuracy-only confidence（Confidence 必须经温度标定）
- RLHF / DPO（用 RLCD）
- "低端机"（用靶机）
