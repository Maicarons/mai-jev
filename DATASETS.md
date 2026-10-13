# DATASETS

mai-jev 训练/评测数据集的来源、许可、标签映射与已知偏差。
本文档由 issue [#7](https://github.com/Maicarons/mai-jev/issues/7) 产出；
原始审计见 [`docs/research/datasets-and-licenses.md`](docs/research/datasets-and-licenses.md)。

> **一句话诚实结论**：三个开源数据源里，**没有任何一个**同时提供语义正确的
> `choice` + `score` + `noul` 三元标签。因此 v0.2 训练集里 `route_team` 是
> 真实标签，`urgency` 与 `needs_refund_or_escalation` 是**有据可查的 proxy**。
> 下面逐项标注了每个 proxy 的来源与偏差，评测集则是完整人工 gold。

## 1. 数据源

| 简称 | 上游 | 许可 | 进入哪部分 | 行数 |
|---|---|---|---|---|
| `zh-tw` | [`albertkingdom/custom-service-ticket-zh-tw`](https://huggingface.co/datasets/albertkingdom/custom-service-ticket-zh-tw) | CC BY 4.0 | train / calib / test | 3,148 |
| `mri` | [`MoriyaResearchInstitute/MRI-decision-data-100k`](https://huggingface.co/datasets/MoriyaResearchInstitute/MRI-decision-data-100k) | MIT | train / calib / test | 27 |
| `zh-bench` | [`CodyQin/zh-decision-bench`](https://huggingface.co/datasets/CodyQin/zh-decision-bench) | CC BY 4.0 | **仅 test** | 34 |

原始数据落在 `$WORK\data\raw\`（不进 git，见 `data/README.md`）。

### 许可义务（发布时必须履行）

- **CC BY 4.0**（zh-tw、zh-bench）：允许商用与改作，但**必须署名**。
  发布时的 model card 与本仓库 README 都要包含第 4 节的署名声明。
- **MIT**（mri）：只需保留版权与许可声明，随仓库分发即可。
- 本项目自身的代码与文档：**Apache-2.0**（继承自基座，许可证结论见
  issue [#11](https://github.com/Maicarons/mai-jev/issues/11) 的研究结论）。
- 数据本身**不随模型发布**，只发布转换后的训练脚本与可复现的构建流程，
  上游数据请各自按其许可获取。

## 2. 标签映射表

我们的三个原语定义在 [`mai_jev/questions.py`](mai_jev/questions.py)。
下表是每个源如何落到这三个标签上，`proxy` 标记为真的即为间接标签。

### 2.1 zh-tw（主力，3,148 行 = 98%）

`ticket` 字段含 `ticket_text` / `category` / `primary_intent` /
`emotion_intensity` / `needs_human`；另有 `label_status`（confirmed 2,770 /
boundary 378）与三个人工校验比率字段。

| 原语 | 上游字段 | 映射 | 性质 |
|---|---|---|---|
| `route_team` | `category` | 6 类 → 5 团队（见下） | **direct**，总量映射 |
| `urgency` | `emotion_intensity` | 1..5 同序直传 | **proxy** |
| `needs_refund_or_escalation` | `needs_human` | 布尔直传 | **proxy** |

类别 → 团队映射（`ZHTW_CATEGORY_TO_TEAM`）：

| zh-tw category | 行数 | → `route_team` | 依据 |
|---|---|---|---|
| 退換貨申請 | 500 | **退款处理** | QUESTIONS 定义「退款处理：要求退货、退款、赔付、价格保护」；换货/维修语境由 `needs_refund_or_escalation` 区分 |
| 帳務爭議 | 538 | 账务 | 「账务：发票、对账、付款、套餐变更」 |
| 帳號/系統問題 | 500 | 技术支持 | 「技术支持：产品故障排查：无法联网、报错、兼容性」 |
| 客訴投訴 | 610 | 投诉升级 | 「投诉升级：明确要求投诉、升级、赔偿或威胁曝光/监管」 |
| 商品諮詢 | 500 | 售后 | 「售后：一般售后咨询：保修、维修、换货、使用问题」 |
| 物流查詢 | 500 | 售后 | 物流属订单履约，不属账务/技术支持 |

繁体台湾国语在脚本内用词表（`TW2SP`）转简体大陆词汇，不引入 OpenCC 依赖。
**词表是不完整的**：未覆盖的繁体字会残留，这会拉低中文准确率，是已知偏差之一。

### 2.2 mri（补充，27 行）

只取 6 个「中文客服工单路由」family，共 300 行，去重后剩 27 行：

| family | 原始行数 | 说明 |
|---|---|---|
| `ticket-routing-intent-v1` | 50 | 单主诉 → 团队 |
| `ticket-routing-language-mix-v1` | 50 | 中英混排主诉 → 团队（**唯一的英文/混排样本来源**） |
| `ticket-routing-negative-disambiguation-v1` | 50 | 否定范围路由（「不是登录失败；是重复扣款」） |
| `ticket-routing-compound-priority-v1` | 50 | 多请求按用户显式书写的先后顺序 |
| `ticket-routing-severity-priority-v1` | 50 | 安全标记优先于主诉 |
| `insufficient-choice-unknown-v1` | 50 | 信息不足 → `unknown` |

| 原语 | 上游字段 | 映射 | 性质 |
|---|---|---|---|
| `route_team` | `targets.route`（英文键） | `unknown→售后`, `account→技术支持`, `billing→账务`, `technical→技术支持`, `fulfillment→售后`, `security→技术支持` | **direct** |
| `urgency` | 无 | 常量 `"3"` | **proxy（占位）** |
| `needs_refund_or_escalation` | 无 | 常量 `"false"` | **proxy（占位）** |

**为什么只有 27 行**：MRI 是模板化数据，300 行抹掉工单号后只有 27 条不同文本。
实测这些模板**互不冲突**（同一文本的 27 个 route 标签无矛盾），所以全部保留 ——
每条模板带不同的路由目标，是关于「否定 / 优先级 / 信息不足」的有效监督，
但**对整体分布的影响不到 1%**。

**排除的 family 及原因**：

- `threshold-*`（200 行）：state 是 `INC-SLA-017：不可用 120 分钟` 这类数值指标，
  不是工单正文；score 语义是「数值区间等级」，不是紧急度。
- `dialogue-*`（4,000 行）：对话轮次 + 领域话题（客服/差旅/教育），不是工单。
- `language-*`（11,000 行）：语言学任务（语义、语用、翻译、语域）。
- `world-*` / `logic-*`（84,600 行）：通用世界知识与逻辑推理。
- `software-*`（~10k 行）：state 是软件缺陷（`work_item` + 缺陷报告），
  criteria 是研发团队（frontend/backend/qa/…），**不是我们的 5 个客服团队**。
- 其余 `insufficient-*`：只有 meta-noul（信息是否明确给出），无路由语义。

### 2.3 zh-bench（仅评测，34 行）

`business_scenarios.jsonl` 里 `domain == "ecommerce_cs"` 的 34 行，
`state` 是**大陆简体自然客服语句**（如「刚才支付的时候钱扣了两笔！订单只显示一笔，
多扣的钱去哪了」），`gold` 是人工审定的三标签。

| 原语 | 上游字段 | 映射 | 性质 |
|---|---|---|---|
| `route_team` | `gold.route`（4 团队） | `物流配送→售后`, `退换售后→售后`, `支付问题→账务`, `账户安全→技术支持` | **direct** |
| `urgency` | `gold.urgency`（3 级） | `不紧急→1`, `一般→3`, `紧急→5` | **direct** |
| `needs_refund_or_escalation` | `gold.escalate` | 布尔直传 | **direct** |

**这 34 行绝不进 train/calib**，是唯一带完整人工 gold 的 held-out 集，
供 issue [#6](https://github.com/Maicarons/mai-jev/issues/6) 判「中文准确率超过原版」用。
分布：route `退换售后 13 / 物流配送 10 / 支付问题 8 / 账户安全 3`；
urgency `一般 21 / 紧急 9 / 不紧急 4`；escalate `True 16 / False 18`。

同文件里 `content_moderation` 的 21 行只带 `scam`/`escalate`，无法映射，丢弃。

## 3. 已知偏差（读这一节再相信上面的分布）

1. **`noul` 语义偏移**。zh-tw 的 `needs_human` 是「是否需要人工介入」，
   zh-bench 的 `escalate` 是「是否需要人工介入」，
   而我们的问句是「客户是否**明确要求**退款、赔付或升级处理」。
   前者包含「信息不足需补充」等非退款/升级场景。后果：
   训练集里 `noul` 正例占 **65%**（2,089/3,209），而评测集只有 **47%**（16/34），
   模型会有向 true 偏的先验。
2. **`urgency` 语义偏移**。`emotion_intensity` 测的是**情绪强度**（1..5），
   我们的 `urgency` 是**业务紧急度**。两者 1..5 同序但不同义：
   一个愤怒但无时限要求的工单，`emotion_intensity=5` 而 `urgency` 可能只有 2。
3. **`route_team` 覆盖不均**。「退款处理」500 行全部来自 zh-tw 的单一类别
   `退換貨申請`，没有独立的第二来源交叉验证。
4. **繁体 → 简体的词表不完整**（见 2.1），残留繁体字是中文语料的噪声源。
5. **`售后` 占 32%**（1,033/3,209），因为 zh-tw 的 `商品諮詢` + `物流查詢`
   共 1,000 行都映射到它，而这两个类别其实更像「一般咨询」而非「售后」。

**这些偏差是 v0.2 的已知上限**，不是 bug。要消解它们需要教师模型合成
（issue [#4](https://github.com/Maicarons/mai-jev/issues/4) 报告 6.3 节的方案），
或改标签定义 —— 两者都超出本票范围，已记录在 map #1 的迷雾区。

## 4. 署名声明（发布 model card / README 时照抄）

```
## 数据来源与许可

mai-jev 的训练与评测数据来自以下公开数据集，经转换后用于中文客服工单分诊：

- albertkingdom/custom-service-ticket-zh-tw — CC BY 4.0
  https://huggingface.co/datasets/albertkingdom/custom-service-ticket-zh-tw
- MoriyaResearchInstitute/MRI-decision-data-100k — MIT
  https://huggingface.co/datasets/MoriyaResearchInstitute/MRI-decision-data-100k
- CodyQin/zh-decision-bench — CC BY 4.0（仅评测）
  https://huggingface.co/datasets/CodyQin/zh-decision-bench

数据本身不随本模型分发。转换脚本见本仓库 scripts/build_dataset.py。
```

## 5. 构建与统计

```powershell
$env:PYTHONIOENCODING = 'utf-8'
python scripts\build_dataset.py `
    --raw $WORK\data\raw --out $WORK\data\adapted --write
```

固定随机种子 `SEED = 20261013`，同一 id 永远落在同一 split，重跑结果一致。

### 5.1 规模（v0.2 实测）

| split | 行数 | 每原语 item 数 | 用途 |
|---|---|---|---|
| train | 2,581 | 2,581 | RLCD 微调（≤3k，满足「先 1 小时跑通管线」的约束） |
| calib | 285 | 285 | 温度标定专用，**绝不进训练** |
| test | 343 | 343 | held-out 评测 |
| 合计 | 3,209 | — | — |

calib 定为 10% 的原因：`docs/research/RUNS.md` 记录 smoke 轮 calib 只有 72 条时，
laya 把三个温度全钳到 0.5（标定失败）；285 条 / 原语足以拟合温度。
注意 laya 自己做 calibration holdout（`calib_frac=0.1, calib_max=400`），
若要真正用上我们这份 285 条的 calib，需要在 `train_rlcd.py` 里显式传入 ——
留给 issue [#12](https://github.com/Maicarons/mai-jev/issues/12) 处理。

### 5.2 三原语分布

```
route_team                  urgency                    needs_refund_or_escalation
售后      1,033 (32.2%)     1    683 (21.3%)            false  1,120 (34.9%)
退款处理    500 (15.6%)     2    622 (19.4%)            true   2,089 (65.1%)
技术支持    515 (16.1%)     3    623 (19.4%)
账务       551 (17.2%)     4    602 (18.8%)
投诉升级    610 (19.0%)     5    679 (21.2%)
```

`urgency` 近乎均匀，是因为上游 `emotion_intensity` 本身就是均匀采样的。

### 5.3 选项基数

我们的三个问题**选项基数固定**，不随样本变化：

| 原语 | 类型 | 基数 | 标签集 |
|---|---|---|---|
| `route_team` | choice | 5 | 售后 / 退款处理 / 技术支持 / 账务 / 投诉升级 |
| `urgency` | score | 5 级 | 等级 index `0..4`（对外报 1..5） |
| `needs_refund_or_escalation` | noul | 2 | `false` / `true` |

上游数据的原始基数（映射前的信息量）：zh-tw 类别 6、情绪 5 级、needs_human 2；
zh-bench 4 / 3 / 2；MRI 路由 5 或 6。映射后统一收敛到 5 / 5 / 2。

### 5.4 token 长度

用基座 `laya-multilingual` 的真实 tokenizer（`tokenizers.Tokenizer.from_file`
读 `tokenizer/tokenizer.json`，vocab 256,000）实测 `state + 三问指令`：

| 分位 | token 数 |
|---|---|
| p50 | 88 |
| p90 | 104 |
| p99 | 119 |
| max | **147** |
| 超 768 预算的样本 | **0** |

分源：zh-tw p50=49（max 108）、zh-bench p50=19（max 30）、MRI p50=25（max 37），
加指令后为上表。

**截断策略**：上游最长样本 147 token，距 768 预算有 5 倍余量，
因此 **v0.2 不启用截断**。若将来引入长会话历史，`build_dataset.py` 的
`STATE_TOKEN_BUDGET = 768` 是唯一的守门常量，超限时按
「保留首尾、截中间」处理（保留诉求开头与工单号结尾）。

### 5.5 去重

- 精确去重：`sha1(state)`
- 近似去重：字符 3-gram Jaccard ≥ `NEAR_DUP_THRESHOLD = 0.85`
- MRI 侧在判重前先抹掉工单号（`normalize_for_dedup`，正则 `[A-Z]{2,}-[A-Z0-9]+-\d+`），
  否则模板化文本会被近似去重错误折叠。实测 300 行 → 27 行，
  且这 27 条模板的路由标签互不冲突。

zh-tw 3,148 行**零重复**，说明上游数据集本身已去重。

## 6. 复现入口

| 阶段 | 脚本 |
|---|---|
| 原始 → laya 训练行 + 统计报告 | [`scripts/build_dataset.py`](scripts/build_dataset.py) |
| 训练行 → laya-train 调用 | [`scripts/train_rlcd.py`](scripts/train_rlcd.py) |
| 规则基线（衡量 proxy 标签的噪声底） | [`scripts/baseline_rules.py`](scripts/baseline_rules.py) |
| 三个原语的唯一定义 | [`mai_jev/questions.py`](mai_jev/questions.py) |