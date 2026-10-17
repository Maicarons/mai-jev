# 中文客服工单分派数据集审计（mai-jev 训练数据调研）

调研日期：2026-10-12
目标：为 mai-jev（`convaiinnovations/laya-multilingual` 的中文 System One 微调）训练三个原语：
`choice`（5 团队：售后 / 退款处理 / 技术支持 / 账务 / 投诉升级）、`score`（紧急度 1–5 有序）、`noul`（客户是否要求退款或升级）。
预算：3–5 万条 × 4 epochs，单次实验 ≤4h；CPU-only；数据落在 `$WORK`（仓库外的磁盘）。
许可约束：必须可再分发（Apache-2.0 模型发布到 HF + ModelScope），**排除 NC / 仅研究 / 非商用**。
**不使用 Jev API 打标**（ToS 风险 + 大陆不可用）；可用本地/开源 teacher 模型（Qwen / GLM / InternLM）合成。

访问方式说明：`huggingface.co/api/*` 对 web_fetch/Invoke-WebRequest 返回 406，datasets-server 不可达；
本报告所有 HF 元数据通过 `hf-mirror.com` 取得（等价镜像，未登录，无 gated 例外）。

---

## 0. 结论摘要（TL;DR）

- **真正可用的中文工单级数据极少。**全网没有现成的「中文客服工单 → 5 团队分派 + 1–5 紧急度」开源数据集。
- 唯一同时覆盖**三个原语**且**格式就是 laya 训练格式**的中文语料是 `MoriyaResearchInstitute/MRI-decision-data-100k`（MIT，**100,000 行 / 104 文件 / 99.1 MiB 全量实测**）。
- 唯一提供**自然语言工单文本 + 1–5 情绪强度**的中文工单集是 `albertkingdom/custom-service-ticket-zh-tw`（CC BY 4.0，3,148 行，台湾繁体，合成）。
- 中文工单分派的**黄金评测集**是 CodyQin/zh-decision-bench（CC BY 4.0，`business_scenarios` 55 条，其中 `ecommerce_cs` 34 条带 route/urgency/escalate 三问齐备；另有 `voice_routing` 323 条）——但它是 **eval 集，不能当训练数据**。
- **建议以「少量真数据定格式 + 开源 teacher 大规模合成」为主**，见第 6、7 节。teacher 选 `Qwen/Qwen3-1.7B`（Apache-2.0，已验证）或 `Qwen/Qwen2.5-0.5B-Instruct`（Apache-2.0）。

---

## 1. 数据集逐一核实结果

### 1.1 MoriyaResearchInstitute/MRI-decision-data-100k ⭐ 核心候选

- **ID / URL**：https://huggingface.co/datasets/MoriyaResearchInstitute/MRI-decision-data-100k （镜像 https://hf-mirror.com/datasets/MoriyaResearchInstitute/MRI-decision-data-100k ）
- **License**：`mit`（HF API metadata 字段，已核实）
- **Gated / Auth**：gated=False，无需登录，直接 resolve 下载（已实测 104 个 jsonl 文件全部可下）
- **Language**：`zh`（每行 `_meta.language: "zh"`，中文占比 100%）
- **Size**：HF 标注 `100K<n<1M`。README 声称 1k 中文决策 + 20k 软件开发 + 50k 世界逻辑 + 29k 人类语言对话。
  **实测**（2026-10-13，104 个文件全部下载并逐行计）：`batch-001..005.jsonl` = 5×200 = 1,000 行；`language-dialogue-batch-001..029.jsonl` = 29×1,000 = 29,000 行；`software-batch-001..020.jsonl` = 20×1,000 = **20,000 行**；`world-logic-batch-001..050.jsonl` = 50×1,000 = **50,000 行**。合计 **100,000 行 / 104 文件 / 103,904,802 B（99.1 MiB）已核实存在**。
  README 的分组声称与实测完全吻合（早期版本只下到 30,000 行，`software`/`world-logic` 两组当时标为未核实，现已补全）。
- **Schema**：与 laya-evals / jev 决策格式**完全一致**，无需重打标：
  ```json
  {"state": {...},
   "questions": {"<name>": {"type": "choice|noul|score", "instructions": "...", "criteria": {...}}},
   "targets": {"<name>": <label>},
   "_meta": {"id","family","domain","language":"zh","label_source":"constructed","evidence":{...}}}
  ```
  与本地 `$WORK\src\laya\research\evals\zh_decision_bench.jsonl` 的 `state/questions/expected` 同构。
- **内容构成（实测）**：
  - `batch-001..005`（1,000 行，20 family × 50 行）：
    `ticket-routing-intent/language-mix/severity-priority/negative-disambiguation/compound-priority-v1`（5 个路由 family = 250 行）、`eligibility-*`（4）、`probability-*`（2）、`threshold-*`（3）、`insufficient-*`（3）。
    路由选项键为英文（`unknown/account/billing/technical/fulfillment`）但带中文定义 → **需重映射**到我们的 5 团队。
  - `software-batch-*`（20,000 行，2026-10-13 补测）：10 个域（api_data / bug_triage / ci / code_review / dependencies / performance / release / requirements / security / testing）× 10 个 aspect（coverage / dependency / evidence / gate / next_step / ownership / release_gate / risk / route / severity）× 200 行。**`software-*-route` / `-severity` / `-next_step` 三个 aspect 是现成的「工单 → 负责方 + 严重度 + 下一步」监督，可直接映射 choice/score/choice 三问。**
  - `world-logic-batch-*`（50,000 行，2026-10-13 补测）：11 个域（biology / civic_safety / communication / daily_living / economy / geography / history_culture / physical / society / technology，每个 2,500 行）× 10 个推理类型（classification / conservation / contrast / interaction / missing / probability / safety / sequence / temporal / transfer）× 250 行。
  - software + world-logic 两组的问题类型总计：**choice=65,000 / noul=75,500 / score=9,000**。
  - `threshold-sla-duration-v1`（50 行）：`state={incident, unavailable_minutes}` → `impact`（score，0–3，按分钟段）、`escalation`（choice standard/priority）、`upgrade_trigger`（noul）。**这是全网唯一核实到的 SLA/优先级式有序监督。**
  - `协商与冲突对话` 的意图选项含 `complaint:"表达不满"` → 可作投诉升级的弱监督。
  - 路由选项实测：200 行用 `{unknown, account, billing, fulfillment, technical}`，另 50 行用 `{account, billing, fulfillment, security, technical}`（多一个 security 团队）。
- **映射到三原语**：
  - `choice`：ticket-routing-*（250 行，需把 billing→账务、technical→技术支持、fulfillment→售后）+ 客户支持对话 intent/policy/repair/turn（1,600 行，但选项是"下一步动作"而非团队，需改造）+ software-*-route（2,000 行，owner 7 选项，含 unknown）。
  - `score`：dialogue-support-score（200 行，0–3 复杂度）+ software-*-severity（2,000 行）+ threshold-sla（50 行）→ 需把 0–3 映射/扩展到 1–5。
  - `noul`：info_complete / turns_known 等布尔问（大量，但语义是"信息是否充分"，**不是**"是否要求退款/升级"）→ 语义不匹配，只能作为 `noul` 的**格式与否定逻辑**监督，不能直接充当业务标签。
- **致命局限（重要）**：文本是**模板化短句合成中文**（如「用户发现同一订单被重复扣款。」），`state` 常为结构化 dict（字段名+值），不是自然客户口语。它教的是**决策格式与推理模式**，不教自然工单语言。自然性必须由 zh-tw 集或 teacher 合成补足。
- **存储**：json，实测 **104 文件 / 100,000 行 / 103,904,802 B（99.1 MiB）**。本地已全量落盘在 `$WORK\data\raw\mri\`。

### 1.2 albertkingdom/custom-service-ticket-zh-tw ⭐ 自然工单文本来源

- **ID / URL**：https://huggingface.co/datasets/albertkingdom/custom-service-ticket-zh-tw
- **License**：`cc-by-4.0`（HF API metadata，已核实）
- **Gated / Auth**：gated=False，无需登录，直接 resolve 下载（已实测 train.jsonl 2,497 行可下）
- **Language**：`zh`，`language_bcp47: zh-TW`。**繁体台湾国语，非大陆简体**（词汇如「超商」「攏」）→ 需繁简转换 + 词汇本地化。
- **Size**：3,148 行 = train 2,497 / validation 326 / test 325（README 与实测一致）
- **Schema**（嵌套）：
  ```
  ticket.{ticket_text, category, primary_intent, order_or_product_info,
          emotion_intensity (int 1-5), needs_human (bool)}
  is_multi_intent, label_status ("confirmed"|"boundary"),
  validator_agreement_ratio, needs_human_agreement_ratio, emotion_dispersion
  ```
- **分布（train 实测）**：
  - 6 类：客訴投訴 439 / 帳號/系統問題 438 / 商品諮詢 433 / 退換貨申請 399 / 帳務爭議 396 / 物流查詢 392（均衡）
  - `emotion_intensity` 1–5：507/495/472/502/521（均衡）→ **正好是 1–5 有序 score 监督**
  - `needs_human`：True=1,638 / False=859
  - `ticket_text` 长度：min 30 / max 161 / **avg 66.2 字**（自然口语）
  - train 全部 `label_status="confirmed"`（由独立 5× 多数投票 LLM 复核）
- **来源**：**全合成**，teacher LLM（Claude/GPT via OpenRouter）schema-first 结构化生成；README 明示「不含任何真實客服工單，無個資/版權疑慮」。
- **映射到三原语**：
  - `choice`：退換貨申請→售后/退款处理；帳務爭議→账务；帳號/系統問題→技术支持（弱）；客訴投訴→投诉升级；物流查詢+商品諮詢→售后。**缺口：没有真正的技术支持类、没有「退款处理 vs 售后」细分。**
  - `score`：`emotion_intensity` 1–5 直接可用（情绪强度 ≈ 紧急度，弱代理）。
  - `noul`：`needs_human` 或 (客訴投訴 + 退換貨申請) 作为退款/升级需求代理（弱代理）。
- **局限**：zh-TW；仅 2,497 train 行；合成；无语义化 SLA/紧急度定义。

### 1.3 CodyQin/zh-decision-bench ⭐ 中文工单黄金评测集（**仅评测**）

- **Repo**：https://github.com/CodyQin/zh-decision-bench （Apache-2.0 code；dataset CC BY 4.0）
- **HF 发布**：`CodyQin/zh-decision-bench`（configs `voice_routing` / `business_scenarios`，见 repo `results/hf_publish/README.md`）
- **License**：Dataset **CC BY 4.0**（MASSIVE 衍生行归因 Amazon MASSIVE，同为 CC BY 4.0；合成行为原创无第三方权利）——已核实（README License 节 + LICENSE-DATA）。
- **内容（v0.2，实测）**：`business_scenarios.jsonl` 55 行 = `ecommerce_cs` 34 行 + `content_moderation` 21 行。
  `ecommerce_cs` 行**同时带三个问题**：`route`（choice 4 选项：物流配送/退换售后/支付问题/账户安全）、`urgency`（score 3 级：不紧急/一般/紧急）、`escalate`（noul）。
  另有 `voice_routing.jsonl` 323 行（MASSIVE zh-CN dev，6 路 choice）。
- **为什么不能当训练数据**：它是**评测/校准集**（55 行，且本地 laya evals 已有 v0.2 的 40 行版本），规模远不足以训练。**但它是唯一格式与业务语义完全对齐的中文工单三原语样本 → 必须作为 held-out eval + 少量 few-shot 种子。**
- 本地副本：`$WORK\src\laya\research\evals\zh_decision_bench.jsonl`（219 行 = 179 MASSIVE + 40 合成，v0.1 版本）。

### 1.4 QingshanAI/ecom-after-sale-synthetic 等（ModelScope，售后场景语料）

- **URL**：https://modelscope.cn/datasets/QingshanAI/ecom-after-sale-synthetic
- **License**：API `License` 字段 = `Apache License 2.0`（已核实）。README 附加条款：允许训练/研究/商用，**禁止整体转售或冒充真实独家数据二次售卖**，建议署名。→ 实质可用，但**再分发条款是自定义的，需法务过一眼**。
- **Size / Schema**：997 行中文，AI 合成（本地 deepseek-r1:14b），字段 `scene / user / bot / tone`。
  scene 分布：退换货 550、质量问题 117、维权投诉 107、差评 98、补偿方案 86、退款 20、质量 5、退货 4。
- **映射**：`scene`（退换货/维权投诉/退款）→ 弱 `choice`；`tone`（安抚 667 / 致歉 213 / 专业 26）→ 弱 `score` 情绪维度；无 `noul` 标签。
- **同作者姊妹集**（均为 apache-2.0，schema 同）：`QingshanAI/ecom-customer-service-synthetic`（992 行）、`QingshanAI/ecom-logistics-synthetic`（987 行）。
- **局限**：是「客服回复」对话语料，不是「工单分类」语料；量小（3×~1000）；无紧急度。

### 1.5 icebergyang/decision-model-tutorial（ModelScope，迷你教程集）

- **URL**：https://modelscope.cn/datasets/icebergyang/decision-model-tutorial
- **License**：apache-2.0（已核实）
- **Size / Schema**：88 行中文合成工单：`data/demo.jsonl`(8) / `calibration.jsonl`(32) / `test.jsonl`(48)；字段 `id/split/family/text/label/reason`；4 类均衡（物流/售后/账户/信息不足）。
- **映射**：可直接转 laya 格式的 `choice` 4 类小样本；含 calibration 划分（对温度重标定有用）。
- **局限**：88 行，仅 eval/教程用途。README 提到 `Shanghai_AI_Laboratory/Intern-Decision-0.8B` 与 `Qwen/Qwen3.5-0.8B`（**后者 id 未核实**）。

### 1.6 BAAI/COIG（中文指令集，非工单）

- **URL**：https://huggingface.co/datasets/BAAI/COIG
- **License**：`apache-2.0`（README metadata + License 节：BAAI 创作部分 Apache-2.0，另含 MIT 的 Unnatural Instructions 与 CC-BY-SA-4.0 的 leetcode 子集）——已核实。
- **内容**：translated 66,858 / exam 63,532 / human-value 34,471 / counterfactural-chat 13,653 / leetcode 11,737，共 5 个子集，338MB。
- **映射到三原语**：**无直接映射**（没有工单文本、没有类别/紧急度标签）。可作为中文通用语言能力的补充预训练/正则化数据，或作为 teacher 合成时的风格参考。
- **局限**：338MB（可承受但没必要全下）；与工单域相距远。

### 1.7 其他已排除 / 未核实的候选

| 候选 | 结论 | 依据 |
|---|---|---|
| `THU-StarLab/CustomerService` | **EXCLUDED**：license=unknown + `gated:auto`（需申请+同意条款） | HF API metadata |
| ChnSentiCorp（`seamew/ChnSentiCorp` 1,549 dl / `lansinuote/ChnSentiCorp` 12k 行） | **许可未核实 → 暂不用**。两仓库 API 均**无 license 字段**；`seamew` 版甚至没有 dataset card（页面显示 "No dataset card yet"）。lansinuote 版 12k 行是酒店/图书评论情感二分类，域也不匹配 | https://hf-mirror.com/datasets/seamew/ChnSentiCorp ，https://hf-mirror.com/datasets/lansinuote/ChnSentiCorp |
| `modelscope/banking77`（PolyAI/banking77 同源） | **英文**，非中文。且 **license 冲突**：ModelScope License 字段写 "Apache License 2.0"，README 版权信息节写 "Creative Commons Attribution 4.0 International" → 不采用 | https://modelscope.cn/datasets/modelscope/banking77 |
| `DatatangBeijing/90000sets-Multi-domainCustomerServiceDialogueTextData` | **EXCLUDED**：README 明示「版权归数据堂所有，商用数据」，而 API License 字段却显示 "Apache License 2.0" → **许可冲突 + 商用付费数据**。且 182 万条多轮对话**无类别/紧急度标签**，对三原语无用 | ModelScope API |
| `decisionmodelhub/support-urgency` | 英文、二分类 urgency（非 1–5）→ 不用 | HF API |
| `LocalLLaMA/typed-decisions` | 英文 apache-2.0 → 仅作**格式参考/teacher 风格参考** | HF API |
| `typesafe/evalsafe-customer-service`、`gorkemsevinc/customer_support_tickets`、`Console-AI/IT-helpdesk-synthetic-tickets` | 英文且/或无 license → 不用 | HF API |
| SMP-ECISA | **未核实 / 未找到**：GitHub platform_search 无结果，web_search 仅噪声 | — |
| COIG（中文开放指令通用） | 已核实存在（见 1.6），但与工单无关 | https://huggingface.co/datasets/BAAI/COIG |
| datawhalechina/jev-cookbook | **无任何中文 CS 训练数据**。唯一 app `apps/dsh-jev-decision` 的 `docs/SOURCES.md`（1,366 字，2026-09-22）只引用工具链（DSH npm 0.1.5-rc.2、https://docs.typesafe.ai/api、https://docs.typesafe.ai/models 默认 jev-1.13.0）。`examples/three-primitives.json` 是会议纪要示例；`examples/routing-cases.json` 明示「人工编写的三个联调案例，不是业务评测集」 | https://github.com/datawhalechina/jev-cookbook |
| laya 自带 benchmark 数据 | `$WORK\src\laya\benchmarks\` 只有速度/延迟基准（`bench_*.py`、results JSON、`bench-rtx4070.png`），**无数据集文件、无下载 URL**。BENCHMARKS.md 记录的评测混合：AG News 4 类、DAIR Emotion 6 类、banking77 77 类、MASSIVE 51 语言（zh-CN 0.650 acc / 0.219 ECE；zh-TW 0.610/0.266）、XNLI、SST-5（score 原语最弱，0.372） | 本地文件 |
| ModelScope「工单/优先级/投诉分类/退款」检索 | 中文检索 `工单分类`/`投诉分类`/`优先级` 均 **Total=0**；`客服` 命中 37 条但绝大多数是**语音 TTS 合成语料**（DatatangBeijing/*CustomerService* 音频）或无 schema 的社区集（`customer_json`、`E_commerce_Customer_Service`、`Agent-AI-customer-service`，均无 README、schema/许可细节**未核实**） | ModelScope API |

---

## 2. 许可阻塞点（License blockers）

1. **`THU-StarLab/CustomerService`** — `license: unknown` + `gated: auto`（需申请）→ **排除**。
2. **`DatatangBeijing/90000sets-Multi-domainCustomerServiceDialogueTextData`** — README 说「版权归数据堂所有，商用数据」，但 API License 字段写 Apache-2.0。**许可自相矛盾 + 商用付费** → **排除**。
3. **`modelscope/banking77`** — API License 写 Apache-2.0，README 版权节写 CC BY 4.0。**许可冲突**（且是英文）→ **排除**。
4. **ChnSentiCorp（seamew / lansinuote）** — 两个仓库 API **均无 license 字段**，`seamew` 版连 dataset card 都没有。**许可未核实 → 暂不使用**。ChnSentiCorp 原始发布方通常为研究用途，需在采用前取得明确授权。
5. **`QingshanAI/ecom-*`（3 个）** — API License = Apache-2.0，但 README 加了「禁止整体转售/冒充独家数据 + 建议署名」条款。**可训练可用，但再分发条款非标准 SPDX → 需法务确认**。这是唯一带自定义附加条款的可用候选。
6. 未发现任何明确 **NC / non-commercial / research-only** 的中文工单候选（这是好消息：真正的 blocker 是「不存在」而非「不可商用」）。

**结论：`MoriyaResearchInstitute/MRI-decision-data-100k`（MIT）与 `albertkingdom/custom-service-ticket-zh-tw`（CC BY 4.0）是许可最干净的两个；`CodyQin/zh-decision-bench`（CC BY 4.0）是评测侧唯一干净选项。**

---

## 3. 推荐数据构成

总量目标 30k（取预算下限，保证 4 epochs 在 4h 内；CPU-only 下 30k×4epochs 对 322M encoder 约 2–3h，需实测验证）。

| 用途 | 来源 | 条数 | 许可 | 说明 |
|---|---|---|---|---|
| **`choice` 主力** | `MoriyaResearchInstitute/MRI-decision-data-100k` → ticket-routing-* 250 行 + 客户支持对话 intent/policy/repair/turn 1,600 行 | ~1,850 | MIT | 需把英文选项键重映射为 5 团队；support-turn/repair 的选项是「下一动作」需改写为团队语义 |
| **`choice` 自然文本** | `albertkingdom/custom-service-ticket-zh-tw` train 2,497 行（6 类） | 2,497 | CC BY 4.0 | 繁→简转换；类别→团队映射（见 1.2）；留 300 条做 eval，不入 train |
| **`choice` 评测** | `CodyQin/zh-decision-bench` business_scenarios `ecommerce_cs` 34 行 + 本地 `zh_decision_bench.jsonl` 合成部分 | 34（+40 本地副本） | CC BY 4.0 | **只作 held-out eval** |
| **`score` 主力** | zh-tw `emotion_intensity` 1–5（train 2,197 条去 eval 后） | ~2,200 | CC BY 4.0 | 唯一自然文本 1–5 监督；需定义「情绪强度→紧急度」的映射说明 |
| **`score` SLA/优先级语义** | MRI `threshold-sla-duration-v1` 50 行 + `dialogue-support-score` 200 行 | 250 | MIT | 0–3 → 需重标为 1–5 或明确告知模型该题为 1–3 |
| **`score` 合成主力** | teacher 合成（见第 7 节） | ~8,000 | 合成产物自行许可 | 按「退款诉求/升级威胁/重复扣款/安全风险」等信号定义 1–5 |
| **`noul` 主力** | teacher 合成（显式「要求退款」vs「仅咨询」vs「要求升级/投诉」） | ~8,000 | 合成产物自行许可 | MRI 的 noul 语义是「信息是否完整」，**不可直接用** |
| **`noul` 真实锚点** | zh-tw `needs_human`（1,638/859）+ 客訴投訴/退換貨申請 | ~2,500 | CC BY 4.0 | 弱代理，标注为 proxy |
| **格式/推理正则** | MRI language-dialogue 中 `noul` 布尔问（info_complete/turns_known/history_available） | ~3,000 抽样 | MIT | 教 noul 的否定与「信息不足→unknown」模式，不承载业务语义 |
| **温度重标定集** | `icebergyang/decision-model-tutorial` calibration 32 行 + zh-tw valid 326 行 + 合成 500 行 | ~860 | apache-2.0 / CC BY 4.0 | laya 出厂未校准（ECE 0.314→0.106 可重标定），**必须做** |
| **总计** | | **≈ 30,000** | | |

**为什么 score 和 noul 主要靠合成**：全开源生态里没有「中文工单 + 1–5 紧急度 + 退款/升级诉求」的真标注集。MRI 的 score 是 0–3 复杂度，语义偏；zh-tw 的 emotion_intensity 是 1–5 但语义是情绪而非紧急度。要拿到**语义正确**的 score/noul，只能自己造。

---

## 4. 去污染 / 去重计划

1. **eval 集与 train 集严格隔离**：`CodyQin/zh-decision-bench`（55 行）、本地 `zh_decision_bench.jsonl`（219 行）、`icebergyang` test 48 行**只进 eval，绝不进 train**。在脚本里用白名单路径强制。
2. **精确去重**：对 `ticket_text` / `state` 序列化后做 SHA1；同 hash 只留 1 条（模板化合成的 MRI 极可能大量重复同构句，需实测重复率）。
3. **近重复去重**：字符 3-gram Jaccard ≥ 0.8 或 MinHash（128 perm，阈值 0.7）聚类，簇内保留 label_status=confirmed 且 validator_agreement_ratio 最高的一条。
4. **跨集去重**：MRI（模板句）与 zh-tw（自然句）、合成集之间两两跑一次 MinHash，防止同一工单语义在多集重复出现造成训练集-评测集泄漏。
5. **与 laya 预训练语料去污**：laya/mmBERT 的预训练语料未公开清单（**未核实**），无法精确比对。缓解：训练前把最终 train 集与 `MASSIVE zh-CN` dev 做一次 13-gram 重叠检测（MASSIVE 是 laya 评测集，重叠即污染），重叠样本剔除。
6. **合成集内部去重**：teacher 合成时记录 prompt seed；prompt 相同则比对输出，输出相同直接丢弃。
7. **记录**：每个子集写 `provenance.jsonl`（source_id / license / dedup_cluster_id / proxy 标记），随模型一起发布，满足 CC BY 4.0 署名要求。

---

## 5. 繁体 → 简体 / 词汇本地化

zh-tw 是台湾繁体。方案：OpenCC `tw2sp`（**依赖项需新增，未经 AGENTS.md 批准前先确认**）+ 人工词表（超商→快递柜/驿站、攏→都、蝦皮风格→淘宝/京东风格）。转换后必须抽样 100 条人工检查，避免「台湾国语」腔调。若 OpenCC 不可用，直接让 teacher 在合成阶段产出大陆简体，zh-tw 仅作为**评测集的难度补充**与**情绪强度分布参考**。

---

## 6. Teacher 合成回退方案（核心）

### 6.1 为什么必须走合成
- 现成数据只能覆盖 `choice` 的 ~4,300 条，`score`/`noul` 的语义正确数据接近 0。
- 预算要 30k–50k，缺口 25k+ 只能自己造。
- **不允许用 Jev API 打标**；但**允许**用开源 teacher 生成+打标。

### 6.2 Teacher 选型（已核实许可）

| 模型 | 许可 | 参数 | CPU 可行性 | 结论 |
|---|---|---|---|---|
| `Qwen/Qwen3-1.7B` | **apache-2.0**（已核实，HF 页面 License 字段） | 1.7B | 32,768 ctx；CPU int4/int8 推理约 3–8 tok/s | **首选**（中文强、JSON 输出稳） |
| `Qwen/Qwen2.5-0.5B-Instruct` | **apache-2.0**（已核实） | 0.49B | CPU 约 15–25 tok/s | 备选（快，但中文工单多样性差） |
| `Qwen/Qwen3-0.6B` | apache-2.0（已核实） | 0.6B | 介于两者 | 备选 |
| `internlm/internlm2_5-1_8b-chat`、`THUDM/GLM-4-9B-0414` | 未核实（API 308 未取到） | — | — | **未核实，不作为首选** |
| `Shanghai_AI_Laboratory/Intern-Decision-0.8B`、`Qwen/Qwen3.5-0.8B` | 未核实 | — | — | **未核实**（后者 id 疑似不存在，勿采用） |

来源：https://hf-mirror.com/Qwen/Qwen3-1.7B ，https://hf-mirror.com/Qwen/Qwen2.5-0.5B-Instruct ，https://hf-mirror.com/Qwen/Qwen3-0.6B

**若 1.7B 太慢**：用 `Qwen2.5-0.5B-Instruct` 做多样性生成，`Qwen3-1.7B` 只做质检重标（两阶段，见 6.5）。

### 6.3 Prompt 设计（关键：先造分布，再造句子）

**不要**直接让模型「生成 8000 条工单」——会得到高度同质的模板。按 laya 的 `state/questions/targets` 格式反向构造：

1. **先采样一个结构化场景配置**（这是多样性的来源）：
   ```
   domain ∈ {电商零售, 虚拟商品/游戏, SaaS订阅, 金融支付, 物流}
   order_state ∈ {已发货, 未发货, 已签收, 已退款, 部分退款}
   issue ∈ {重复扣款, 未收到货, 商品破损, 错发, 质量瑕疵, 账号被盗,
            自动续费争议, 优惠券未生效, 发票问题, 修改地址, 保修, 价保}
   customer_emotion ∈ {平静, 不满, 愤怒, 焦急, 威胁投诉}
   explicit_refund_demand ∈ {true, false}     # 明确说"退款/退钱"
   explicit_escalation_demand ∈ {true, false} # 明确说"投诉/12315/找媒体/告你们"
   info_sufficiency ∈ {完整, 缺订单号, 缺截图, 完全没信息}  # 造 insufficient 样本
   ```
2. **让 teacher 依据该配置写出自然大陆简体工单**（40–120 字，口语，可含错别字/口语省略/表情），**同时**输出三原语答案。
   要求输出严格 JSON：
   ```json
   {"ticket": "...", "route": "售后|退款处理|技术支持|账务|投诉升级|信息不足",
    "urgency": 1, "refund_or_escalation_demand": true, "reason": "..."}
   ```
3. **规则引擎兜底校验**：`refund_or_escalation_demand=true` ⟺ ticket 中出现退款类词（退款/退钱/退货退款/退费）或升级类词（投诉/12315/消协/媒体/举报/工商）。不一致 → 丢弃或回炉。这一条把「teacher 随口胡说」变成「可验证的确定性标签」，是质量的关键。
4. **紧急度 1–5 的判定标准要写进 prompt**，避免 teacher 自由发挥：
   - 1 = 纯咨询，无损失，语气平和
   - 2 = 轻微不便，无金钱损失
   - 3 = 明确金钱损失或功能不可用，要求处理
   - 4 = 金钱损失明确 + 情绪强烈 + 时限压力（明天/今天/赶紧）
   - 5 = 威胁投诉/曝光/法律途径，或安全/盗号/数据泄露
   合成时按目标分布采样（如 20/25/25/20/10），避免全挤在 3。
5. **5 团队的定义也要写死**，尤其区分 售后 vs 退款处理 vs 投诉升级（这是现成数据最大的缺口）：
   - 售后 = 退换货、维修、破损、价保、保修
   - 退款处理 = 退款到账、重复扣款、部分退、优惠券/发票退款
   - 技术支持 = 账号登录/系统报错/功能不可用/App 崩溃
   - 账务 = 账单疑问、自动续费、订阅、支付方式、对账
   - 投诉升级 = 明确要求投诉、威胁曝光/法律、要求见负责人、多次未解决
   - 信息不足 = 无法判断（造 5–8% 这类样本，laya 的 `insufficient` family 表明这很重要）
6. **难例增强**：混合意图（同时要退款+投诉）、负面排歧（「不是账号问题，是扣款问题」——对应 MRI 的 negative-disambiguation family）、语言混合（中英混排订单号）。

### 6.4 保质量的四道闸

1. **规则闸**（6.3-3）：refund/escalation 词表与标签一致性；urgency 与情绪词/时限词共现校验。不匹配 → 丢弃。目标丢弃率 <15%，>30% 说明 prompt 有问题。
2. **二阶段重标闸**：用**第二个 teacher**（或同 teacher 不同 temperature）对每条只给 `ticket` 重新打标，两轮一致才留。一致率应 >85%；<70% 说明任务定义不清，停下来改 prompt 而不是继续生成。
3. **锚点集闸**：把 34 条 `ecommerce_cs` + 40 条本地合成 + 300 条 zh-tw 留出做「锚点」。teacher 在锚点上的准确率（route macro / urgency exact±1 / escalate）必须**先达标**（建议 route ≥0.85、urgency 邻级准确率 ≥0.90、escalate ≥0.90）才开始量产；每生成 2,000 条抽 100 条人工复核一次。
4. **分布闸**：每 1,000 条统计 route/urgency/noul 的分布与 token 长度分布，偏离目标分布 >10% 就调整采样权重。

### 6.5 如何验证合成质量（不能只看 teacher 自评）

- **与 laya 现有能力对齐**：合成集上先跑 `laya-multilingual` zero-shot，记录它在哪些 family 上错——这些恰是训练最需要的样本类型。
- **人审抽样**：随机 200 条 + 每个 route 各 20 条，人判 route/urgency/noul 三问，报告人-teacher 一致率。这是唯一能证明「teacher 没系统性错」的方法。
- **训练后 eval**：`CodyQin/zh-decision-bench` business_scenarios（route/urgency/escalate 三指标）+ 本地 `zh_decision_bench.jsonl` + zh-tw test 325 行（转 laya 格式后）。**上线前 score 必须报邻级准确率**（laya 的 SST-5 0.372、位置偏差 0/290 是已知弱点）。
- **校准验证**：训练后按 (type, option-count) 重拟合温度，报告 ECE 变化（laya 出厂 0.314 → 目标 <0.12）。

### 6.6 合成数据量与耗时估算（CPU）

- Qwen3-1.7B CPU int4：约 4 tok/s，单条工单 prompt+输出 ≈ 250 token → 约 60 s/条。
- 25,000 条 ≈ 416 CPU 小时 → **单机不可行**。
- 缓解：① 用 Qwen2.5-0.5B-Instruct 生成（约 20 tok/s，≈12 s/条，25k≈83 CPU 小时，仍偏慢）；
  ② **降量**：合成 10,000–12,000 条（48–144 CPU 小时），配合真实数据 ~5,000 条，总 16k，仍满足「3–5 万」的下限附近，4 epochs 更宽松；
  ③ 批量并发（多进程，按 CPU 核数扩）；
  ④ **优先保 score/noul 的语义正确**（这两项全靠合成），choice 可更多依赖真实数据。

---

## 7. 已核实 vs 推断（诚实标注）

**已核实（有 URL / API / 实测下载）**：
- 上述所有 HF 数据集的 license / gated / size_categories / language / format 字段（hf-mirror API）
- zh-tw 的全部行数、分布、schema、文本长度统计（train.jsonl 实测）
- MRI 的 **100,000 行 / 104 文件 / 99.1 MiB 全量实测**、family 构成、area 分布、问题类型计数、schema（逐文件下载计行）
- zh-decision-bench 的 55 行内容、三问结构、CC BY 4.0 许可、Apache-2.0 code 许可（GitHub raw README + hf_publish README）
- COIG 的 5 子集构成与 Apache-2.0/MIT/CC-BY-SA-4.0 混合许可
- ModelScope 各候选的 License 字段、行数、scene 分布（modelscope API）
- Qwen3-1.7B / Qwen3-0.6B / Qwen2.5-0.5B-Instruct 的 apache-2.0 许可
- laya-multilingual 的 apache-2.0 许可与模型卡全部限制（未校准、score 位置偏差、noul under-report、choice <20 options、head 256 token 预算）
- jev-cookbook 无 CS 数据；laya/benchmarks 无数据集

**未核实（明确标注）**：
- ChnSentiCorp 的实际许可条款
- ModelScope `customer_json` / `E_commerce_Customer_Service` / `Agent-AI-customer-service` / `customer-service-training-dataset2.0` 的 schema 与内部许可（API 无 README，文件列表端点 405）
- `internlm/internlm2_5-1_8b-chat`、`THUDM/GLM-4-9B-0414` 的许可（hf-mirror 模型 API 308 未取到）
- SMP-ECISA 是否存在（全网未找到）
- `Qwen/Qwen3.5-0.8B`、`Shanghai_AI_Laboratory/Intern-Decision-0.8B` 的 id 与许可
- laya/mmBERT 预训练语料清单（影响去污的彻底性）

**推断（非核实，需验证）**：
- 「30k×4 epochs 在 CPU 上 2–3h」是基于 322M encoder + 512–1024 token 的经验估算，**必须实测**
- Qwen 系列 CPU 推理吞吐（tok/s）为公开基准的量级推断，需在本机实测
- zh-tw 情绪强度 → 紧急度的映射合理性
- 合成目标分布（20/25/25/20/10）与 5 团队配额为先验设定，应依据锚点集实测调整

---

## 8. 立即可执行的下一步

1. ~~`git clone https://github.com/CodyQin/zh-decision-bench`~~ → 已改为从 HF `CodyQin/zh-decision-bench` 直取（`business_scenarios.jsonl` 55 行 + `voice_routing.jsonl` 323 行，CC BY 4.0），落盘 `$WORK\data\raw\zh-decision-bench\`。
2. MRI 全量（104 文件 / 100,000 行）已下到 `$WORK\data\raw\mri\`；zh-tw 三个 jsonl（3,148 行）已下到 `$WORK\data\raw\zh-tw\`。**下载环节已完成。**
3. 先跑一次「锚点达标测试」：Qwen3-1.7B 在 34+40+300 条锚点上的 route/urgency/escalate 准确率，达标后才开量产合成。
4. 落一个 `$WORK\data\build.py`：下载 → 繁简转换 → 选项重映射 → 规则闸 → MinHash 去重 → 输出 laya 训练 JSONL + `provenance.jsonl`。
