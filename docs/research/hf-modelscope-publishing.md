# Hugging Face 与 ModelScope 发布流程与合规（issue #11）

调研日期：2026-10-13
目标：为 mai-jev（`convaiinnovations/laya-multilingual` 中文优先 System One 决策模型微调）确定
**合法的许可证继承方式**、**数据许可义务**、**双平台发布形态与命令**，并交付一份用户可照做的 HITL 清单。

> 数据集层面的详细盘点见 `docs/research/datasets-and-licenses.md`（票 #4）。本文件只覆盖**发布侧**义务。

---

## 0. 结论摘要（TL;DR）

| 问题 | 结论 |
|---|---|
| 基座许可证 | **Apache-2.0，无附加条款**。可原样继承 Apache-2.0 发布。 |
| 训练数据许可 | MIT ×1 + CC BY 4.0 ×1（进训练集）+ CC BY 4.0 ×1（仅 eval）。CC BY 4.0 触发**署名 + 许可证声明 + 变更说明**义务，写在 model card / README 即可满足。 |
| 发布形态 | 一个 HF repo，总体积 **≈ 1.0 GB**（F16 safetensors 614MB + int8 ONNX ~330MB + tokenizer 32.8MB + configs）。fp32 ONNX（~1.29GB）**建议不放入主仓库**，用 git tag 分发。 |
| ModelScope 侧 | `modelscope` PyPI 包（当前 1.40.2，requires-python ≥3.10），`HubApi().create_model()` + `upload_folder()`。**没有官方 HF→ModelScope 一键镜像**，需自己按文件逐个上传（或本地推 HF 后再传 ModelScope）。 |
| HITL | 2 个 token：HF write token、ModelScope SDK access token；ModelScope 需手机号实名，但创建模型仓库本身不需要组织认证。 |
| 验收 | `scripts/verify_published_model.py --source hf|modelscope` 冒烟脚本（本文档第 6 节）。 |

---

## 1. 许可证继承（最关键一条）

### 1.1 证据

通过 Hugging Face Hub API 直接读取模型仓库的 `cardData` 与 README frontmatter：

- https://huggingface.co/api/models/convaiinnovations/laya-multilingual → `"license": "apache-2.0"`
- https://huggingface.co/api/models/convaiinnovations/laya → `"license": "apache-2.0"`
- https://huggingface.co/convaiinnovations/laya-multilingual/raw/main/README.md，YAML frontmatter：
  ```yaml
  ---
  license: apache-2.0
  library_name: transformers
  pipeline_tag: text-classification
  language: [multilingual, en, ..., zh, ...]
  tags: [laya, multilingual, mmbert, system-one, calibrated-decisions, rlcd, classification, routing, guardrails, moderation, commercial-use]
  ---
  ```
  正文末尾一行：`Apache 2.0 · Convai Innovations`
- https://huggingface.co/convaiinnovations/laya/raw/main/README.md 同样：`license: apache-2.0`，页脚 `Apache 2.0 · Convai Innovations`。
- tags 中含 `commercial-use`（HF 侧商用标记）；仓库**未 gated**，`siblings` 列表里**没有独立 LICENSE 文件**，也**没有 `extra_gated_prompt` / `custom_code` / 使用限制字段**。

### 1.2 结论

基座 `convaiinnovations/laya` 与 `convaiinnovations/laya-multilingual` 是**标准 Apache-2.0**：

- 没有 "just ask" / 联系作者条款
- 没有非商业（NC）限制
- 没有署名（attribution）以外的附加条件
- 没有基于研究用途的 use-based restriction
- HF 侧可直接下载（未 gated）

**因此 mai-jev 可以、也应当以 Apache-2.0 发布。**

### 1.3 需要履行的 Apache-2.0 义务

微调派生权重仍受 §4 约束，发布时必须：

1. 保留**版权声明与许可证声明**——随附一份完整 `LICENSE`（Apache-2.0 全文）。
2. 标注**修改**：`mai-jev` 是 laya-multilingual 的修改版本（中文工单域微调 + 温度标定）。
3. 保留上游版权归属：Convai Innovations（原 laya 系列）。**不要删除上游署名**。
4. 若含 NOTICE 文件则需保留（本项目不涉及）。

### 1.4 建议的 model card 措辞（许可证段）

```markdown
license: apache-2.0

---

## License

mai-jev is licensed under the [Apache License 2.0](LICENSE).

It is a **modified version** of [`convaiinnovations/laya-multilingual`]
(Apache-2.0), fine-tuned for Chinese customer-service ticket triage.

- Original work: Laya Multilingual — Copyright (c) Convai Innovations — Apache-2.0.
- Modifications: Chinese-domain RLCD fine-tuning, per-(primitive, option-count)
  temperature calibration, ONNX int8 export. Distributed under the same
  Apache-2.0 terms, as required by the original license.
```

---

## 2. 数据集许可与发布侧义务

### 2.1 三个数据源的许可（Hub API `cardData` 实测）

| 数据集 | 用途 | 许可 | 证据 |
|---|---|---|---|
| `MoriyaResearchInstitute/MRI-decision-data-100k` | 训练（主） | **MIT** | `https://huggingface.co/api/datasets/MoriyaResearchInstitute/MRI-decision-data-100k` → `"license":"mit"` |
| `albertkingdom/custom-service-ticket-zh-tw` | 训练（自然中文工单） | **CC BY 4.0** | `.../api/datasets/albertkingdom/custom-service-ticket-zh-tw` → `"license":"cc-by-4.0"` |
| `CodyQin/zh-decision-bench` | **仅 held-out eval** | **CC BY 4.0** | `.../api/datasets/CodyQin/zh-decision-bench` → `"license":"cc-by-4.0"` |

上游署名信息（应写进 model card）：

- MRI-decision-data-100k — Moriya Research Institute
- custom-service-ticket-zh-tw — albertkingdom（客服工单分類與結構化抽取（合成資料））
- zh-decision-bench — CodyQin

### 2.2 义务分析

**MIT（MRI-decision-data-100k）**
- 要求：保留版权与许可声明。
- 实操：训练权重构成"衍生作品"的概率极低，但保守做法是照常署名，零成本。

**CC BY 4.0（两个数据集）**
CC BY 4.0 施加四项义务（https://creativecommons.org/licenses/by/4.0/legalcode）：

1. **保留性（Retain）** — 保留版权声明、许可声明、免责声明。
2. **署名（Attribution）** — 提供 Credit：创作者、版权方、来源、许可证名称（若来源方已指定许可证，则不必另附许可副本）。
3. **变更说明（Indicate Changes）** — 声明是否做了修改。
4. **不附加限制** — 不得对 Material 施加 DRM 或技术性法律限制。

> **注意**：CC BY 4.0 的 3(a) 授予不得施加额外限制，意味着**不得把 CC BY 数据单独再分发成"仅此部分受 CC BY"的子包**而不给同等自由。好在权重是转换产物，不是 Material 的再分发，因此**不构成违规**——但 model card 里把来源说清楚是正确做法。

**关键判断：训练数据中的 CC BY 4.0 是否传染到模型权重？**

不传染。CC BY 4.0 §2(a) 授予的是对 Material 本身的复制/分发/展示许可；经训练产生的权重不是 Material 也不是其改编（adaptation 通常指可被人类感知的衍生作品，模型权重不是）。业内通行做法（SciBERT、BERT 及大量 HF 模型）是在 model card 声明训练数据来源与许可，而不因此改变模型许可。

**唯一有实感义务的是「变更说明」**——如果你发布训练数据构建脚本或数据本身，那份数据就是 Material，需要完整履行 CC BY 4.0。因此：

- **权重发布**：只需 model card 署名 + 变更说明（第二节表）。
- **数据集发布（如果做）**：需要 `LICENSE-CC-BY-4.0` + `NOTICE.md` + 每行/每文件的来源标注，**必须同样以 CC BY 4.0 发布**。

### 2.3 建议的 model card 数据段措辞（可直接复制）

```markdown
## Training data

| dataset | role | license | attribution |
|---|---|---|---|
| [MoriyaResearchInstitute/MRI-decision-data-100k](https://huggingface.co/datasets/MoriyaResearchInstitute/MRI-decision-data-100k) | training (format/decision-pattern supervision) | MIT | © Moriya Research Institute |
| [albertkingdom/custom-service-ticket-zh-tw](https://huggingface.co/datasets/albertkingdom/custom-service-ticket-zh-tw) | training (natural Chinese ticket text) | CC BY 4.0 | © albertkingdom |
| [CodyQin/zh-decision-bench](https://huggingface.co/datasets/CodyQin/zh-decision-bench) | **evaluation only, never trained on** | CC BY 4.0 | © CodyQin |

Additional training data was synthesised with open-weight teacher models
(`Qwen/Qwen3-1.7B`, Apache-2.0) from templates over the sources above; the
synthesis prompt templates and generation seeds are in the repository under
`data/`.

**Changes made:** these datasets were re-mapped into the Laya decision schema
(`state` / `questions` / `targets`), filtered, de-duplicated, and mixed with
synthetic samples. Labels were remapped onto mai-jev's 5-team / 1–5-urgency /
refund-or-escalate label space. The original raw data was not modified in place.

Attribution notice: the CC BY 4.0 datasets above are used under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
```

`CodyQin/zh-decision-bench` 只评测不训练这一点**必须在 model card 里写明**，同时论文/报告里的评测数字也应有对应说明（避免暗示同分布）。

### 2.4 仓库内是否要放 LICENSE / NOTICE

| 文件 | 放哪 | 内容 |
|---|---|---|
| `LICENSE` | HF repo 根 + GitHub repo 根 | Apache-2.0 全文 |
| `NOTICE`（可选） | 同上 | 一段：`mai-jev / Copyright (c) 2026 Maicarons. Derived from Laya Multilingual / Copyright (c) Convai Innovations, Apache-2.0.` |
| `THIRD_PARTY_NOTICES.md` | HF repo 根 | 第 2.3 节数据表 + 模型表 |

ModelScope 侧的模型仓库**同样上传 `LICENSE`**；ModelScope 支持 markdown README 作为 model card，会自动渲染。

---

## 3. 发布形态与体积

### 3.1 产物实测

在 `$WORK\runs\smoke\`（与上游同构的导出目录）实测：

| 文件 | 实测大小 |
|---|---|
| `model.safetensors` | **614.0 MB**（170 个张量，**dtype 全为 `F16`**） |
| `tokenizer/tokenizer.json` | **32.8 MB**（256k 词表） |
| `tokenizer/tokenizer_config.json` | <1 MB |
| `config.json`（ModernBERT 编码器） | ~4 KB，`"dtype": "float32"`, `vocab_size: 256000`, `num_hidden_layers: 22`, `hidden_size: 768`, `max_position_embeddings: 8192` |
| `encoder/config.json` | <1 KB |
| `rl_agent_config.json` | <1 KB（决策头配置） |

> **关键修正**：上游与本项目当前导出的是 **F16**，不是 fp32。614 MB / 170 张量 → 约 3.07e8 参数，与「322M（含词表与头）」一致。
> 因此 **HF 仓库的 `model.safetensors` 应为 ~614 MB F16**，不需要额外的 fp32 safetensors（要 fp32 可另打 tag）。

ONNX 导出（`scripts/export_onnx.py`，opset 17，dynamic axes 支持变长 batch/seq）：

| ONNX 变体 | 估算体积（3.22e8 参数） | 说明 |
|---|---|---|
| `model.onnx`（fp32） | **≈ 1.29 GB** | 精度基线，靶机跑不动 |
| `model_int8.onnx`（动态量化 int8） | **≈ 0.33 GB** | 部署主力 |
| `model_q8.onnx`（静态 QDQ int8） | ≈ 0.34 GB | 可选，opset 差异更大 |

体积来自参数数目：fp32 = 4 B/param，int8 ≈ 1 B/param（含少量 fp32 scale/zero-point 开销）。

### 3.2 推荐仓库文件清单

```
Maicarons/mai-jev-<tag>/
├── README.md                       # model card（含第 2.3 节数据表 + 许可证段）
├── LICENSE                         # Apache-2.0 全文
├── NOTICE                          # 上游归属
├── THIRD_PARTY_NOTICES.md          # 数据集 + 基座的许可与署名
├── .gitattributes                  # *.safetensors filter=lfs diff=lfs merge=lfs -text
├── config.json                     # 编码器配置
├── encoder/config.json
├── rl_agent_config.json            # 决策头配置（choice/score/noul）
├── model.safetensors               # 614 MB  F16 主权重
├── onnx/
│   ├── model_int8.onnx             # ~330 MB  部署主力（靶机）
│   └── model_int8.onnx.data        # 若用 external-data 方案
├── tokenizer/
│   ├── tokenizer.json              # 32.8 MB
│   └── tokenizer_config.json
├── calibration/
│   └── temperatures.json           # 每 (primitive, option_count) 的温度 + ECE
├── eval/
│   └── metrics.json                # zh-decision-bench 结果、ECE、base 对比
├── scripts/
│   ├── train_rlcd.py
│   ├── build_dataset.py
│   ├── export_onnx.py
│   └── verify_published_model.py   # 本仓库根 scripts/ 的副本
├── requirements.txt                # 顶层依赖
└── requirements.lock.txt           # 精确版本（pip freeze 子集）
```

**体积预算**

| 组成 | 体积 |
|---|---|
| safetensors | 614 MB |
| tokenizer | 33 MB |
| onnx int8 | 330 MB |
| 其它（configs / docs / scripts） | < 3 MB |
| **主仓库合计** | **≈ 0.98 GB** |

**控制规则：**
- **fp32 ONNX（1.29 GB）不进主仓库**，用 git tag 分发：`git tag fp32-onnx` 指向同一 commit，只在 tag 上额外 push `onnx/model_fp32.onnx`。想要的人自己 checkout tag。
- 不放 `checkpoint_latest/`、训练日志、`data/raw`（99 MB+）。
- tokenizer 只放 `tokenizer.json` + `tokenizer_config.json`；**不放 `vocab.txt` / `merges.txt`**（SentencePiece 双份冗余，`tokenizer.json` 已含全部信息）。
- `.gitattributes` 必须为 safetensors/onnx 声明 LFS，否则 HF 网页端无法预览且 CLI 推送会慢一个数量级。

### 3.3 仓库命名与版本

- 建议一个 **主仓库** `Maicarons/mai-jev-<domain>`（如 `Maicarons/mai-jev-zh-support`），所有版本走 **revision**，而不是开多个仓库。
- 版本用语义化：checkpoint 用 HF revision tag `v0.1.0`；fp32 ONNX 用 `fp32-onnx`。
- 在 README 顶部固定一段「⚠️ 未完成 temperature calibration」警告，直到 ECE ≤ 0.10 达成（对齐上游 laya 的诚实标注风格）。

---

## 4. ModelScope 侧

### 4.1 SDK

PyPI `modelscope`，当前版本 **1.40.2**，`requires_python >= 3.10`
（https://pypi.org/pypi/modelscope/json ；项目本身也是 Apache-2.0）。

核心 API（取自 `modelscope/hub/api.py`，master 分支）：

```python
class HubApi:
    def __init__(self, endpoint: str = None, token: str = None, ...)
    def create_model(self, model_id: str, **kwargs) -> str          # 建仓
    def get_model_url(self, model_id: str) -> str
    def upload_folder(self, repo_id: str, folder_path=None, **kwargs)   # 传目录（大文件/多文件）
    def upload_file(self, repo_id=None, path_or_fileobj=None, **kwargs) # 单文件

class ModelScopeConfig:
    def save_token(token) / get_token() / save_git_token(git_token) / get_git_token()
```

`modelscope/hub/snapshot_download.py` 提供 `snapshot_download(model_id, revision=..., cache_dir=...)`。

### 4.2 鉴权

1. **网页端**取 token：https://www.modelscope.cn/profile/me → 「访问令牌 / Access Token」→ 新建，勾选 `write`（上传）权限。
2. 环境变量（推荐，避免明文落盘）：

   ```
   MODELSCOPE_API_TOKEN=<你的 access token>
   ```

   `ModelScopeConfig.get_token()` 会读这个环境变量。
3. 或 CLI 一次登录持久化：
   ```bash
   modelscope login --token $env:MODELSCOPE_API_TOKEN
   ```
4. ModelScope 同时支持 **git lfs** 走 HTTPS：`ModelScopeConfig.save_git_token()` 保存的 git token 用于 `https://www.modelscope.cn/<user>/<repo>.git`。

### 4.3 能不能直接镜像 HF 仓库？

**没有官方一键 HF→ModelScope 导入。** ModelScope 网页端支持「从 HuggingFace 导入」的上线版本在不同时间点形态不一（多为站点内 `?hf=owner/repo` 的第三方工具），**不作为可依赖路径**。

可靠做法（本项目采用）：

> **本地同一目录 → 先推 HF，再推 ModelScope。** 产物只构建一次、只上传一次（各平台各一次），两份仓库内容天然一致。

### 4.4 meta 需要什么

| 项 | 要求 |
|---|---|
| 账号 | 手机号注册 + **实名认证**（大陆手机号，验证码）。这是 ModelScope 的硬性门槛。 |
| 仓库创建 | 个人账号即可，**不需要组织认证 / 企业认证**。 |
| 仓库可见性 | 建议选 **公开**（本项目目标）。私有仓库也支持。 |
| 协议 | 首次创建模型仓库需在网页端勾选魔搭社区发布协议。 |
| 组织 | 无。 |
| 备注 | ModelScope 服务中国大陆；同一份 README 中英皆可，建议中文在前。 |

### 4.5 可复制的命令序列（PowerShell）

> 全部命令**未执行**，等用户拿到 token 后再跑。所有 `huggingface`/`modelscope` 包若未安装请先按第 4.6 节安装。

```powershell
# ---- 0) 环境变量（PowerShell 会话级）----
$env:PYTHONIOENCODING='utf-8'
$env:HF_TOKEN='<HF write token>'                       # 用户提供
$env:MODELSCOPE_API_TOKEN='<ModelScope access token>'   # 用户提供
$env:HF_HUB_ENABLE_HF_TRANSFER='1'                     # 大文件加速（需 hf_transfer）

# ---- 1) 组装发布目录（唯一事实源）----
$repo = 'Maicarons/mai-jev-zh-support'
$stage = '$WORK\export\publish'
Remove-Item $stage -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $stage | Out-Null

# 从训练产物复制
Copy-Item '$WORK\runs\<run>\model.safetensors'      $stage
Copy-Item '$WORK\runs\<run>\config.json'            $stage
Copy-Item '$WORK\runs\<run>\rl_agent_config.json'   $stage
Copy-Item '$WORK\runs\<run>\encoder'                $stage -Recurse
Copy-Item '$WORK\runs\<run>\tokenizer'              $stage -Recurse

# ONNX int8
& python scripts\export_onnx.py `
    --ckpt "$WORK\runs\<run>" --out-dir "$stage\onnx"
# 然后把 fp32 产物重命名/量化为 model_int8.onnx（量化步骤见 export_onnx.py 头部说明）

# 文档与脚本
Copy-Item 'LICENSE'            $stage
Copy-Item 'NOTICE'             $stage
Copy-Item 'docs\model-card.md' "$stage\README.md"
Copy-Item 'scripts'            "$stage\scripts" -Recurse

# LFS 规则（必须）
"LFS 跟踪 safetensors/onnx" | Out-Host
Set-Content -Path "$stage\.gitattributes" -Encoding utf8 -Value @'
*.safetensors filter=lfs diff=lfs merge=lfs -text
*.onnx       filter=lfs diff=lfs merge=lfs -text
*.json       text
*.md         text
*.py         text
'@

# ---- 2) 推 Hugging Face ----
$env:PYTHONIOENCODING='utf-8'
& python -c @"
from huggingface_hub import HfApi
api = HfApi(token='$($env:HF_TOKEN)')
api.create_repo(repo_id='$repo', repo_type='model', exist_ok=True, private=False)
api.upload_folder(repo_id='$repo', folder_path=r'$stage', commit_message='mai-jev v0.1.0: zh support triage')
print(api.model_info('$repo').id)
"@

# 可选：单独打 fp32-onnx tag（不进 main）
# & python -c "from huggingface_hub import HfApi; HfApi(token='...').create_commit('$repo', ..., revision='fp32-onnx')"

# ---- 3) 推 ModelScope（同名仓库 Maicarons/mai-jev-zh-support）----
& python -c @"
from modelscope.hub.api import HubApi
import os
api = HubApi(token=os.environ['MODELSCOPE_API_TOKEN'])
repo_id = 'Maicarons/mai-jev-zh-support'   # 魔搭用户名/仓库名
api.create_model(model_id=repo_id, chinese_name='mai-jev 中文客服工单决策模型')
api.upload_folder(repo_id=repo_id, folder_path=r'$stage')
print(api.get_model_url(repo_id))
"@

# 若只想用命令行登录态：
# modelscope login --token $env:MODELSCOPE_API_TOKEN

# ---- 4) 验收（第 6 节）----
& python scripts\verify_published_model.py `
    --source hf        --repo $repo
& python scripts\verify_published_model.py `
    --source modelscope --repo $repo
```

### 4.6 依赖安装（按需，现在未安装）

本机 venv `$WORK\venv` 现状（实测）：

```
huggingface_hub  OK
transformers     OK
torch            OK
onnxruntime      OK
modelscope       MISSING
optimum          MISSING
```

需要时安装（**不要现在装**）：

```powershell
& python -m pip install -U modelscope hf_transfer huggingface_hub
# int8 量化需要：
& python -m pip install -U optimum onnxruntime onnx
```

---

## 5. HITL 清单（必须由用户本人完成）

> 这一节是这张票最终交给用户的东西。**前面任何一步没做完，后面都跑不通。**

### 步骤 1 — Hugging Face write token

1. 打开 https://huggingface.co/settings/tokens
2. 点 **+ Create new token**
3. **Token name**：填 `mai-jev-publish`
4. **Token type**：选 **Write**（只读 token 无法上传）
5. 生成后**立即复制**（页面只显示一次），存进密码管理器
6. 把 token 值交给 agent（或自己在本地设 `$env:HF_TOKEN` 后让 agent 复用已登录环境）

### 步骤 2 — ModelScope access token

1. 打开 https://www.modelscope.cn/profile/me
2. 若未登录：**手机号 + 验证码注册并完成实名认证**（这一步 agent 无法代替，且是唯一的硬性人工门槛）
3. 页面内进入「**访问令牌 / Access Token**」
4. 点新建，**勾选写权限（Write / 上传）**，命名 `mai-jev-publish`
5. 复制 token

### 步骤 3 — ModelScope 建仓（agent 可代做，但需确认名称）

- 建议**与 HF 同名**：`Maicarons/<hf-repo-id>`，如 `Maicarons/mai-jev-zh-support`
- 需要确认：魔搭用户名是否就是 `Maicarons`。若不同，把下文命令中的 owner 换掉。
- 建仓时会要求勾选**魔搭社区发布协议**（法律同意，**必须由用户本人勾选**）。

### 步骤 4 — HF 协议 / 组织

- 个人账号发个人 repo，**无需任何协议勾选**。
- 若要发到 organization（需邀请），才需要组织管理员批准。

### 步骤 5 — 发布后确认

- 两个平台各打开一次模型页，确认 README 渲染正常、文件列表完整、体积符合预期。
- 运行第 6 节冒烟脚本。

### 需要用户回答的一个问题

> ModelScope 仓库的命名：与 HF 完全同名 `Maicarons/<repo>`（推荐），还是另取中文友好的名字？

### token 传递的安全约定

- **不要**把 token 写进任何仓库文件、`.env` 被提交、或 issue 评论。
- 推荐：用户自己设环境变量，agent 只读 `$env:HF_TOKEN` / `$env:MODELSCOPE_API_TOKEN`。
- 若必须由 agent 使用，会话结束后用户可在两个平台** revoke 并重建** token。

---

## 6. 验收口径：发布冒烟测试

### 6.1 验收标准

发布「成功」= 同时满足：

1. **文件齐全** — `config.json` / `model.safetensors` / `tokenizer/tokenizer.json` / `rl_agent_config.json` 都在平台上，且 `model.safetensors` 大小与本地一致（614 MB 附近）。
2. **可下载** — 能通过 `huggingface_hub.snapshot_download` / `modelscope.snapshot_download` 完整拉到本地缓存。
3. **可加载** — `transformers` 能构建模型并前向一次。
4. **可推理** — 中文样例上三个原语（`choice` / `score` / `noul`）都返回**非 NaN、在 [0,1] 范围内**的概率，且 `choice` 落在候选集内。
5. **ONNX 可加载**（若发布了 int8） — `onnxruntime.InferenceSession` 能建 session 并跑一次。

### 6.2 脚本

见仓库根目录 **`scripts/verify_published_model.py`**：

```powershell
& python scripts\verify_published_model.py --source hf
& python scripts\verify_published_model.py --source modelscope
```

设计要点（对应交付要求）：

- `--source hf|modelscope` 切换下载通道；
- `huggingface_hub` 与 `modelscope` **惰性 import**，缺包时给出明确的 `pip install` 提示并以非零码退出，**不自动安装**；
- 强制缓存到 `$WORK\hf-cache` / `$WORK\ms-cache`（系统盘空间紧张）；
- 打印三个原语的中文样例输出；
- 退出码 0 = 全过；非 0 = 失败，附带断言原因。

### 6.3 期望输出（发布成功后）

```
[verify] source=hf  repo=Maicarons/mai-jev-zh-support
[verify] downloading snapshot -> $WORK\hf-cache
[verify] files: 8  total 0.98 GB
[verify] safetensors size: 614.0 MB (expected ~614 MB)
[verify] transformers: loaded 170 tensors, dtype float16
[verify] onnx int8: onnxruntime session created (providers: CPUExecutionProvider)
[verify] --- sample ---
[verify] state: 您好，3 月的账单被重复扣款两次，请今天之内退还多收的费用，否则我们将取消订阅。
[verify] choice  route        -> refund_team   p=0.881
[verify] score   urgency      -> 4 / 5        p=0.742
[verify] noul    refund_asked  -> True         p=0.903
[verify] OK: published model loads and answers all three primitives.
```

> 概率具体数值取决于最终 checkpoint，上表只是形状示意。冒烟测试只断言**结构合法**（非 NaN、范围合法、`choice` 在候选集内），不断言数值质量——质量由 `docs/research` 的评测口径把关。

---

## 7. 未决 / 交给后续票

| 项 | 归属 |
|---|---|
| 最终 repo id 后缀（`mai-jev-zh-support` vs 其它） | 用户定名 |
| ModelScope 账号是否实名、用户名是否 = `Maicarons` | 用户确认（HITL 步骤 2–3） |
| fp32 ONNX 是否需要单独 tag 发布 | 等量化票结论 |
| README 中英文排版比例 | 票 #5 发布执行时定 |

---

## 参考

- 基座模型页 https://huggingface.co/convaiinnovations/laya-multilingual · https://huggingface.co/convaiinnovations/laya
- Hub API https://huggingface.co/api/models/convaiinnovations/laya-multilingual
- ModelScope PyPI https://pypi.org/pypi/modelscope/json
- ModelScope SDK 源码 `modelscope/hub/api.py`（`create_model` / `upload_folder` / `upload_file`）
- ModelScope 文档 https://www.modelscope.cn/docs/models/upload
- CC BY 4.0 法律文本 https://creativecommons.org/licenses/by/4.0/legalcode
- HF 仓库管理文档 https://huggingface.co/docs/hub/repositories-managing
- `scripts/export_onnx.py`（本仓库，opset 17，dynamic axes）
- `docs/research/datasets-and-licenses.md`（票 #4）