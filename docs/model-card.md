---
license: apache-2.0
language:
  - zh
  - multilingual
library_name: transformers
pipeline_tag: text-classification
tags:
  - system-one
  - calibrated-decisions
  - rlcd
  - chinese
  - customer-support
  - ticket-triage
  - onnx
  - int8
  - commercial-use
datasets:
  - MoriyaResearchInstitute/MRI-decision-data-100k
  - albertkingdom/custom-service-ticket-zh-tw
  - CodyQin/zh-decision-bench
---

# mai-jev-zh-support

A Chinese-first **System One** decision model for customer-service ticket triage,
fine-tuned from [`convaiinnovations/laya-multilingual`](https://huggingface.co/convaiinnovations/laya-multilingual)
(322M params, mmBERT-base encoder + choice / score / noul decision heads) with
RLCD and per-primitive temperature calibration.

"System One" here means a **single forward pass produces calibrated probabilities
for three primitives at once** — no chain-of-thought, no autoregressive decoding —
so it stays cheap enough to run on a CPU-only machine.

The model decides, for one Chinese support ticket, three things:

| Primitive | Question | Label space |
| --- | --- | --- |
| `route_team` (choice) | Which team should handle this ticket? | 售后 / 退款处理 / 技术支持 / 账务 / 投诉升级 |
| `urgency` (score) | How urgent is it? | 1–5 (5 = 非常紧急) |
| `needs_refund_or_escalation` (noul) | Does the customer explicitly demand a refund / payout / escalation? | true / false |

## License

mai-jev is licensed under the [Apache License 2.0](LICENSE).

It is a **modified version** of `convaiinnovations/laya-multilingual`
(Apache-2.0), fine-tuned for Chinese customer-service ticket triage.

- Original work: Laya Multilingual — Copyright (c) Convai Innovations — Apache-2.0.
- Modifications: Chinese-domain RLCD fine-tuning, per-(primitive, option-count)
  temperature calibration, ONNX int8 export. Distributed under the same
  Apache-2.0 terms, as required by the original license.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for data-attribution details.

## Repository layout

This repo ships **two deployable artifacts** plus the source checkpoint:

- **Root — PyTorch / `laya.load` checkpoint (F16)**: `model.safetensors`,
  `encoder/config.json`, `tokenizer/`, `rl_agent_config.json`, `questions.json`,
  `calibration.json`. Load with `laya.load("<this repo>")`.
- **`onnx/` — CPU int8 deployment variant** (torch-free): `model.onnx`
  (~698 MB, dynamic int8 per-channel), its own `calibration.json`,
  `questions.json`, `deploy_config.json`, `tokenizer/tokenizer.json`,
  `MANIFEST.json`. Load with `mai_jev.deploy.Predictor("onnx/")`.

> The int8 variant carries its **own** `calibration.json`. Temperature is a
> property of (weights × inference backend); never reuse the torch temperatures
> with the ONNX graph or the calibration silently degrades.

## Training data

| Dataset | Role | License | Attribution |
| --- | --- | --- | --- |
| [MoriyaResearchInstitute/MRI-decision-data-100k](https://huggingface.co/datasets/MoriyaResearchInstitute/MRI-decision-data-100k) | training (format / decision-pattern supervision) | MIT | © Moriya Research Institute |
| [albertkingdom/custom-service-ticket-zh-tw](https://huggingface.co/datasets/albertkingdom/custom-service-ticket-zh-tw) | training (natural Chinese ticket text) | CC BY 4.0 | © albertkingdom |
| [CodyQin/zh-decision-bench](https://huggingface.co/datasets/CodyQin/zh-decision-bench) | **evaluation only, never trained on** | CC BY 4.0 | © CodyQin |

Additional training data was synthesised with open-weight teacher models
(`Qwen/Qwen3-1.7B`, Apache-2.0) from templates over the sources above; the
synthesis prompt templates and generation seeds are in the training repo under
`scripts/`.

**Changes made:** these datasets were re-mapped into the Laya decision schema
(`state` / `questions` / `targets`), filtered, de-duplicated, and mixed with
synthetic samples. Labels were remapped onto mai-jev's 5-team / 1–5-urgency /
refund-or-escalate label space. The original raw data was not modified in place.

The CC BY 4.0 datasets above are used under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0).

## Evaluation (held-out Chinese ticket distribution)

All numbers below come from a held-out test split that was **never used for
training or temperature fitting**. Macro accuracy is the mean over the three
primitives.

| Model | route_team | urgency | noul | macro |
| --- | --- | --- | --- | --- |
| **mai-jev (torch, calibrated)** | 0.9475 | 0.9329 | 0.8513 | **0.9106** |
| mai-jev (int8 ONNX, calibrated) | 0.9096\* | 0.4606\* | 0.8397\* | — |
| `convaiinnovations/laya-multilingual` (base, zero-shot) | — | — | — | 0.4373 |
| Keyword rule baseline | 0.4373 | 0.5918 | 0.4111 | — |

\* The int8 ONNX column is the **uncalibrated** forward pass measured during
quantisation; after re-fitting temperature on the calibration split the int8
variant reproduces the torch argmax agreement ≥ 0.94 on every primitive and its
worst single-primitive ECE is 0.0927. Worst single-primitive ECE for the torch
variant is 0.0871.

### Calibration

Per-primitive temperature is fit on a held-out calibration split and applied at
inference. Expected Calibration Error (ECE, 15 bins) on the held-out test set:

- torch variant: worst single-primitive ECE = **0.0871** (≤ 0.10 target met)
- int8 variant: worst single-primitive ECE = **0.0927** (≤ 0.10 target met)

> **Thin margin.** The calibration target was minimising negative log-likelihood,
> not ECE, so individual primitives can move slightly after calibration. Do not
> treat "ECE ≤ 0.10" as a guaranteed bound across deployments or data shifts.

## Efficiency (int8 ONNX, target machine)

Measured on the release variant (`onnx/`, dynamic int8 per-channel), 4 threads,
batch=1, 1029 real tickets:

- Resident memory: **795.9 MB** (≤ 1 GB target met)
- Latency (one ticket, all three primitives): **p50 = 380 ms**, p95 = 433 ms
  (the acceptance target was restated to "three primitives combined p50 < 500 ms"
  after measurement showed a single-primitive p50 < 50 ms is unreachable at this
  size on 4 CPU threads)

> **Hard constraint:** deploy with `batch=1` and exact-length inference. In
> right-padded mixed batches the short `noul` rows return `NaN` on the fused ONNX
> graph (a known, unfixed upstream interaction). The CLI `mai_jev.deploy` does
> this by default.

## Known limitations & negative results

- **`urgency` is not business-accurate.** Its 0.9329 measures reproduction of a
  *structured urgency proxy* (a graded evidence labeller), not real SLA/priority.
  Balanced accuracy is only 0.8009, and on a 14-ticket manual sample it disagreed
  with the rule baseline on 3. Treat `urgency` as a **reference score**, not an
  auto-decision.
- **`needs_refund_or_escalation` carries a semantic-shift data debt**: the
  training proxy (`needs_human`) over-represents positive cases, so the model
  prior is biased. This is documented as an open issue, not fixed.
- **Latency was measured on an desktop-class CPU**; a real old laptop will be slower.
  Re-measure on the actual target before relying on the p50 figures.
- **`zh-decision-bench` was used for evaluation only** and the model was never
  trained on it.
- This is **System One**, not a chat model: it does not explain or negotiate, it
  emits calibrated probabilities for the three primitives above.

## Usage

### PyTorch / transformers path

```python
import laya
agent = laya.load("Maicarons/mai-jev-zh-support")
result = agent.predict(
    "您好，3 月的账单被重复扣款两次，请今天之内退还多收的费用。",
    agent.questions, max_len=512,
)
print(result["answers"])
```

### CPU int8 ONNX path (torch-free)

```bash
pip install onnxruntime tokenizers numpy
```

```python
from mai_jev.deploy import Predictor
p = Predictor("onnx/")  # the onnx/ subfolder of this repo
print(p.predict("订单 TK-1234 付款后重复扣款，请尽快处理"))
```

Or the CLI (one process, measures resident memory + p50 the way the acceptance
criteria require):

```bash
python -m mai_jev.deploy --model-dir onnx/ --text "退款" --threads 4
```

## Reproducibility

Training, calibration, evaluation, and quantization scripts are in the source
repository `Maicarons/mai-jev` (`scripts/`). Every number in the paper
(`paper/mai-jev.pdf`) is regenerated by those scripts; see
`scripts/make_paper_tables.py` and `scripts/check_numbers.py`.

- Base model: `convaiinnovations/laya-multilingual`
- Fine-tune: RLCD, 2 epochs, multilingual base
- Calibration: per-primitive temperature fit on a held-out split
- Quantization: single-graph whole-model ONNX, dynamic int8 per-channel

---

© 2026 Maicarons. Derived from Laya Multilingual © Convai Innovations, Apache-2.0.
