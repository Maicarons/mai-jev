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
| `needs_refund_or_escalation` (noul) | Does the customer demand — or ask about getting — a refund / payout / escalation? (threats to complain count; pure process questions do not) | true / false |

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
  (~698 MB, `int8-gather`: only the embedding `Gather` is quantised, every `MatMul`
  stays fp32 — whole-graph dynamic int8 perturbs logits by 2.31 and was rejected),
  its own `calibration.json`, `questions.json`, `deploy_config.json`,
  `tokenizer/tokenizer.json`, `MANIFEST.json`. Load with
  `mai_jev.deploy.Predictor("onnx/")`.

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
| mai-jev (shipped int8-gather ONNX, calibrated) | 0.9475 | 0.9359 | 0.8484 | 0.9106 |
| `convaiinnovations/laya-multilingual` (base, zero-shot) | 0.4898 | 0.0845 | 0.6414 | 0.4052 |
| Keyword rule baseline | 0.4373 | 0.5918 | 0.4111 | — |
| Majority class | 0.3586 | 0.7026 | 0.6356 | — |

> **Which labels these numbers use.** The table above describes the **published v0.3**
> model under the **v0.3 label definition** (in particular the `noul` proxy — see the
> `noul` bullet under *Known limitations* below). `route_team` and `urgency` are
> unchanged in v0.4; `noul` was relabelled there, so its 0.8513 / 0.6414 / 0.4111 /
> 0.6356 entries are **not comparable** to any v0.4 `noul` figure.

### v0.4 (in-repo, same test split, new `noul` labels)

| Model | route_team | urgency | noul | macro |
| --- | --- | --- | --- | --- |
| **mai-jev v0.4 (torch, calibrated)** | 0.9300 | 0.9475 | 0.9300 | **0.9359** |
| — same, balanced accuracy | 0.9378 | 0.8555 | **0.8923** | — |
| v0.3 `noul` head, re-scored on the new labels | — | — | 0.4898 (bal. 0.5733) | — |
| Keyword rule baseline (new labels) | 0.4373 | 0.5918 | 0.8484 (bal. 0.6711) | — |
| Majority class (new labels) | 0.3586 | 0.7026 | 0.7959 | — |
| `convaiinnovations/laya-multilingual` base, zero-shot | 0.4898 | 0.0845 | 0.7055 | 0.4266 |

The headline change is that the v0.3 `noul` head, re-scored against the honest
question, **falls below the majority class** (0.4898 against 0.7959): its old 0.8513
was mostly the label definition. The retrained head reaches 0.8923 balanced against
0.6711 for the keyword table — and note that the keyword table is *competitive on raw
accuracy* (0.8484), which is why `noul` is judged on balanced accuracy.

> **v0.4 is in-repo, not published.** Hugging Face / ModelScope still ship the v0.3
> package. The packaging figures below (size, memory, latency, quantisation cost)
> belong to that published package; v0.4 differs from it only in the `noul` head's
> weights, and its ONNX export has an identical graph size.

Quantisation costs at most **0.29 accuracy points** on any primitive (whole 343-row
test split, eager vs. int8-gather graph; measured on the v0.3 package). Worst
single-primitive ECE after re-fitting temperature on each variant's own calibration
split: **0.0871 (torch) / 0.0927 (int8)**.

### Calibration

Per-primitive temperature is fit on a held-out calibration split and applied at
inference. Expected Calibration Error (ECE, 15 bins) on the held-out test set:

- torch variant (v0.3): worst single-primitive ECE = **0.0871** (≤ 0.10 target met)
- int8 variant (v0.3): worst single-primitive ECE = **0.0927** (≤ 0.10 target met)
- torch variant (v0.4): worst single-primitive ECE = **0.1061** — **above the 0.10
  target, reported as a miss.** The routing temperature fitted by likelihood *raises*
  test ECE (0.0466 before fitting → 0.1061 after) while accuracy is unchanged at
  0.9300; the previous checkpoint behaved the same way at nearly the same temperature
  (1.1057 vs 1.1131), so at 343 test rows this line is decided by sampling noise
  rather than by method. We did not pick the temperature by test ECE — that would be
  fitting on the test set. Details: `docs/research/RUNS.md`.

> **Thin margin.** The calibration target was minimising negative log-likelihood,
> not ECE, so individual primitives can move slightly after calibration. Do not
> treat "ECE ≤ 0.10" as a guaranteed bound across deployments or data shifts.

## Efficiency (int8-gather ONNX, target machine)

Measured on the release variant (`onnx/`, `int8-gather`, `max_len=256`), 4 threads,
batch=1, exact-length inference, 150 real tickets, one ticket through all three
primitives per sample (`python -m mai_jev.deploy --input ... --repeats 150`):

- Resident memory: **796 MB** (≤ 1 GB target met)
- Latency: **p50 = 421–499 ms, p95 = 862–976 ms**, mean 530–584 ms (range over two
  runs on the same host)

> **Read this before quoting the latency.** The numbers above are the release
> variant's own end-to-end measurement, and they are the ones to trust. An earlier
> published figure — p50 380 ms — came from `mai-jev-int8-ml96`, whose `max_len=96`
> cannot fit this checkpoint's 109/110-token question heads; that variant silently
> dropped the ticket body and scored 0.187 on `route_team`. It is **not** a valid
> measurement for this model and must not be quoted.
>
> With p50 at 84–100% of the 500 ms budget and p95 at 1.7–2.0× it, the restated
> acceptance line ("three primitives combined p50 < 500 ms") is **met only
> marginally and not robustly**, and it was measured on an desktop-class CPU — a real
> target machine will be slower. The original line (single-primitive p50 < 50 ms)
> is unreachable at this size on 4 CPU threads and is reserved for a future
> ≤500 MB distilled tier.

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
- **`needs_refund_or_escalation` (noul) has a label-provenance caveat** (relabelled in
  v0.4, issue [#17](https://github.com/Maicarons/mai-jev/issues/17)). The published
  v0.3 `noul` head was trained *and evaluated* on an upstream **proxy** label
  (`needs_human`, "should a human step in?"), and 70% of that proxy's positives
  contain no refund/payout/escalation wording at all. So the v0.3 `noul` pair above
  largely measures *"is the customer in a hurry"*, i.e. it is close to a second
  `urgency` head — **0.8513 is a reproduction of a proxy, not a refund-demand
  accuracy**. In v0.4 the label was rebuilt by a deterministic rule labeller
  (`scripts/label_noul.py`, gated by a 37-row human-written antibody set) under a
  ratified definition (explicit demand, escalation threats, and eligibility
  questions all count as true; process/rule questions and pure emotion do not).
  Positives drop from ~65% to **~20%**, which flips the majority class: the
  always-`false` baseline is **0.796**, so raw accuracy is meaningless for `noul` and
  **balanced accuracy** is reported instead. Consequences to state whenever `noul`
  numbers are quoted: (a) v0.3 vs v0.4 `noul` figures are **not comparable**, never
  present them as a delta; (b) after relabelling the **keyword rule baseline scores
  0.8484 (balanced 0.6711)** on the same test split — the model must beat that to be
  worth anything. Metrics: `docs/research/RUNS.md`. Definition: `CONTEXT.md`.
  v0.4 is **in-repo, not published**; the Hugging Face / ModelScope artifacts still
  ship v0.3.
- **The re-graded `noul` label is reproduced, not yet learned.** On a 37-row
  hand-written counterexample slice (present in **no** split, built so that the cue
  words and the correct verdict deliberately disagree) the v0.4 head scores
  **0.4865 accuracy / 0.4583 balanced** — *below* always answering `false`
  (0.5676 / 0.5000). The keyword rules do worse (0.3243 / 0.3080), so the head is not
  merely a keyword table, but with 2,581 training rows it reproduces the new label
  distribution without acquiring the boundary cases that define it. Until those rows
  are trained on, or a human-audited `noul` set exists, treat this primitive as
  **review-only**. Reproduce: `scripts/eval_noul_antibody.py`.
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
`scripts/make_paper_tables.py` (which fails rather than emit a stale value when an
input artefact is missing), `scripts/label_noul.py` (gated by the counterexample set)
and `scripts/eval_noul_antibody.py`.

- Base model: `convaiinnovations/laya-multilingual`
- Fine-tune: RLCD, 2 epochs, multilingual base
- Calibration: per-primitive temperature fit on a held-out split
- Quantization: single-graph whole-model ONNX, dynamic int8 per-channel

---

© 2026 Maicarons. Derived from Laya Multilingual © Convai Innovations, Apache-2.0.
