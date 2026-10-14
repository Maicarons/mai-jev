# Third-party notices

mai-jev is a modified version of
[`convaiinnovations/laya-multilingual`](https://huggingface.co/convaiinnovations/laya-multilingual)
(Apache-2.0, Copyright (c) Convai Innovations). The base weights, tokenizer, and
decision-head configuration are redistributed under the same Apache-2.0 license.

## Training data

| Dataset | Role | License | Attribution |
| --- | --- | --- | --- |
| [MoriyaResearchInstitute/MRI-decision-data-100k](https://huggingface.co/datasets/MoriyaResearchInstitute/MRI-decision-data-100k) | training (format / decision-pattern supervision) | MIT | © Moriya Research Institute |
| [albertkingdom/custom-service-ticket-zh-tw](https://huggingface.co/datasets/albertkingdom/custom-service-ticket-zh-tw) | training (natural Chinese ticket text) | CC BY 4.0 | © albertkingdom |
| [CodyQin/zh-decision-bench](https://huggingface.co/datasets/CodyQin/zh-decision-bench) | **evaluation only, never trained on** | CC BY 4.0 | © CodyQin |

Additional training data was synthesised with open-weight teacher models
(`Qwen/Qwen3-1.7B`, Apache-2.0) from templates over the sources above; the
synthesis prompt templates and generation seeds are in the repository under
`scripts/`.

**Changes made:** these datasets were re-mapped into the Laya decision schema
(`state` / `questions` / `targets`), filtered, de-duplicated, and mixed with
synthetic samples. Labels were remapped onto mai-jev's 5-team / 1–5-urgency /
refund-or-escalate label space. The original raw data was not modified in place.

The CC BY 4.0 datasets above are used under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0).
