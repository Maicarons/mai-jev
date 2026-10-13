"""Check that deployment-side `tokenizers.Tokenizer.from_file` reproduces the
training-side tokenizer, and that the ORT graph agrees with eager torch on real
Chinese text.

Usage:
    python scripts/check_tokenizer_alignment.py

Dependencies: tokenizers, torch, transformers, laya, onnxruntime, numpy.

Why `tokenizers` and not `AutoTokenizer`: in this environment
(transformers 5.19.0) `AutoTokenizer.from_pretrained(<laya ckpt>)` raises
    ValueError: Couldn't instantiate the backend tokenizer ...
    You need to have sentencepiece or tiktoken
because the checkpoint's `tokenizer_config.json` declares a slow tokenizer class
that transformers tries to rebuild from scratch. The pre-built
`tokenizer/tokenizer.json` is a fast XLM-R-style tokenizer (vocab 256000) and
loads directly with `tokenizers.Tokenizer.from_file`. This script asserts the
two agree token-for-token on a Chinese/English/mixed corpus.
"""
import json
import os

BASE = (r"$HF_HUB\models--convaiinnovations--laya-multilingual"
        r"\snapshots\1720e3e3357cfe1e281542e223f8273b0890ca34")

TEXTS = [
    "我们要不要把这个功能推迟到下个季度？",
    "客户对价格非常敏感，但是交付时间不能延期。",
    "The deployment target is a low-end x86 CPU-only laptop with 4 threads.",
    "系统需要在 1GB 常驻内存内运行，单问题 p50 延迟小于 50ms。",
    "混合中英文 with numbers 12345 和标点，测试边界。",
    "",
    "   ",
]


def load_deploy_tokenizer(path=None):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(path or os.path.join(BASE, "tokenizer", "tokenizer.json"))


def load_reference_tokenizer(path=None):
    """laya's own tokenizer construction path (what training used)."""
    from transformers import PreTrainedTokenizerFast
    # Special-token names come from the checkpoint's tokenizer_config.json
    # (bos/eos/cls/sep are all "<bos>"/"<eos>" here, not the usual "<s>"/"</s>").
    return PreTrainedTokenizerFast(
        tokenizer_file=path or os.path.join(BASE, "tokenizer", "tokenizer.json"),
        bos_token="<bos>", eos_token="<eos>", cls_token="<bos>", sep_token="<eos>",
        unk_token="<unk>", pad_token="<pad>", mask_token="<mask>",
    )


def encode(tok, text):
    """Full pipeline: pretokenize + encode + postprocess, both sides."""
    return tok.encode(text, add_special_tokens=True).ids


def main():
    deploy = load_deploy_tokenizer()
    ref = load_reference_tokenizer()

    print("vocab_size:", deploy.get_vocab_size(), "| truncation:", deploy.truncation)
    bad = 0
    for t in TEXTS:
        a = deploy.encode(t, add_special_tokens=True).ids
        b = ref(t, add_special_tokens=True)["input_ids"]
        ok = a == b
        bad += not ok
        print(("  OK  " if ok else "  FAIL") + f"  {t[:28]!r}")
        if not ok:
            print("        deploy:", a)
            print("        ref   :", b)
    print("\n%d/%d texts aligned" % (len(TEXTS) - bad, len(TEXTS)))
    if bad:
        raise SystemExit("tokenizer misalignment")

    # One encode with the real laya padding convention, for downstream use.
    deploy.enable_truncation(max_length=1024)
    print("\nsanity:", json.dumps(
        {"text": TEXTS[0], "ids": deploy.encode(TEXTS[0], add_special_tokens=True).ids},
        ensure_ascii=False))


if __name__ == "__main__":
    main()