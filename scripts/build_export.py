r"""把「一个量化后的 ONNX 图 + 一份标定」组装成一个自洽的发布目录（ticket #9）。

    python scripts/build_export.py --fp32 <laya_fp32.onnx> --model-dir <checkpoint> ^
        --calibration <calibration.json> --out export\mai-jev-int8-ml96 ^
        --variant int8 --max-len 96

产物目录（`mai_jev.deploy.Predictor` 认的形状，也是 #5 要发到 HF/ModelScope 的形状）：

    mai-jev-int8-ml96/
        model.onnx              int8 动态 per-channel 单图
        calibration.json        该变体专属温度（laya.calibration_payload 形状）
        questions.json          三原语定义（训练时冻结的那一份）
        deploy_config.json      max_len / head_max_len / 变体名 / 口径声明
        tokenizer/tokenizer.json XLM-R BPE（部署侧只需这一个文件）
        MANIFEST.json           逐文件字节数 + sha256 + 来源

为什么不直接发 checkpoints 目录：部署方要装的是 torch-free 的三件套（onnxruntime +
tokenizers + numpy），把 tokenizer 和标定随图放在一起，才能保证「换个量化变体就必须
换标定」这条纪律在文件系统层面成立——温度是「模型 + 推理后端」的耦合属性，
跨变体复用标定会让 ECE 悄悄回到未标定水平。

参数：
    --variant   int8 | fp32 | int8-gather  决定主图，也写进 deploy_config.json
    --max-len   该变体的序列预算（默认沿用 checkpoint 的 rl_agent_config.json）

变体实测（40 条真实工单，对 fp32 图的 raw logit）：

    int8（全图动态 QInt8）   max|dlogit| 2.31（logit 量级 3.3），urgency argmax 只有 0.417
    int8 + reduce_range      同量级，没有改善
    int8 per_tensor          同量级
    int8-gather（只量化 Gather，即 embedding/逐元素查表；MatMul 全留 fp32）
                             666 MB，max|dlogit| 0.105，三原语 argmax 全 1.000

所以发布用 `int8-gather`：动态激活量化在这个图上必须整条 MatMul 一起做，而 encoder 每层
的权重/激活动态范围太宽，int8 逐通道权重压不住，误差直接淹掉 logits 的区分度。只量化
Gather 保留了 embedding 查表这一块（占模型体积的一半）而 MatMul 不动，体积 1229→666 MB。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

# 部署侧的依赖下限。onnxruntime 只在 CPU 上跑，所以 CPU 目标机不需要 CUDA。
DEPLOY_REQUIRES = {"onnxruntime": ">=1.17", "tokenizers": ">=0.15", "numpy": ">=1.24"}

# 工单正文至少要留下这么多 token 才算可用输入（见 mai_jev.deploy.MIN_STATE_ROOM）。
MIN_STATE_ROOM = 32

# 随变体发布的口径声明。放在模块级是为了让「修正已发布变体的声明」这类维护动作
# 能直接 import 这几个字符串，而不是手抄一份到别处再慢慢漂移。
BATCH_NOTE = ("batch=1 且按真实长度推理。混合长度的右 padding batch 会让短的 "
              "qtype=2 行返回 NaN（#8 报告 §1.5 实测 340/1029 行），未修复。")
TEMPERATURE_NOTE = "温度随量化变体绑定，换变体必须重新标定；不要跨变体复用 calibration.json。"
PARITY_NOTE = ("本目录的 calibration.json 是在本变体自己的图上、用它自己的 calib 切分重拟合的"
               "（scripts/calibrate_export.py）。int8-gather 与 fp32 图几乎一致（max|dlogit| 0.105、"
               "三原语 argmax 一致 1.000），因此重拟合出的温度与 torch 侧只差 ~1e-3 —— "
               "那是测出来的巧合，不是可以省掉逐变体标定的理由：整图 int8（max|dlogit| 2.31）"
               "不满足这条，会真退化，已弃用。")


def deploy_head_lengths(model_dir: Path, head_max_len: int) -> Dict[str, int]:
    """用与部署侧完全相同的代码量出每个 question head 的 token 数。

    必须走 mai_jev.deploy 而不是 laya：这里量的是「发布时真正会拼进图里的长度」，
    两条路径哪怕差一个 token 都会让 state 预算算错。
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from mai_jev.deploy import SPECIAL_TOKENS, _Head  # noqa: E402
    from tokenizers import Tokenizer  # noqa: E402

    tok = Tokenizer.from_file(str(model_dir / "tokenizer" / "tokenizer.json"))
    tok.no_truncation()
    tok.no_padding()
    ids = {t: tok.token_to_id(t) for t in SPECIAL_TOKENS}
    missing = [t for t, i in ids.items() if i is None or i < 0]
    if missing:
        raise SystemExit("tokenizer.json 缺少 special token: %s" % missing)
    questions = json.loads((model_dir / "questions.json").read_text(encoding="utf-8"))
    return {
        qid: len(_Head(tok, q, head_max_len, ids["mask"], ids["cls"], ids["sep"]).ids)
        for qid, q in questions.items()
    }


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def quantize(fp32: Path, out: Path, per_channel: bool = True,
             op_types: Optional[Sequence[str]] = None) -> Path:
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from onnxruntime.quantization.shape_inference import quant_pre_process

    # ORT 的 symbolic Reshape handler 在 torch 2.6 出的动态 shape 图上会 assert，
    # 所以量化前必须过一遍 quant_pre_process(skip_symbolic_shape=True)。
    prep = fp32.with_name(fp32.stem + "_prep.onnx")
    if not prep.exists():
        quant_pre_process(str(fp32), str(prep), skip_symbolic_shape=True)
    quantize_dynamic(str(prep), str(out), weight_type=QuantType.QInt8,
                     per_channel=per_channel, op_types_to_quantize=list(op_types) if op_types else None)
    return out


# variant -> 只量化哪些算子。None = 整图动态量化。
VARIANT_OPS: Dict[str, Optional[Sequence[str]]] = {
    "int8": None,
    "int8-gather": ["Gather"],
}


def load_cfg(model_dir: Path) -> dict:
    cfg = json.loads((model_dir / "rl_agent_config.json").read_text(encoding="utf-8"))
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="训练产出目录（含 rl_agent_config.json 与 encoder/）")
    ap.add_argument("--fp32", required=True, help="fp32 ONNX（由 scripts/export_onnx.py 生成）")
    ap.add_argument("--calibration", required=True, help="scripts/calibrate_eval.py 写出的 calibration.json")
    ap.add_argument("--out", required=True, help="组装目录")
    ap.add_argument("--variant", default="int8-gather",
                    choices=["int8", "fp32", "int8-gather"])
    ap.add_argument("--max-len", type=int, help="覆盖 checkpoint 的 max_len")
    ap.add_argument("--head-max-len", type=int)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    (out / "tokenizer").mkdir(parents=True, exist_ok=True)

    fp32 = Path(args.fp32)
    if args.variant == "fp32":
        main_graph = fp32
    else:
        main_graph = quantize(fp32, fp32.with_name(fp32.stem + f"_{args.variant.replace('-', '_')}.onnx"),
                            op_types=VARIANT_OPS[args.variant])
    shutil.copyfile(main_graph, out / "model.onnx")

    # tokenizer：只拷 tokenizer.json。部署侧不需要 vocab.txt / merges.txt / sentencepiece。
    tok_dir = model_dir / "tokenizer"
    src_tok = tok_dir / "tokenizer.json"
    if not src_tok.exists():
        raise SystemExit(f"missing {src_tok}; the checkpoint has no tokenizer/")
    shutil.copyfile(src_tok, out / "tokenizer" / "tokenizer.json")

    shutil.copyfile(model_dir / "questions.json", out / "questions.json")
    shutil.copyfile(args.calibration, out / "calibration.json")

    cfg = load_cfg(model_dir)
    max_len = args.max_len or int(cfg.get("max_len", 512))
    head_max_len = args.head_max_len or int(cfg.get("head_max_len", 64))

    # 硬闸门：question head 比 max_len 还长时，整条 state 会被静默丢弃
    # （Predictor._feed 的 room = max_len - len(head.ids) - 1 变成负数），
    # 模型退化成「不看工单、只看问句」——v03 在 max_len=96 上就是这样
    # route_team 0.945→0.187 的。不要用「能跑就行」糊过去。
    head_lens = deploy_head_lengths(model_dir, head_max_len)
    longest = max(head_lens.values())
    if longest + 1 > max_len:
        raise SystemExit(
            "max_len=%d 装不下最长的 question head（%d tokens，来自 %s）："
            "state 预算为 %d，整条工单会被丢弃，模型只看问句。"
            "请把 --max-len 提到 %d 以上。"
            % (max_len, longest, max(head_lens, key=lambda k: head_lens[k]),
               max(0, max_len - longest - 1), longest + 1)
        )
    deploy_config = {
        "variant": args.variant,
        "max_len": max_len,
        "head_max_len": head_max_len,
        "threads_default": args.threads,
        "requires": DEPLOY_REQUIRES,
        "batch": 1,
        "batch_note": BATCH_NOTE,
        "temperature_note": TEMPERATURE_NOTE,
        "parity_note": PARITY_NOTE,
    }
    (out / "deploy_config.json").write_text(
        json.dumps(deploy_config, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "format": "mai-jev-export/1",
        "source_checkpoint": str(model_dir),
        "source_fp32_onnx": str(fp32),
        "variant": args.variant,
        "files": {},
    }
    for rel in ["model.onnx", "calibration.json", "questions.json", "deploy_config.json",
                os.path.join("tokenizer", "tokenizer.json")]:
        p = out / rel
        manifest["files"][rel.replace("\\", "/")] = {"bytes": p.stat().st_size, "sha256": sha256(p)}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    total = sum(f["bytes"] for f in manifest["files"].values())
    print("export dir:", out)
    for k, v in manifest["files"].items():
        print("  %-28s %8.1f MB  %s" % (k, v["bytes"] / 1024 / 1024, v["sha256"][:16]))
    print("  %-28s %8.1f MB" % ("TOTAL", total / 1024 / 1024))
    return 0


if __name__ == "__main__":
    sys.exit(main())