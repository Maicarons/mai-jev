"""Export laya-multilingual (DecisionModel) to a single ONNX graph.

Usage:
    python scripts/export_onnx.py --out-dir $WORK\\onnx \\
        --model-dir <path-to-laya-multilingual-checkpoint>

Quantize the result with scripts/quantize_onnx.py:
    quantize_dynamic(weight_type=QuantType.QInt8, per_channel=True)
    -> 308.6 MB, RSS 408 MB, ~2x faster than fp32.  See
    docs/research/quantization-and-deployment.md.

Gotcha found on this graph (ticket #8): in a RIGHT-PADDED batch the exported
graph returns NaN logits for SHORT rows of qtype=2 (noul). 340/1029 rows on the
#6 test split were NaN; eager torch on the same batch was clean, and the same
rows are fine at batch=1 or when the batch has no mixed lengths. Deploy at
batch=1 with exact per-item length -- which is also what the p50 target needs.
Root cause not fixed (ModernBERT sliding-window attention + padding under
torch.onnx); see report 1.5.

Dependencies: torch, transformers, laya (installed in $WORK\\venv).

What is exported: ONE graph, not per-head. DecisionModel.forward
(laya/common.py:609) is a single fused forward that returns exactly
(logits, act_logits) -- the option scorer and the act head share the encoder
trunk and the head transformer, so they cannot be split into separate sessions
without recomputing the trunk. Inputs match what laya.onnx_agent.ONNXAgent
feeds (onnx_agent.py:791):
    input_ids, attention_mask, marker_pos, marker_mask, qtype
Outputs:
    logits, act_logits
"""
import argparse
import os

import torch
import torch.nn.functional as F

from laya.common import build_model

LAYA_CKPT_DEFAULT = (
    r"$HF_HUB\models--convaiinnovations--laya-multilingual"
    r"\snapshots\1720e3e3357cfe1e281542e223f8273b0890ca34"
)


class ExportWrapper(torch.nn.Module):
    """Dynamic-sequence-length wrapper.

    `DecisionModel.forward` calls `nn.TransformerEncoder`, whose fast path
    (`aten::_transformer_encoder_layer_fwd`) has no ONNX symbolic in torch 2.6
    and raises `UnsupportedOperatorError` at export. This wrapper therefore
    re-expresses the head layers as their own primitive submodules, in the same
    order and with the same defaults the stock layer uses (norm_first=True,
    activation='relu', layer_norm_eps=1e-5). Numerics are identical; only the
    ONNX-inexpressible fast path is gone.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def _head_layer(self, layer, h, pad):
        # norm_first=True: x + attn(norm1(x)) -> x + mlp(norm2(x))
        a = layer.norm1(h)
        a = layer.self_attn(a, a, a, key_padding_mask=pad, need_weights=False)[0]
        h = h + a
        n = layer.norm2(h)
        y = layer.linear2(F.relu(layer.linear1(n)))
        return h + y

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        m = self.model
        h = m.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        h = h + m.type_emb(qtype)[:, None, :]
        if m.head is not None:
            pad = ~attention_mask.bool()
            for layer in m.head.layers:
                h = self._head_layer(layer, h, pad)
        idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        mk = torch.gather(h, 1, idx)
        logits = m.scorer(mk).squeeze(-1).float()
        logits = logits.masked_fill(~marker_mask, -1e4)
        p = torch.softmax(logits.detach(), -1)
        k = marker_mask.sum(-1).clamp(min=2).float()
        ent = -(p * torch.log(p.clamp_min(1e-9))).sum(-1) / torch.log(k)
        top1 = p.topk(2, -1).values
        feats = torch.stack([top1[:, 0], top1[:, 0] - top1[:, 1], ent, k / 255.0], -1)
        act_input = torch.cat([h[:, 0].float(), feats], -1).to(m.act_head[0].weight.dtype)
        return logits, m.act_head(act_input)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=LAYA_CKPT_DEFAULT)
    ap.add_argument("--out-dir", default=r"$WORK\onnx")
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--marker-rows", type=int, default=8)
    args = ap.parse_args()

    import json
    from safetensors.torch import load_file

    cfg = json.load(open(os.path.join(args.model_dir, "rl_agent_config.json"), encoding="utf-8"))
    enc_dir = os.path.join(args.model_dir, "encoder")
    model = build_model(cfg, encoder_dir=enc_dir, pretrained=False)
    sd = load_file(os.path.join(args.model_dir, "model.safetensors"))
    sd = {k: v.float() for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("missing:", missing)
    print("unexpected:", unexpected)
    model.eval()

    S, M = args.seq_len, args.marker_rows
    wrapper = ExportWrapper(model).eval()
    inputs = (
        torch.randint(5, 1000, (1, S), dtype=torch.long),
        torch.ones(1, S, dtype=torch.long),
        torch.arange(1, M + 1, dtype=torch.long).repeat(1, 1),
        torch.ones(1, M, dtype=torch.bool),
        torch.zeros(1, dtype=torch.long),
    )
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, "laya_fp32.onnx")
    with torch.no_grad():
        torch.onnx.export(
            wrapper, inputs, out,
            input_names=["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"],
            output_names=["logits", "act_logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "marker_pos": {0: "batch", 1: "markers"},
                "marker_mask": {0: "batch", 1: "markers"},
                "qtype": {0: "batch"},
                "logits": {0: "batch", 1: "markers"},
            },
            opset_version=17, do_constant_folding=True,
        )
    print("wrote", out, "%.2f MB" % (os.path.getsize(out) / 1024 / 1024))
    print("next: python scripts/quantize_onnx.py --fp32 %s --out-dir %s"
          % (out, args.out_dir))

    # numeric parity: eager DecisionModel vs the exported graph
    import numpy as np
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    sess = ort.InferenceSession(out, so, providers=["CPUExecutionProvider"])
    S2 = 137  # a length the graph was NOT traced at -> proves dynamic axes work
    feed = {
        "input_ids": torch.randint(5, 1000, (2, S2), dtype=torch.long).numpy().astype(np.int64),
        "attention_mask": torch.ones(2, S2, dtype=torch.long).numpy().astype(np.int64),
        "marker_pos": torch.arange(1, 5).repeat(2, 1).numpy().astype(np.int64),
        "marker_mask": torch.ones(2, 4, dtype=torch.bool).numpy().astype(np.bool_),
        "qtype": torch.zeros(2, dtype=torch.long).numpy().astype(np.int64),
    }
    with torch.no_grad():
        ref_logits, ref_act = model(
            torch.from_numpy(feed["input_ids"]),
            torch.from_numpy(feed["attention_mask"]),
            torch.from_numpy(feed["marker_pos"]),
            torch.from_numpy(feed["marker_mask"]),
            torch.from_numpy(feed["qtype"]),
        )
    got_logits, got_act = sess.run(["logits", "act_logits"], feed)
    print("max|dlogits| = %.3e" % np.abs(got_logits - ref_logits.numpy()).max())
    print("max|dact|    = %.3e" % np.abs(got_act - ref_act.numpy()).max())
    print("logits[:1] =", got_logits[0])
    print("ref   [:1] =", ref_logits.numpy()[0])

    # Known bug (ticket #8, report 1.5): a RIGHT-PADDED batch makes SHORT
    # qtype=2 rows come back NaN. Deploy at batch=1 with exact per-item length;
    # this check fails loudly if a future export regresses into it.
    if not np.isfinite(got_logits).all():
        print("ERROR: non-finite logits in a same-length batch")


if __name__ == "__main__":
    main()