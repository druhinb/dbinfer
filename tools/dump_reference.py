#!/usr/bin/env python3
"""Dump fixed-seed HF reference tensors for the golden-tensor test layer.

C-contiguous fp32 `.bin` blobs, and emits a manifest the C++ reload test reads.
This captures reference tensors from HF transformers for a fixed model + fixed input + fixed seed,
writes them as raw little-endian Captured tensors (all from layer 0 where a layer index applies):
  embedding, rmsnorm_in/out, rope_q_in/out, rope_k_in/out, attn_head0_out,
  layer0_out, swiglu_in/out, logits.

Outputs (under tests/golden/):
  *.bin, manifest.json, manifest_generated.hpp
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone

import numpy as np

# --- Fixed, recorded inputs -------------------------------------------------

MODEL_ALIASES = {
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
}

DEFAULT_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
SEED = 0
INPUT_IDS = [40, 1079, 264, 4128, 1614, 13]
DEFAULT_GGUF_NAME = "qwen2.5-0.5b-instruct-fp16.gguf"


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="qwen2.5-0.5b", help="model alias")
    ap.add_argument("--revision", default=DEFAULT_REVISION, help="pinned HF revision")
    ap.add_argument(
        "--out-dir",
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "tests",
            "golden",
        ),
        help="where to write goldens (default: tests/golden)",
    )
    ap.add_argument(
        "--gguf",
        default=None,
        help="path to the parity GGUF whose SHA256 is recorded (optional)",
    )
    args = ap.parse_args()

    if args.model not in MODEL_ALIASES:
        print(
            f"error: unknown model alias {args.model!r}; "
            f"known: {sorted(MODEL_ALIASES)}",
            file=sys.stderr,
        )
        return 2
    repo = MODEL_ALIASES[args.model]

    import torch
    import transformers
    from transformers import AutoModelForCausalLM

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"loading {repo} (revision={args.revision}) on CPU/fp32, eager attention ...")
    model = AutoModelForCausalLM.from_pretrained(
        repo,
        revision=args.revision,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    model.eval()

    cfg = model.config
    num_heads = cfg.num_attention_heads
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // num_heads)

    # --- Capture plumbing ---------------------------------------------------
    captured: dict[str, np.ndarray] = {}

    def store(name: str, t: "torch.Tensor") -> None:
        arr = t.detach().to(torch.float32).cpu().numpy()
        if arr.ndim >= 1 and arr.shape[0] == 1:
            arr = arr[0]
        captured[name] = np.ascontiguousarray(arr, dtype="<f4")

    layer0 = model.model.layers[0]
    handles = []

    # embedding lookup
    handles.append(
        model.model.embed_tokens.register_forward_hook(
            lambda mod, inp, out: store("embedding", out)
        )
    )

    # RMSNorm (input_layernorm of layer 0): in + out
    handles.append(
        layer0.input_layernorm.register_forward_pre_hook(
            lambda mod, inp: store("rmsnorm_in", inp[0])
        )
    )
    handles.append(
        layer0.input_layernorm.register_forward_hook(
            lambda mod, inp, out: store("rmsnorm_out", out)
        )
    )

    # Attention head 0 output: input to o_proj is (b, q_len, num_heads*head_dim).
    def o_proj_pre(mod, inp):
        x = inp[0]
        b, q_len, _ = x.shape
        heads = x.view(b, q_len, num_heads, head_dim)
        store("attn_head0_out", heads[:, :, 0, :])

    handles.append(layer0.self_attn.o_proj.register_forward_pre_hook(o_proj_pre))

    # SwiGLU (whole MLP of layer 0): in + out
    handles.append(
        layer0.mlp.register_forward_pre_hook(
            lambda mod, inp: store("swiglu_in", inp[0])
        )
    )
    handles.append(
        layer0.mlp.register_forward_hook(lambda mod, inp, out: store("swiglu_out", out))
    )

    # Full layer-0 output (DecoderLayer returns a tuple; [0] is hidden states).
    def layer0_hook(mod, inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        store("layer0_out", hs)

    handles.append(layer0.register_forward_hook(layer0_hook))

    # RoPE: monkeypatch apply_rotary_pos_emb, capture the first (layer-0) call.
    import transformers.models.qwen2.modeling_qwen2 as qwen2_mod

    orig_rope = qwen2_mod.apply_rotary_pos_emb
    rope_done = {"v": False}

    def patched_rope(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        q_out, k_out = orig_rope(q, k, cos, sin, position_ids, unsqueeze_dim)
        if not rope_done["v"]:
            store("rope_q_in", q)
            store("rope_k_in", k)
            store("rope_q_out", q_out)
            store("rope_k_out", k_out)
            rope_done["v"] = True
        return q_out, k_out

    qwen2_mod.apply_rotary_pos_emb = patched_rope

    # --- Forward pass -------------------------------------------------------
    input_ids = torch.tensor([INPUT_IDS], dtype=torch.long)
    position_ids = torch.arange(len(INPUT_IDS), dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        out = model(input_ids=input_ids, position_ids=position_ids, use_cache=False)
    store("logits", out.logits)

    qwen2_mod.apply_rotary_pos_emb = orig_rope
    for h in handles:
        h.remove()

    order = [
        "embedding",
        "rmsnorm_in",
        "rmsnorm_out",
        "rope_q_in",
        "rope_q_out",
        "rope_k_in",
        "rope_k_out",
        "attn_head0_out",
        "layer0_out",
        "swiglu_in",
        "swiglu_out",
        "logits",
    ]
    missing = [n for n in order if n not in captured]
    if missing:
        print(f"error: did not capture {missing}", file=sys.stderr)
        return 3

    # --- Write + self-check (bit-exact write-side round-trip) ---------------
    manifest_tensors = []
    print("--- writing goldens (self-check on each) ---")
    for name in order:
        arr = captured[name]
        assert arr.dtype == np.dtype("<f4") and arr.flags["C_CONTIGUOUS"]
        rel = f"{name}.bin"
        path = os.path.join(args.out_dir, rel)
        arr.tofile(path)

        in_mem_bytes = arr.tobytes()
        digest = sha256_bytes(in_mem_bytes)

        reloaded = np.fromfile(path, dtype="<f4")
        if not np.array_equal(reloaded, arr.reshape(-1)):
            print(f"FAIL {name}: reload != in-memory", file=sys.stderr)
            return 4
        if sha256_file(path) != digest:
            print(f"FAIL {name}: file sha256 != in-memory sha256", file=sys.stderr)
            return 4

        n = int(arr.size)
        manifest_tensors.append(
            {
                "name": name,
                "file": rel,
                "dtype": "f32",
                "shape": list(arr.shape),
                "n_elements": n,
                "sha256": digest,
            }
        )
        print(f"PASS {name:16s} shape={list(arr.shape)} n={n} sha256={digest[:16]}...")

    gguf_path = args.gguf
    if gguf_path is None:
        cand = os.path.join(args.out_dir, DEFAULT_GGUF_NAME)
        gguf_path = cand if os.path.exists(cand) else None
    gguf_entry = None
    if gguf_path and os.path.exists(gguf_path):
        gguf_entry = {
            "name": os.path.basename(gguf_path),
            "sha256": sha256_file(gguf_path),
        }
        print(f"recorded parity GGUF sha256 for {os.path.basename(gguf_path)}")
    else:
        print(
            "note: parity GGUF not found; recording gguf: null "
            "(pass --gguf to record it)"
        )

    resolved_commit = None
    try:
        from huggingface_hub import HfApi

        resolved_commit = HfApi().model_info(repo, revision=args.revision).sha
    except Exception as e:  # noqa: BLE001 - provenance is best-effort
        print(f"note: could not resolve commit sha ({e})")

    manifest = {
        "provenance": {
            "hf_repo": repo,
            "hf_revision": args.revision,
            "hf_resolved_commit": resolved_commit,
            "seed": SEED,
            "input_ids": INPUT_IDS,
            "positions": list(range(len(INPUT_IDS))),
            "versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "torch": __import__("torch").__version__,
                "transformers": transformers.__version__,
            },
            "generated_utc": datetime.now(timezone.utc).isoformat(),
        },
        "tensors": manifest_tensors,
        "parity_gguf": gguf_entry,
    }

    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    write_generated_header(
        os.path.join(args.out_dir, "manifest_generated.hpp"), manifest_tensors
    )

    print(
        f"--- wrote {len(manifest_tensors)} tensors + manifest.json + "
        f"manifest_generated.hpp to {args.out_dir} ---"
    )
    return 0


def write_generated_header(path: str, tensors: list[dict]) -> None:
    """Emit a constexpr Entry[] so the C++ test needs no JSON parser."""
    lines = [
        "// GENERATED by tools/dump_reference.py pls do not edit by hand.",
        "#ifndef DBINFER_TESTS_GOLDEN_MANIFEST_GENERATED_HPP",
        "#define DBINFER_TESTS_GOLDEN_MANIFEST_GENERATED_HPP",
        "",
        "#include <cstddef>",
        "",
        "namespace dbinfer::golden {",
        "",
        "struct Entry {",
        "  const char *name;",
        "  const char *relpath;",
        "  std::size_t n_elements;",
        "  const char *sha256hex;",
        "};",
        "",
        "inline constexpr Entry kEntries[] = {",
    ]
    for t in tensors:
        lines.append(
            f'    {{"{t["name"]}", "{t["file"]}", {t["n_elements"]}, "{t["sha256"]}"}},'
        )
    lines += [
        "};",
        "",
        "} // namespace dbinfer::golden",
        "",
        "#endif // DBINFER_TESTS_GOLDEN_MANIFEST_GENERATED_HPP",
        "",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
