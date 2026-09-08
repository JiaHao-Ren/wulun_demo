#!/usr/bin/env python3
# 导出 Qwen 词表和 merge 表给 C++ 用。

from __future__ import annotations

import argparse
import os
import struct
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="onnx-community/Qwen2.5-0.5B-Instruct")
    p.add_argument("--local-dir", default="models/qwen05b_onnx")
    args = p.parse_args()

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from transformers import AutoTokenizer
    from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

    out_dir = Path(args.local_dir)
    tok = AutoTokenizer.from_pretrained(str(out_dir) if (out_dir / "tokenizer.json").exists()
                                        else args.model)

    byte_decoder = {ch: b for b, ch in bytes_to_unicode().items()}

    def tok_to_bytes(s: str) -> bytes:
        try:
            return bytes(byte_decoder[c] for c in s)
        except KeyError:
            return s.encode("utf-8")

    vocab = tok.get_vocab()
    size = max(vocab.values()) + 1
    table = [b""] * size
    for s, i in vocab.items():
        table[i] = tok_to_bytes(s)

    n_special = 0
    for s in tok.all_special_tokens + ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]:
        i = vocab.get(s)
        if i is not None and table[i] != b"":
            table[i] = b""
            n_special += 1

    with open(out_dir / "qwen_vocab.bin", "wb") as f:
        f.write(struct.pack("<I", len(table)))
        for e in table:
            f.write(struct.pack("<I", len(e)))
            f.write(e)
    print(f"qwen_vocab.bin  : {len(table)} entries, {n_special} specials blanked")

    import json
    tj = json.loads((out_dir / "tokenizer.json").read_text(encoding="utf-8"))
    merges = tj["model"]["merges"]

    with open(out_dir / "qwen_merges.bin", "wb") as f:
        f.write(struct.pack("<I", len(merges)))
        for m in merges:
            a, b = (m[0], m[1]) if isinstance(m, list) else m.split(" ", 1)
            ab, bb = tok_to_bytes(a), tok_to_bytes(b)
            f.write(struct.pack("<I", len(ab))); f.write(ab)
            f.write(struct.pack("<I", len(bb))); f.write(bb)
    print(f"qwen_merges.bin : {len(merges)} merge rules")

    probe = "你好,今天天气怎么样?"
    ids = tok(probe, add_special_tokens=False).input_ids
    with open(out_dir / "qwen_probe.txt", "w", encoding="utf-8") as f:
        f.write(probe + "\n")
        f.write(" ".join(str(i) for i in ids) + "\n")
    print(f"qwen_probe.txt  : {probe!r} -> {ids}")

    specials = {}
    for s in ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]:
        i = vocab.get(s)
        if i is not None:
            specials[s] = i
    with open(out_dir / "qwen_special_tokens.txt", "w", encoding="utf-8") as f:
        for k, v in specials.items():
            f.write(f"{k} {v}\n")
    print(f"qwen_special_tokens.txt : {specials}")

    msgs = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "USR"}]
    rendered = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    (out_dir / "qwen_chat_template.txt").write_text(rendered, encoding="utf-8")
    print("qwen_chat_template.txt written:")
    print("  " + rendered.replace("\n", "\\n"))

if __name__ == "__main__":
    main()
