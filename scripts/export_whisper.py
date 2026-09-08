#!/usr/bin/env python3
# 导出 Whisper ONNX、mel 滤波和词表。

from __future__ import annotations

import argparse
import os
import struct
from pathlib import Path

def export_onnx(model_id: str, out_dir: Path):
    from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
    from transformers import AutoProcessor

    print(f"exporting {model_id} -> ONNX (this downloads ~150 MB)")
    m = ORTModelForSpeechSeq2Seq.from_pretrained(model_id, export=True)
    m.save_pretrained(out_dir)
    AutoProcessor.from_pretrained(model_id).save_pretrained(out_dir)
    print(f"  wrote {out_dir}")

def export_mel_filters(model_id: str, out: Path):
    import numpy as np
    from transformers import WhisperFeatureExtractor

    fe = WhisperFeatureExtractor.from_pretrained(model_id)
    filters = np.asarray(fe.mel_filters, dtype=np.float32)

    if filters.shape[0] != 80:
        filters = filters.T
    assert filters.shape == (80, 201), f"unexpected mel filter shape {filters.shape}"

    with open(out, "wb") as f:
        f.write(struct.pack("<II", filters.shape[0], filters.shape[1]))
        f.write(filters.astype("<f4").tobytes())
    print(f"  wrote {out}  ({filters.shape[0]}x{filters.shape[1]} float32)")

def export_vocab(model_id: str, out: Path):
    from transformers import WhisperTokenizer

    tok = WhisperTokenizer.from_pretrained(model_id)
    size = len(tok)

    from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
    byte_decoder = {ch: b for b, ch in bytes_to_unicode().items()}

    entries = []
    n_special = 0
    for i in range(size):
        tokens = tok.convert_ids_to_tokens([i])
        s = tokens[0] if tokens else ""
        if s is None:
            entries.append(b"")
            continue
        if s.startswith("<|") and s.endswith("|>"):
            entries.append(b"")
            n_special += 1
            continue
        try:
            entries.append(bytes(byte_decoder[c] for c in s))
        except KeyError:
            entries.append(s.encode("utf-8"))

    with open(out, "wb") as f:
        f.write(struct.pack("<I", len(entries)))
        for e in entries:
            f.write(struct.pack("<I", len(e)))
            f.write(e)
    print(f"  wrote {out}  ({len(entries)} tokens, {n_special} special)")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="openai/whisper-tiny")
    p.add_argument("--out-dir", default="models/whisper_tiny_onnx")
    p.add_argument("--skip-onnx", action="store_true",
                   help="only regenerate mel_filters.bin and vocab.bin")
    args = p.parse_args()

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not args.skip_onnx:
        export_onnx(args.model, out)
    export_mel_filters(args.model, out / "mel_filters.bin")
    export_vocab(args.model, out / "vocab.bin")

if __name__ == "__main__":
    main()
