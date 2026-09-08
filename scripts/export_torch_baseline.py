#!/usr/bin/env python3
# 导出 Whisper encoder，给 Python 对比用。

from __future__ import annotations

import argparse
import os
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="whisper_tiny_encoder_ts.pt")
    p.add_argument("--model", default="openai/whisper-tiny")
    args = p.parse_args()

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    import torch
    from transformers import WhisperForConditionalGeneration

    full = WhisperForConditionalGeneration.from_pretrained(args.model)
    encoder = full.model.encoder.eval()

    class EncoderWrapper(torch.nn.Module):
        def __init__(self, enc):
            super().__init__()
            self.enc = enc

        def forward(self, input_features):
            return self.enc(input_features).last_hidden_state

    wrapper = EncoderWrapper(encoder).eval()

    example = torch.zeros(1, 80, 3000, dtype=torch.float32)
    with torch.inference_mode():
        traced = torch.jit.trace(wrapper, example, strict=False)
        traced = torch.jit.freeze(traced)
        ref = wrapper(example)
        got = traced(example)
        assert torch.allclose(ref, got, atol=1e-4), "traced module diverged from eager"

    out = Path(args.out)
    traced.save(str(out))
    print(f"saved {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    print(f"output shape: {tuple(got.shape)}")

if __name__ == "__main__":
    main()
