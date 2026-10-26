"""SraVaani-1.0: 16 kHz mono 16-bit WAV in, text out. No language argument, no
decoder switch. Latency is returned, never a metric - evaluate.py derives those.

    python -I src/speechSys/asr.py     # self-check the resampler, no model load
"""
import time
from pathlib import Path

TARGET_SR = 16000


def load(repo, device, token=None):
    from transformers import AutoModel
    return AutoModel.from_pretrained(repo, trust_remote_code=True, token=token).to(device).eval()


def duration(path):
    """True duration of the clip as given, before any resampling."""
    import soundfile as sf
    return sf.info(str(path)).duration


def as_16k_mono(src, out):
    """Path to a 16 kHz mono PCM_16 copy at `out`; `src` itself if already conformant.

    Applied identically to every acoustic condition, so the comparison stays controlled.
    """
    import soundfile as sf
    src, out = Path(src), Path(out)
    info = sf.info(str(src))
    if info.samplerate == TARGET_SR and info.channels == 1 and info.subtype == "PCM_16":
        return src
    if out.exists() and out.stat().st_mtime >= src.stat().st_mtime:
        return out

    import torch, torchaudio
    x, sr = sf.read(str(src), dtype="float32", always_2d=True)
    y = torch.from_numpy(x.mean(axis=1))
    if sr != TARGET_SR:
        y = torchaudio.functional.resample(y, sr, TARGET_SR)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), y.numpy(), TARGET_SR, subtype="PCM_16")
    return out


def transcribe(model, wav):
    hyps = model.transcribe(str(wav), return_hypotheses=True)
    h = hyps[0]
    return getattr(h, "text", h)


def transcribe_timed(model, wav):
    t = time.perf_counter()
    text = transcribe(model, wav)
    return text, time.perf_counter() - t


def _self_check():
    import tempfile
    import numpy as np
    import soundfile as sf

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        # 24 kHz stereo, like the bench clips -> must come back 16 kHz mono PCM_16
        n, sr = 24000, 24000
        t = np.arange(n) / sr
        stereo = np.stack([np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 440 * t)], 1)
        src = d / "in.wav"
        sf.write(str(src), stereo.astype("float32"), sr, subtype="FLOAT")

        out = as_16k_mono(src, d / "sub" / "out.wav")
        i = sf.info(str(out))
        assert out != src and i.samplerate == 16000 and i.channels == 1, i
        assert i.subtype == "PCM_16", i.subtype
        assert abs(i.frames - 16000) <= 2, i.frames
        assert abs(duration(out) - duration(src)) < 0.01

        # reused when newer than the source, rewritten when the source changes
        mt = out.stat().st_mtime
        assert as_16k_mono(src, out) == out and out.stat().st_mtime == mt
        import os
        os.utime(src, (mt + 10, mt + 10))
        as_16k_mono(src, out)
        assert out.stat().st_mtime > mt

        # already-conformant input is passed through untouched
        ok = d / "ok.wav"
        sf.write(str(ok), np.zeros(16000, "float32"), 16000, subtype="PCM_16")
        assert as_16k_mono(ok, d / "never.wav") == ok
    print("asr self-check ok")


if __name__ == "__main__":
    _self_check()
