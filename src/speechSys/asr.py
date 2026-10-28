"""SraVaani-1.0: 16 kHz mono 16-bit WAV in, text out. No language argument, no
decoder switch. Latency is returned, never a metric - evaluate.py derives those.

A model name starting with "saaras" selects the hosted Sarvam API instead, so the
backend swap is config-only (params.yaml, or --asr-model).

    python -I src/speechSys/asr.py     # self-check the resampler, no model load
"""
import time
from pathlib import Path

TARGET_SR = 16000


def is_api(model):
    """Hosted backends need no local load, no VRAM and no 16 kHz copy."""
    return str(model).startswith("saaras")


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


# --- Saaras API backend -------------------------------------------------------
# The clip is sent as-is: Saaras resamples server side, so the 16 kHz step the local
# path needs does not apply and must not be introduced here.
SAARAS_URL = "https://api.sarvam.ai/speech-to-text"
CRLF = "\r\n"


def _multipart(fields, filename, blob):
    """(boundary, body) for one multipart/form-data file upload."""
    import uuid
    b = "----" + uuid.uuid4().hex
    out = b""
    for k, v in fields.items():
        out += (f"--{b}{CRLF}Content-Disposition: form-data; "
                f'name="{k}"{CRLF}{CRLF}{v}{CRLF}').encode()
    out += (f"--{b}{CRLF}Content-Disposition: form-data; "
            f'name="file"; filename="{filename}"{CRLF}'
            f"Content-Type: audio/wav{CRLF}{CRLF}").encode()
    return b, out + blob + CRLF.encode() + f"--{b}--{CRLF}".encode()


def transcribe_api(model, clip, key, mode="transcribe", retries=6):
    """(text, latency_s, detected_lang) - the API counterpart of transcribe_timed().

    Latency is a network round-trip, not compute, so it is NOT comparable to the local
    path's and must never be reported as RTF. Backs off on 429 and 5xx: a sustained
    rate limit otherwise loses every call already paid for.
    """
    import json
    import urllib.error
    import urllib.request
    blob = Path(clip).read_bytes()
    for attempt in range(retries):
        b, body = _multipart({"model": model, "mode": mode}, Path(clip).name, blob)
        req = urllib.request.Request(SAARAS_URL, data=body, headers={
            "api-subscription-key": key,
            "Content-Type": f"multipart/form-data; boundary={b}"})
        t = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                d = json.loads(r.read())
            return ((d.get("transcript") or "").strip(),
                    time.perf_counter() - t, d.get("language_code"))
        except urllib.error.HTTPError as e:
            if (e.code == 429 or e.code >= 500) and attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise SystemExit(
                f"Saaras HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:300]}")


def _self_check():
    import tempfile
    import numpy as np
    import soundfile as sf

    assert is_api("saaras:v4") and not is_api("ARTPARK-IISc/SraVaani-1.0")

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

    b, body = _multipart({"model": "saaras:v4", "mode": "transcribe"}, "x.wav", b"RIFFdata")
    text = body.decode("latin-1")
    assert text.startswith(f"--{b}\r\n"), text[:40]
    assert text.endswith(f"--{b}--\r\n"), text[-40:]
    assert 'name="model"\r\n\r\nsaaras:v4\r\n' in text
    assert 'name="mode"\r\n\r\ntranscribe\r\n' in text
    assert 'name="file"; filename="x.wav"\r\n' in text
    assert b"RIFFdata" in body          # the payload survives byte for byte
    print("asr self-check ok")


if __name__ == "__main__":
    _self_check()
