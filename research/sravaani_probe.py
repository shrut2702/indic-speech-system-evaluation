"""Run SraVaani-1.0 in isolation: env check, transcribe, RTF, decoder/batch probing.

Gated model - accept the terms at huggingface.co/ARTPARK-IISc/SraVaani-1.0, then
either `huggingface-cli login` or pass --token.

    python research/sravaani_probe.py sample.wav
    python research/sravaani_probe.py --inspect        # what the model object exposes
"""
import argparse, os, time, wave, contextlib, sys
from pathlib import Path

REPO = "ARTPARK-IISc/SraVaani-1.0"

def env_token():
    """HF_TOKEN from the environment, else from .env at the repo root."""
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    f = Path(__file__).resolve().parent.parent / ".env"
    if f.exists():
        for line in f.read_text(encoding="utf-8").splitlines():
            k, _, v = line.partition("=")
            if k.strip() == "HF_TOKEN":
                return v.strip().strip("\"'")
    return None

def duration(path):
    with contextlib.closing(wave.open(path)) as w:
        if w.getframerate() != 16000 or w.getnchannels() != 1:
            print(f"  warn: {path} is {w.getframerate()}Hz/{w.getnchannels()}ch, want 16kHz mono")
        return w.getnframes() / w.getframerate()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", nargs="*")
    ap.add_argument("--token", default=env_token())
    ap.add_argument("--inspect", action="store_true", help="dump public API and exit")
    ap.add_argument("--repeat", type=int, default=1, help="runs for a median RTF")
    args = ap.parse_args()

    import torch
    from transformers import AutoModel

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"torch {torch.__version__} device={dev}")
    if dev == "cuda":
        cap = torch.cuda.get_device_capability()
        print(f"gpu {torch.cuda.get_device_name()} sm{cap[0]}{cap[1]} "
              f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB "
              f"bf16={'yes' if cap[0] >= 8 else 'no, use fp16'}")

    t = time.perf_counter()
    model = AutoModel.from_pretrained(REPO, trust_remote_code=True, token=args.token).to(dev).eval()
    print(f"load {time.perf_counter() - t:.1f}s")
    if dev == "cuda":
        print(f"vram {torch.cuda.memory_allocated() / 1e9:.2f}GB")

    if args.inspect:
        # the card documents only transcribe(); find the decoder switch and batch args
        import inspect
        print("\npublic attrs:", [a for a in dir(model) if not a.startswith("_")])
        try:
            print("\ntranscribe signature:", inspect.signature(model.transcribe))
        except (TypeError, ValueError) as e:
            print(f"\nsignature unavailable: {e}")
        print("\ndoc:", (model.transcribe.__doc__ or "none")[:800])
        return

    if not args.audio:
        sys.exit("no audio given; use --inspect to explore the API")

    audio_s = sum(duration(p) for p in args.audio)

    model.transcribe(args.audio[0], return_hypotheses=True)  # warm up
    times = []
    for _ in range(args.repeat):
        t = time.perf_counter()
        hyps = model.transcribe(args.audio, return_hypotheses=True)
        times.append(time.perf_counter() - t)

    med = sorted(times)[len(times) // 2]
    print(f"\ntranscribe median {med:.2f}s over {args.repeat} run(s) "
          f"for {audio_s:.2f}s audio  rtf={med / audio_s:.3f}")
    print(f"all times: {[round(x, 2) for x in times]}")
    for p, h in zip(args.audio, hyps):
        print(f"\n{p}\n  {getattr(h, 'text', h)}")

if __name__ == "__main__":
    main()
