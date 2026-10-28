"""SraVaani over one bench condition, no LLM. Emits pipeline-shaped records.

Throwaway, so it duplicates the resample + transcribe bits rather than importing
src/speechSys. The records carry no LLM fields, which main.py eval tolerates, so
WER/CER/entErr come from the real harness and not from a second implementation:

    python research/asr_only_run.py --cond noisy_0db
    python main.py eval artifacts/runs/asr_only_noisy_0db_sravaani1.0.jsonl
"""
import argparse, json, os, platform, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = "ARTPARK-IISc/SraVaani-1.0"
TARGET_SR = 16000
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
# model downloads must not land on the system drive
os.environ.setdefault("HF_HOME", str(ROOT / "artifacts" / "hf"))


def env_token():
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    f = ROOT / ".env"
    for line in f.read_text(encoding="utf-8").splitlines() if f.exists() else []:
        k, _, v = line.partition("=")
        if k.strip() == "HF_TOKEN":
            return v.strip().strip("\"'")
    return None


def as_16k_mono(src, out):
    import soundfile as sf
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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cond", required=True)
    ap.add_argument("--langs", nargs="*")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--run-id")
    a = ap.parse_args()

    import soundfile as sf
    import torch
    from transformers import AutoModel

    bench = ROOT / "artifacts" / "bench"
    langs = a.langs or sorted(p.stem.split("_", 1)[1] for p in bench.glob("items_*.jsonl"))
    rows = []
    for lang in langs:
        for it in [json.loads(l) for l in
                   (bench / f"items_{lang}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]:
            clip = bench / a.cond / lang / it["category"] / f"{it['item_id']}.wav"
            if clip.exists():
                rows.append((it, clip))
            else:
                print(f"  missing {clip}")
    if not rows:
        sys.exit(f"no clips under {bench / a.cond}")
    if a.limit:
        rows = rows[:a.limit]

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(REPO, trust_remote_code=True,
                                      token=env_token()).to(dev).eval()
    machine = (f"{platform.system()} {platform.machine()} / "
               f"{torch.cuda.get_device_name() if dev == 'cuda' else platform.processor()}")
    run_id = a.run_id or f"asr_only_{a.cond}_sravaani1.0"
    out = ROOT / "artifacts" / "runs" / f"{run_id}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"device={dev}  {len(rows)} clips  cond={a.cond}")

    # warm up: the first call per input shape pays a ~3x cuDNN autotune cost
    warm = as_16k_mono(rows[0][1], ROOT / "artifacts/work/16k/warmup.wav")
    for _ in range(3):
        model.transcribe(str(warm), return_hypotheses=True)

    with out.open("w", encoding="utf-8") as fh:
        for i, (it, clip) in enumerate(rows, 1):
            wav = as_16k_mono(clip, ROOT / "artifacts/work/16k" / a.cond / it["lang"]
                              / f"{it['item_id']}.wav")
            t = time.perf_counter()
            h = model.transcribe(str(wav), return_hypotheses=True)[0]
            dt = time.perf_counter() - t
            hyp = getattr(h, "text", h)
            fh.write(json.dumps({
                "run_id": run_id,
                "item_id": it["item_id"],
                "lang": it["lang"],
                "category": it["category"],
                "acoustic_condition": a.cond,
                "clip_path": str(clip.relative_to(ROOT)).replace("\\", "/"),
                "audio_duration_s": round(sf.info(str(clip)).duration, 3),
                "reference_transcript": it["reference_transcript"],
                "reference_answer": it["reference_answer"],
                "answer_aliases": it.get("answer_aliases") or [],
                "reference_entities": it.get("reference_entities") or [],
                "trap_answer": it.get("trap_answer"),
                "asr_hypothesis": hyp,
                "asr_model": REPO,
                "asr_params": {"batch_size": 1},
                "asr_latency_s": round(dt, 4),
                "machine": machine,
            }, ensure_ascii=False) + "\n")
            print(f"  [{i}/{len(rows)}] {it['lang']} {it['item_id']:7} {dt:5.2f}s | {hyp[:60]!r}")

    print(f"\n{len(rows)} records -> {out}")


if __name__ == "__main__":
    main()
