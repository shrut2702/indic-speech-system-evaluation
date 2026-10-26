"""Transcribe the trial clips and log one raw record per clip.

Timing is per clip (not batched) so the latency is attributable. No metrics here -
evaluate.py derives those from the records.

    python research/log_trial.py --repeat 3
"""
import argparse, json, os, platform, time, wave, contextlib, statistics, uuid
from pathlib import Path

REPO = "ARTPARK-IISc/SraVaani-1.0"
ROOT = Path(__file__).resolve().parent.parent
TRIAL = ROOT / "artifacts" / "trial"

# which bench item each trial clip is, since the filenames do not say
CLIPS = [
    ("ent_01", "hi", "entity_number", "ElevenLabs_2026-10-25T23_12_46__s50_v4"),
    ("cs_04", "pa", "code_switch", "multi-speaker"),
    ("cs_02", "gu", "code_switch", "multi-speaker-gu"),
    ("lng_01", "ta", "long_15s", "multi-speaker-ta"),
    ("lng_01", "gu", "long_15s", "multi-speaker-gu-2"),
    ("dis_01", "en", "disfluent", "multi-speaker-en"),
]
CONDITIONS = [("clean", "_clean16k"), ("noisy_5db", "_noisy5db")]

def env_token():
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    f = ROOT / ".env"
    for line in f.read_text(encoding="utf-8").splitlines() if f.exists() else []:
        k, _, v = line.partition("=")
        if k.strip() == "HF_TOKEN":
            return v.strip().strip("\"'")
    return None

def duration(path):
    with contextlib.closing(wave.open(str(path))) as w:
        return w.getnframes() / w.getframerate()

def manifest(lang):
    f = ROOT / "artifacts" / "bench" / f"items_{lang}.jsonl"
    return {json.loads(l)["item_id"]: json.loads(l)
            for l in f.read_text(encoding="utf-8").splitlines() if l.strip()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch
    from transformers import AutoModel

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(REPO, trust_remote_code=True, token=env_token()).to(dev).eval()
    machine = f"{platform.system()} {platform.machine()} / {torch.cuda.get_device_name() if dev == 'cuda' else 'cpu'}"
    run_id = f"trial-{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"

    out = Path(args.out) if args.out else ROOT / "artifacts" / "runs" / f"{run_id}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    model.transcribe(str(TRIAL / f"{CLIPS[0][3]}_clean16k.wav"))  # warm up CUDA
    n = 0
    with out.open("w", encoding="utf-8") as fh:
        for item_id, lang, category, stem in CLIPS:
            ref = manifest(lang)[item_id]
            for cond, suffix in CONDITIONS:
                clip = TRIAL / f"{stem}{suffix}.wav"
                if not clip.exists():
                    print(f"  skip (missing): {clip.name}")
                    continue
                times, hyp = [], None
                for _ in range(args.repeat):
                    t = time.perf_counter()
                    hyps = model.transcribe(str(clip), return_hypotheses=True)
                    times.append(time.perf_counter() - t)
                    hyp = getattr(hyps[0], "text", hyps[0])
                fh.write(json.dumps({
                    "run_id": run_id,
                    "item_id": item_id,
                    "lang": lang,
                    "category": category,
                    "acoustic_condition": cond,
                    "clip_path": str(clip.relative_to(ROOT)).replace("\\", "/"),
                    "audio_duration_s": round(duration(clip), 3),
                    "reference_transcript": ref["text"],
                    "reference_answer": ref["reference_answer"],
                    "reference_entities": ref.get("reference_entities"),
                    "asr_hypothesis": hyp,
                    "asr_model": REPO,
                    "asr_params": {"decoder": "default", "batch_size": 1},
                    "asr_latency_s": round(statistics.median(times), 4),
                    "asr_latency_all_s": [round(x, 4) for x in times],
                    "llm_model": None,
                    "llm_response": None,
                    "llm_latency_s": None,
                    "machine": machine,
                }, ensure_ascii=False) + "\n")
                n += 1
                print(f"  {item_id:7} {cond:11} {statistics.median(times):6.3f}s  {hyp[:48]}")
    print(f"\n{n} records -> {out}")

if __name__ == "__main__":
    main()
