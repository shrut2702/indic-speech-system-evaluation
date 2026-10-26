"""ASR -> LLM over the bench, one JSONL record per example.

No metrics are computed here. evaluate.py derives every number from the records, so
re-scoring never re-runs inference.
"""
import collections
import json
import os
import platform
import statistics
import time
import uuid
from pathlib import Path

from . import asr, dataset, llm

ROOT = Path(__file__).resolve().parents[2]


def hf_token():
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    f = ROOT / ".env"
    for line in f.read_text(encoding="utf-8").splitlines() if f.exists() else []:
        k, _, v = line.partition("=")
        if k.strip() == "HF_TOKEN":
            return v.strip().strip("\"'")
    return None


def run(bench_dir, runs_dir, work_dir, params, langs=None,
        conds=dataset.CONDITIONS, limit=None, run_id=None, repeat=1):
    import torch

    found, missing = dataset.examples(bench_dir, langs, conds)
    if missing:
        by_set = collections.Counter((it["lang"], cond) for it, cond, _ in missing)
        print(f"  skip (clips missing): {len(missing)} - " +
              ", ".join(f"{lang}/{cond} x{n}" for (lang, cond), n in sorted(by_set.items())))
    if not found:
        raise SystemExit(f"no clips found under {bench_dir}")
    if limit:
        found = found[:limit]

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    token = hf_token()
    a = params["asr"]
    g = params["llm"]
    print(f"device={dev}  asr={a['model']}  llm={g['model']}  {len(found)} examples")

    asr_model = asr.load(a["model"], dev, token)
    tok, llm_model = llm.load(g["model"], dev, token, g.get("dtype", "float16"))

    machine = (f"{platform.system()} {platform.machine()} / "
               f"{torch.cuda.get_device_name() if dev == 'cuda' else platform.processor()}")
    run_id = run_id or f"run-{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"
    out = Path(runs_dir) / f"{run_id}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    # warm up both stages; one ASR call is not enough, cuDNN still autotunes on the next
    warm = asr.as_16k_mono(found[0][2], Path(work_dir) / "warmup.wav")
    for _ in range(3):
        asr.transcribe(asr_model, warm)
    llm.respond(tok, llm_model, "hello", max_new_tokens=4)

    with out.open("w", encoding="utf-8") as fh:
        for i, (item, cond, clip) in enumerate(found, 1):
            wav16 = asr.as_16k_mono(
                clip, Path(work_dir) / cond / item["lang"] / f"{item['item_id']}.wav")
            # repeated because ASR latency shows sporadic multi-second autotune spikes
            times = []
            for _ in range(repeat):
                hyp, dt = asr.transcribe_timed(asr_model, wav16)
                times.append(dt)
            asr_s = statistics.median(times)
            resp, llm_s, n_tok = llm.respond(
                tok, llm_model, hyp, max_new_tokens=g.get("max_new_tokens", 48))
            fh.write(json.dumps({
                "run_id": run_id,
                "item_id": item["item_id"],
                "lang": item["lang"],
                "category": item["category"],
                "acoustic_condition": cond,
                "clip_path": str(clip.relative_to(ROOT)).replace("\\", "/"),
                "audio_duration_s": round(asr.duration(clip), 3),
                "reference_transcript": item["text"],
                "reference_answer": item["reference_answer"],
                "answer_aliases": item.get("answer_aliases") or [],
                "reference_entities": item.get("reference_entities") or [],
                "trap_answer": item.get("trap_answer"),
                "asr_hypothesis": hyp,
                "llm_response": resp,
                "asr_model": a["model"],
                "asr_params": {k: v for k, v in a.items() if k != "model"},
                "llm_model": g["model"],
                "llm_params": {k: v for k, v in g.items() if k != "model"},
                "asr_latency_s": round(asr_s, 4),
                "asr_latency_all_s": [round(x, 4) for x in times],
                "llm_latency_s": round(llm_s, 4),
                "llm_new_tokens": n_tok,
                "machine": machine,
            }, ensure_ascii=False) + "\n")
            print(f"  [{i}/{len(found)}] {item['lang']} {item['item_id']:7} {cond:9} "
                  f"asr {asr_s:5.2f}s llm {llm_s:5.2f}s | {resp[:40]!r}")

    print(f"\n{len(found)} records -> {out}")
    return out
