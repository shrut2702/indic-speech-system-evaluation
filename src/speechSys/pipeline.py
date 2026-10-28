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

    # a model name starting with "saaras"/"sarvam" selects a hosted API: no VRAM, no
    # local load. Both live on the same Sarvam key.
    asr_api = asr.is_api(a["model"])
    api = g["model"].startswith("sarvam")
    key = llm._sarvam_key(ROOT) if (api or asr_api) else None
    asr_model = None if asr_api else asr.load(a["model"], dev, token)
    if api:
        tok = llm_model = None
    else:
        tok, llm_model = llm.load(g["model"], dev, token, g.get("dtype", "float16"))
    if asr_api and repeat > 1:
        raise SystemExit("--repeat >1 bills the ASR API once per pass; use --repeat 1")

    def hear(clip, work):
        """(hypothesis, latency_s). The API gets the clip as-is; local path needs 16 kHz."""
        if asr_api:
            txt, dt, _ = asr.transcribe_api(a["model"], clip, key,
                                            mode=a.get("mode", "transcribe"))
            return txt, dt
        return asr.transcribe_timed(asr_model, asr.as_16k_mono(clip, work))

    def answer(question):
        if api:
            return llm.respond_api(g["model"], question, key,
                                   max_new_tokens=g.get("max_new_tokens", 48),
                                   reasoning=g.get("reasoning_effort", "default"))
        return llm.respond(tok, llm_model, question,
                           max_new_tokens=g.get("max_new_tokens", 48))

    machine = (f"{platform.system()} {platform.machine()} / "
               f"{torch.cuda.get_device_name() if dev == 'cuda' else platform.processor()}")
    run_id = run_id or f"run-{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"
    out = Path(runs_dir) / f"{run_id}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    # warm up both stages; one local ASR call is not enough, cuDNN still autotunes on
    # the next. The API path is not warmed: every warm-up call would be billed.
    if not asr_api:
        warm = asr.as_16k_mono(found[0][2], Path(work_dir) / "warmup.wav")
        for _ in range(3):
            asr.transcribe(asr_model, warm)
    answer("hello")  # warm the API connection or the local decode path

    with out.open("w", encoding="utf-8") as fh:
        for i, (item, cond, clip) in enumerate(found, 1):
            t0 = time.perf_counter()
            work = Path(work_dir) / cond / item["lang"] / f"{item['item_id']}.wav"
            # repeat>1 medians away the ~3x first-pass cuDNN autotune cost per clip shape;
            # at the default 1 the logged latency is the real cold cost of one request
            times = []
            for _ in range(repeat):
                hyp, dt = hear(clip, work)
                times.append(dt)
            asr_s = statistics.median(times)
            resp, llm_s, n_tok, llm_ttft, tries = answer(hyp)
            e2e_s = time.perf_counter() - t0
            fh.write(json.dumps({
                "run_id": run_id,
                "item_id": item["item_id"],
                "lang": item["lang"],
                "category": item["category"],
                "acoustic_condition": cond,
                "clip_path": str(clip.relative_to(ROOT)).replace("\\", "/"),
                "audio_duration_s": round(asr.duration(clip), 3),
                "reference_transcript": item["reference_transcript"],
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
                "llm_ttft_s": round(llm_ttft, 4) if llm_ttft is not None else None,
                "llm_new_tokens": n_tok,
                "llm_attempts": tries,
                # measured, not asr+llm: includes resampling and request overhead.
                # Only meaningful at repeat=1, where ASR ran once.
                "e2e_latency_s": round(e2e_s, 4),
                "machine": machine,
            }, ensure_ascii=False) + "\n")
            print(f"  [{i}/{len(found)}] {item['lang']} {item['item_id']:7} {cond:9} "
                  f"asr {asr_s:5.2f}s llm {llm_s:5.2f}s | {resp[:40]!r}")

    print(f"\n{len(found)} records -> {out}")
    return out


def run_llm_only(bench_dir, runs_dir, params, langs=None, limit=None, run_id=None):
    """LLM on reference transcripts: the control with no ASR error to compound.

    Same records as run(), minus every audio/ASR field. Latency is not logged - there
    is no acoustic condition to attribute it to, and the retry path makes it noisy.
    """
    g = params["llm"]
    api = g["model"].startswith("sarvam")
    if api:
        key = llm._sarvam_key(ROOT)
        tok = llm_model = None
    else:
        import torch
        key = None
        tok, llm_model = llm.load(g["model"], "cuda" if torch.cuda.is_available() else "cpu",
                                  hf_token(), g.get("dtype", "float16"))

    def answer(question):
        if api:
            return llm.respond_api(g["model"], question, key,
                                   max_new_tokens=g.get("max_new_tokens", 48),
                                   reasoning=g.get("reasoning_effort", "default"))
        return llm.respond(tok, llm_model, question,
                           max_new_tokens=g.get("max_new_tokens", 48))

    rows = [it for lang in (langs or dataset.languages(bench_dir))
            for it in dataset.items(bench_dir, lang)]
    if not rows:
        raise SystemExit(f"no manifests found under {bench_dir}")
    if limit:
        rows = rows[:limit]

    run_id = run_id or "llm_only_eval_results"
    out = Path(runs_dir) / f"{run_id}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"llm={g['model']}  {len(rows)} reference transcripts, no ASR stage")
    answer("hello")

    empty = retried = 0
    with out.open("w", encoding="utf-8") as fh:
        for i, item in enumerate(rows, 1):
            resp, _, n_tok, _, tries = answer(item["reference_transcript"])
            empty += not resp
            retried += tries - 1
            fh.write(json.dumps({
                "run_id": run_id,
                "item_id": item["item_id"],
                "lang": item["lang"],
                "category": item["category"],
                "input_source": "reference_transcript",
                "input_text": item["reference_transcript"],
                "reference_transcript": item["reference_transcript"],
                "reference_answer": item["reference_answer"],
                "answer_aliases": item.get("answer_aliases") or [],
                "reference_entities": item.get("reference_entities") or [],
                "trap_answer": item.get("trap_answer"),
                "llm_response": resp,
                "llm_model": g["model"],
                "llm_params": {k: v for k, v in g.items() if k != "model"},
                "llm_new_tokens": n_tok,
                "llm_attempts": tries,
            }, ensure_ascii=False) + "\n")
            print(f"  [{i}/{len(rows)}] {item['lang']:8} {item['item_id']:7} "
                  f"try {tries} | {resp[:45]!r}")

    print(f"\n{len(rows)} records -> {out}")
    print(f"empty {empty}/{len(rows)}   retries spent {retried}")
    return out
