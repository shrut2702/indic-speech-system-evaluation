"""Single entry point.

    python main.py pipeline --langs hi ta --conditions clean
    python main.py llm-only
    python main.py eval artifacts/runs/<run_id>.jsonl

Inference and scoring are decoupled: eval reads logged records and never calls a model.
"""
import argparse
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

# redirected stdout defaults to cp1252 on Windows and dies on Indic progress lines
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="mode", required=True)

    p = sub.add_parser("pipeline", help="run ASR -> LLM over the bench, log records")
    p.add_argument("--langs", nargs="*", help="default: every items_*.jsonl manifest")
    p.add_argument("--conditions", nargs="*", default=None)
    p.add_argument("--limit", type=int, help="first N examples, for a smoke test")
    p.add_argument("--run-id")
    p.add_argument("--repeat", type=int, default=1,
                   help="ASR timings per example; median is logged (RTF needs >=3)")
    p.add_argument("--bench", help="override the bench dir (e.g. a trial set)")
    p.add_argument("--asr-model", help="override params.yaml asr.model, e.g. saaras:v4")
    p.add_argument("--asr-mode", help="saaras only: transcribe|verbatim|translit|codemix")
    p.add_argument("--llm-model", help="override params.yaml llm.model")
    p.add_argument("--llm-reasoning", help="sarvam only: off|low|medium|high")

    l = sub.add_parser("llm-only", help="LLM on reference transcripts, no ASR stage")
    l.add_argument("--langs", nargs="*", help="default: every items_*.jsonl manifest")
    l.add_argument("--limit", type=int, help="first N items, for a smoke test")
    l.add_argument("--run-id", default="llm_only_eval_results")
    l.add_argument("--bench", help="override the bench dir (e.g. a trial set)")
    l.add_argument("--llm-model", help="override params.yaml llm.model")
    l.add_argument("--llm-reasoning", help="sarvam only: off|low|medium|high")

    e = sub.add_parser("eval", help="score already-logged records")
    e.add_argument("records", help="artifacts/runs/<run_id>.jsonl")

    rs = sub.add_parser("responses", help="score logged LLM responses against the gold")
    rs.add_argument("records", nargs="+", help="one or more artifacts/runs/*.jsonl")

    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "config" / "config.yaml").read_text())["paths"]
    # before any transformers/hub import: a 600M checkpoint must not land on C:
    os.environ.setdefault("HF_HOME", str(ROOT / cfg.get("hf_cache", "artifacts/hf")))
    params = yaml.safe_load((ROOT / "params.yaml").read_text())

    # CLI beats params.yaml, so a backend swap needs no file edit
    for flag, section, field in (("asr_model", "asr", "model"),
                                 ("asr_mode", "asr", "mode"),
                                 ("llm_model", "llm", "model"),
                                 ("llm_reasoning", "llm", "reasoning_effort")):
        if getattr(args, flag, None):
            params[section][field] = getattr(args, flag)

    if args.mode == "pipeline":
        from speechSys import dataset, pipeline
        pipeline.run(
            bench_dir=ROOT / (args.bench or cfg["bench"]),
            runs_dir=ROOT / cfg["runs"],
            work_dir=ROOT / cfg["work"],
            params=params,
            langs=args.langs or None,
            conds=tuple(args.conditions) if args.conditions else dataset.CONDITIONS,
            limit=args.limit,
            run_id=args.run_id,
            repeat=args.repeat,
        )
    elif args.mode == "llm-only":
        from speechSys import pipeline
        pipeline.run_llm_only(
            bench_dir=ROOT / (args.bench or cfg["bench"]),
            runs_dir=ROOT / cfg["runs"],
            params=params,
            langs=args.langs or None,
            limit=args.limit,
            run_id=args.run_id,
        )
    elif args.mode == "responses":
        from speechSys import responses
        responses.run(args.records, ROOT / cfg.get("results", "artifacts/results"))
    else:
        from speechSys import evaluate
        evaluate.run(args.records, ROOT / cfg.get("results", "artifacts/results"))


if __name__ == "__main__":
    main()
