"""Single entry point.

    python main.py pipeline --langs hi ta --conditions clean
    python main.py eval artifacts/runs/<run_id>.jsonl

Inference and scoring are decoupled: eval reads logged records and never calls a model.
"""
import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


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

    e = sub.add_parser("eval", help="score already-logged records")
    e.add_argument("records", help="artifacts/runs/<run_id>.jsonl")

    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "config" / "config.yaml").read_text())["paths"]
    params = yaml.safe_load((ROOT / "params.yaml").read_text())

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
    else:
        raise SystemExit(
            "eval is not implemented yet. src/speechSys/evaluate.py does not exist.\n"
            f"The records are logged and complete, so scoring {args.records} needs no "
            "re-run of inference once evaluate.py lands:\n"
            "  - raw + normalized WER/CER via speechSys.normalize.normalize(text, lang)\n"
            "  - entity error rate over reference_entities\n"
            "  - answer correctness against reference_answer + answer_aliases\n"
            "  - ASR RTF = asr_latency_s / audio_duration_s; LLM latency and tok/s"
        )


if __name__ == "__main__":
    main()
