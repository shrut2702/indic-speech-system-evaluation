"""Bench manifests and clip paths. The manifest is the source of truth."""
import json
from pathlib import Path

# noisy_5db clips and results are kept, but are not a reported condition: it failed
# to discriminate (see CLAUDE.md). The default pair is what gets reported.
CONDITIONS = ("clean", "noisy_0db")


def languages(bench_dir):
    return sorted(p.stem.split("_", 1)[1] for p in Path(bench_dir).glob("items_*.jsonl"))


def items(bench_dir, lang):
    f = Path(bench_dir) / f"items_{lang}.jsonl"
    return [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]


def clip(bench_dir, cond, lang, category, item_id):
    return Path(bench_dir) / cond / lang / category / f"{item_id}.wav"


def examples(bench_dir, langs=None, conds=CONDITIONS):
    """(found, missing) lists of (item, condition, clip_path). Clips may not exist yet."""
    found, missing = [], []
    for lang in langs or languages(bench_dir):
        for it in items(bench_dir, lang):
            for cond in conds:
                p = clip(bench_dir, cond, lang, it["category"], it["item_id"])
                (found if p.exists() else missing).append((it, cond, p))
    return found, missing
