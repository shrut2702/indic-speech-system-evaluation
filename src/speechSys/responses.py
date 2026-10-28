"""Score logged LLM responses against the gold answer. No model call, no judge.

    python main.py responses artifacts/runs/<a>.jsonl [<b>.jsonl ...]

Closed-form items with a known reference_answer + answer_aliases, so a match is
decidable. Anything the matcher cannot settle is printed for a human to label -
that set is also the only part a judge would ever be needed for.
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

from .normalize import normalize

def label(r):
    """correct / incorrect / empty. Never guesses.

    Hallucination, language drift and trap answers are deliberately out of scope.
    """
    lang, resp = r["lang"], (r.get("llm_response") or "")
    gold = [r["reference_answer"], *(r.get("answer_aliases") or [])]
    nr = normalize(resp, lang)
    hit = next((g for g in gold if normalize(g, lang) in nr), None)
    return {
        "item_id": r["item_id"], "lang": lang, "category": r["category"],
        "condition": r.get("acoustic_condition") or r.get("input_source", "-"),
        "label": "empty" if not resp.strip() else "correct" if hit else "incorrect",
        "matched_on": hit or "",
        "llm_attempts": r.get("llm_attempts"),
        "response": resp,
        "reference_answer": r["reference_answer"],
    }


def _acc(rows):
    n = len(rows)
    c = sum(r["label"] == "correct" for r in rows)
    return {"n": n, "correct": c, "acc": c / n if n else 0.0,
            "incorrect": sum(r["label"] == "incorrect" for r in rows),
            "empty": sum(r["label"] == "empty" for r in rows)}


def _table(title, groups):
    print("\n" + title)
    head = f"{'group':18}{'n':>5}{'correct':>9}{'acc':>7}{'incorr':>8}{'empty':>7}"
    print(head)
    print("-" * len(head))
    for name, a in groups:
        print(f"{name:18}{a['n']:>5}{a['correct']:>9}{a['acc']:>7.3f}"
              f"{a['incorrect']:>8}{a['empty']:>7}")


def score_file(path):
    rows = [label(json.loads(l))
            for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        raise SystemExit(f"{path}: no records")
    return rows


def run(paths, results_dir):
    scored = {}
    for p in paths:
        p = Path(p)
        rows = score_file(p)
        scored[p.stem] = rows
        out = Path(results_dir) / p.stem
        out.mkdir(parents=True, exist_ok=True)
        with (out / "responses.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

        def by(key):
            g = defaultdict(list)
            for r in rows:
                g[r[key]].append(r)
            return [(k, _acc(v)) for k, v in sorted(g.items())]

        print("\n" + "=" * 70)
        print(p.stem)
        _table("OVERALL", [("all", _acc(rows))])
        _table("BY LANGUAGE", by("lang"))
        _table("BY CATEGORY", by("category"))
        print(f"\nwrote {out / 'responses.csv'}")

    # paired attribution: same item, control vs pipeline. Needs no judge.
    names = list(scored)
    ctrl = next((n for n in names if "llm_only" in n), names[0])
    for other in [n for n in names if n != ctrl]:
        a = {(r["item_id"], r["lang"]): r for r in scored[ctrl]}
        b = {(r["item_id"], r["lang"]): r for r in scored[other]}
        keys = sorted(set(a) & set(b))
        cell = defaultdict(list)
        for k in keys:
            cell[(a[k]["label"] == "correct", b[k]["label"] == "correct")].append(k)
        print(f"\nPAIRED  {ctrl}  ->  {other}   (n={len(keys)})")
        print(f"  both correct                 {len(cell[(True, True)]):>4}")
        print(f"  ASR-induced failure          {len(cell[(True, False)]):>4}"
              "   (control right, pipeline wrong)")
        print(f"  LLM limitation               {len(cell[(False, False)]):>4}"
              "   (wrong either way)")
        print(f"  recovered under ASR          {len(cell[(False, True)]):>4}"
              "   (investigate)")
        for k in cell[(True, False)][:12]:
            print(f"    ASR-induced: {k[1]} {k[0]:7} "
                  f"gold {a[k]['reference_answer'][:22]!r} -> {b[k]['response'][:34]!r}")
    return scored


def _self_check():
    base = {"item_id": "x", "category": "normal", "answer_aliases": [],
            "llm_attempts": 1}
    ok = label({**base, "lang": "hi", "reference_answer": "1788",
                "llm_response": "१७८८ रुपये"})
    assert ok["label"] == "correct", ok          # native digits normalize to ASCII

    # the normalizer already folds Tamil number words to digits, so "88" matches the
    # spelled-out reference directly - the alias is not what rescues it
    al = label({**base, "lang": "ta", "reference_answer": "எண்பத்தி எட்டு",
                "answer_aliases": [], "llm_response": "88"})
    assert al["label"] == "correct", al
    contracted = label({**base, "lang": "ta", "reference_answer": "எண்பத்தி எட்டு",
                        "llm_response": "எண்பத்தெட்டு"})
    assert contracted["label"] == "incorrect", contracted   # known alias gap

    bad = label({**base, "lang": "gu", "reference_answer": "1788",
                 "llm_response": "૧૭૯૮"})
    assert bad["label"] == "incorrect", bad

    assert label({**base, "lang": "hi", "reference_answer": "x",
                  "llm_response": "   "})["label"] == "empty"

    # the normalizer folds Latin into the native script, so a Latinized answer counts
    latin = label({**base, "lang": "hi", "reference_answer": "गोदावरी",
                   "llm_response": "Godavari"})
    assert latin["label"] == "correct", latin
    print("responses self-check ok")


if __name__ == "__main__":
    _self_check()
