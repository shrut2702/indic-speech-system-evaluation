"""Score logged records. Never calls a model - re-scoring needs no re-inference.

    python main.py eval artifacts/runs/<run_id>.jsonl

WER is reported micro (pooled errors / pooled reference words, what ASR papers
publish) and macro (mean of per-sample rates). Writes per_sample.csv and
summary.json under artifacts/results/<run_id>/.
"""
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from .normalize import normalize

BUCKET_S = 5


def edit_ops(ref, hyp):
    """(substitutions, deletions, insertions, len(ref)) by Levenshtein backtrace."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1,
                          d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    s = dl = ins = 0
    i, j = n, m
    while i or j:
        if i and j and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            s += ref[i - 1] != hyp[j - 1]
            i, j = i - 1, j - 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            dl += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return s, dl, ins, n


def rate(s, dl, ins, n):
    """Not clamped: insertions can push WER above 1."""
    return (s + dl + ins) / n if n else 0.0


def percentile(values, p):
    """Nearest-rank, so it is defined even for a single sample."""
    if not values:
        return None
    v = sorted(values)
    return v[max(0, math.ceil(p / 100 * len(v)) - 1)]


def bucket(seconds):
    lo = int(seconds // BUCKET_S) * BUCKET_S
    return f"{lo}-{lo + BUCKET_S}s"


def score_record(r):
    ref, hyp, lang = r["reference_transcript"], r["asr_hypothesis"], r["lang"]
    nref, nhyp = normalize(ref, lang), normalize(hyp, lang)

    raw = edit_ops(ref.split(), hyp.split())
    norm = edit_ops(nref.split(), nhyp.split())
    cer = edit_ops(nref.replace(" ", ""), nhyp.replace(" ", ""))  # spaces do not count

    ents = r.get("reference_entities") or []
    kept = sum(1 for e in ents if normalize(e, lang) in nhyp)

    dur = r["audio_duration_s"]
    asr_s, llm_s = r.get("asr_latency_s"), r.get("llm_latency_s")
    ttft, n_tok = r.get("llm_ttft_s"), r.get("llm_new_tokens")
    resp = r.get("llm_response")
    # measured wall clock when the pipeline logged it; older records only have the sum
    e2e = r.get("e2e_latency_s")
    if e2e is None:
        e2e = asr_s + llm_s if (asr_s is not None and llm_s is not None) else asr_s
    return {
        "item_id": r["item_id"], "lang": lang, "category": r["category"],
        "acoustic_condition": r["acoustic_condition"], "audio_duration_s": dur,
        "duration_bucket": bucket(dur),
        "raw_wer": rate(*raw), "norm_wer": rate(*norm), "norm_cer": rate(*cer),
        "raw_errors": raw[0] + raw[1] + raw[2], "raw_ref_words": raw[3],
        "wer_sub": norm[0], "wer_del": norm[1], "wer_ins": norm[2],
        "wer_errors": norm[0] + norm[1] + norm[2], "ref_words": norm[3],
        "cer_errors": cer[0] + cer[1] + cer[2], "ref_chars": cer[3],
        "entities": len(ents), "entities_kept": kept,
        "asr_latency_s": asr_s,
        # computed for every backend. For a hosted ASR it includes the network
        # round-trip, so label it as wall-clock RTF, not a compute figure.
        "asr_rtf": (asr_s / dur) if asr_s and dur else None,
        "llm_latency_s": llm_s, "llm_ttft_s": ttft,
        "llm_empty": 1 if (resp is not None and not str(resp).strip()) else 0,
        "llm_attempts": r.get("llm_attempts"),
        "llm_tok_per_s": (n_tok / llm_s) if (n_tok and llm_s) else None,
        "e2e_latency_s": e2e,
    }


def _agg(rows):
    if not rows:
        return {}

    def micro(err, unit):
        u = sum(r[unit] for r in rows)
        return sum(r[err] for r in rows) / u if u else 0.0

    lat = [r["asr_latency_s"] for r in rows if r["asr_latency_s"] is not None]
    rtf = [r["asr_rtf"] for r in rows if r["asr_rtf"] is not None]
    e2e = [r["e2e_latency_s"] for r in rows if r["e2e_latency_s"] is not None]
    ttft = [r["llm_ttft_s"] for r in rows if r.get("llm_ttft_s") is not None]
    llm = [r["llm_latency_s"] for r in rows if r.get("llm_latency_s") is not None]
    tps = [r["llm_tok_per_s"] for r in rows if r.get("llm_tok_per_s") is not None]
    audio = sum(r["audio_duration_s"] for r in rows)
    ents = sum(r["entities"] for r in rows)
    return {
        "n": len(rows),
        "raw_wer_micro": micro("raw_errors", "raw_ref_words"),
        "raw_wer_macro": statistics.fmean(r["raw_wer"] for r in rows),
        "norm_wer_micro": micro("wer_errors", "ref_words"),
        "norm_wer_macro": statistics.fmean(r["norm_wer"] for r in rows),
        "norm_cer_micro": micro("cer_errors", "ref_chars"),
        "norm_cer_macro": statistics.fmean(r["norm_cer"] for r in rows),
        "sub": sum(r["wer_sub"] for r in rows),
        "del": sum(r["wer_del"] for r in rows),
        "ins": sum(r["wer_ins"] for r in rows),
        "entity_error_rate": (1 - sum(r["entities_kept"] for r in rows) / ents) if ents else None,
        "audio_total_s": audio,
        "asr_total_s": sum(lat) if lat else None,
        "rtf_pooled": (sum(lat) / audio) if lat and audio else None,
        "asr_p50": percentile(lat, 50), "asr_p95": percentile(lat, 95),
        "rtf_p50": percentile(rtf, 50), "rtf_p95": percentile(rtf, 95),
        "e2e_p50": percentile(e2e, 50), "e2e_p95": percentile(e2e, 95),
        "llm_p50": percentile(llm, 50), "llm_p95": percentile(llm, 95),
        "ttft_p50": percentile(ttft, 50), "ttft_p95": percentile(ttft, 95),
        "tok_per_s_p50": percentile(tps, 50),
        "llm_empty_rate": (sum(r["llm_empty"] for r in rows) / len(rows)) if any(
            r.get("llm_latency_s") is not None for r in rows) else None,
        "llm_retries": sum((r["llm_attempts"] or 1) - 1 for r in rows),
    }


WER_COLS = ("n", "raw_wer_micro", "raw_wer_macro", "norm_wer_micro", "norm_wer_macro",
            "norm_cer_micro", "norm_cer_macro", "sub", "del", "ins", "entity_error_rate")
SHORT = {"raw_wer_micro": "rawWERu", "raw_wer_macro": "rawWERM",
         "norm_wer_micro": "nWERu", "norm_wer_macro": "nWERM",
         "norm_cer_micro": "nCERu", "norm_cer_macro": "nCERM",
         "entity_error_rate": "entErr"}


def _table(title, groups):
    print("\n" + title)
    head = f"{'group':20}" + "".join(f"{SHORT.get(k, k):>9}" for k in WER_COLS)
    print(head)
    print("-" * len(head))
    for name, a in groups:
        cells = ""
        for k in WER_COLS:
            v = a.get(k)
            if v is None:
                cells += f"{'-':>9}"
            elif isinstance(v, int):
                cells += f"{v:>9}"
            else:
                cells += f"{v:>9.3f}"
        print(f"{name:20}{cells}")


def run(records_path, results_dir):
    records_path = Path(records_path)
    rows = [score_record(json.loads(l))
            for l in records_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        raise SystemExit(f"{records_path}: no records")

    out = Path(results_dir) / records_path.stem
    out.mkdir(parents=True, exist_ok=True)
    with (out / "per_sample.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def by(key):
        g = defaultdict(list)
        for r in rows:
            g[r[key]].append(r)
        return [(k, _agg(v)) for k, v in sorted(g.items())]

    langs = sorted({r["lang"] for r in rows})
    conds = sorted({r["acoustic_condition"] for r in rows})
    summary = {
        "records": str(records_path), "n": len(rows),
        "overall": _agg(rows),
        "by_lang": dict(by("lang")),
        "by_category": dict(by("category")),
        "by_condition": dict(by("acoustic_condition")),
        "by_duration_bucket": dict(by("duration_bucket")),
        "by_lang_condition": {
            l + "/" + c: _agg([r for r in rows
                               if r["lang"] == l and r["acoustic_condition"] == c])
            for l in langs for c in conds},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                      encoding="utf-8")

    _table("OVERALL", [("all", summary["overall"])])
    _table("BY LANGUAGE", by("lang"))
    _table("BY CATEGORY", by("category"))
    _table("BY CONDITION", by("acoustic_condition"))
    _table("BY LANGUAGE x CONDITION",
           [(k, v) for k, v in summary["by_lang_condition"].items() if v])

    def f(v):
        return "-" if v is None else f"{v:.3f}"

    print("\nASR LATENCY BY DURATION BUCKET")
    head = (f"{'bucket':10}{'n':>5}{'asr_p50':>9}{'asr_p95':>9}"
            f"{'rtf_p50':>9}{'rtf_p95':>9}{'e2e_p50':>9}{'e2e_p95':>9}")
    print(head)
    print("-" * len(head))
    for name, a in sorted(by("duration_bucket"), key=lambda kv: int(kv[0].split("-")[0])):
        print(f"{name:10}{a['n']:>5}{f(a['asr_p50']):>9}{f(a['asr_p95']):>9}"
              f"{f(a['rtf_p50']):>9}{f(a['rtf_p95']):>9}"
              f"{f(a['e2e_p50']):>9}{f(a['e2e_p95']):>9}")

    o = summary["overall"]
    print("\npooled RTF = total asr time / total audio = " + f(o["rtf_pooled"])
          + "   (" + f(o["asr_total_s"]) + "s over "
          + format(o["audio_total_s"], ".1f") + "s audio)")
    print("wrote " + str(out / "per_sample.csv"))
    print("wrote " + str(out / "summary.json"))
    return summary


def _blank(**kw):
    base = {"wer_errors": 0, "ref_words": 1, "norm_wer": 0.0, "raw_wer": 0.0,
            "raw_errors": 0, "raw_ref_words": 1, "cer_errors": 0, "ref_chars": 1,
            "wer_sub": 0, "wer_del": 0, "wer_ins": 0, "norm_cer": 0.0,
            "entities": 0, "entities_kept": 0, "audio_duration_s": 1.0,
            "asr_latency_s": None, "asr_rtf": None, "e2e_latency_s": None,
            "llm_latency_s": None, "llm_ttft_s": None, "llm_tok_per_s": None,
            "llm_empty": 0, "llm_attempts": None}
    base.update(kw)
    return base


def _self_check():
    assert edit_ops("a b c".split(), "a b c".split()) == (0, 0, 0, 3)
    assert edit_ops("a b c".split(), "a x c".split()) == (1, 0, 0, 3)
    assert edit_ops("a b c".split(), "a c".split()) == (0, 1, 0, 3)
    assert edit_ops("a b".split(), "a b c d".split()) == (0, 0, 2, 2)
    assert rate(0, 0, 2, 2) == 1.0
    assert rate(0, 0, 4, 2) == 2.0          # insertions may push WER above 1
    assert rate(1, 1, 0, 0) == 0.0          # empty reference must not divide by zero
    assert percentile([1], 95) == 1
    assert percentile([1, 2, 3, 4], 50) == 2
    assert percentile([1, 2, 3, 4], 95) == 4
    assert percentile([], 50) is None
    assert bucket(0.5) == "0-5s" and bucket(5.0) == "5-10s" and bucket(21.8) == "20-25s"

    # RTF is wall clock / audio duration for every backend, hosted included
    base = {"reference_transcript": "a", "asr_hypothesis": "a", "lang": "en",
            "item_id": "x", "category": "normal", "acoustic_condition": "clean",
            "audio_duration_s": 2.0, "asr_latency_s": 0.5, "reference_answer": "a"}
    assert score_record({**base, "asr_model": "saaras:v4"})["asr_rtf"] == 0.25
    assert score_record({**base, "asr_model": "ARTPARK-IISc/SraVaani-1.0"})["asr_rtf"] == 0.25

    # micro weights by length, macro does not: 1 error in 1 word + 1 in 9 words
    a = _agg([_blank(wer_errors=1, ref_words=1, norm_wer=1.0, wer_sub=1),
              _blank(wer_errors=1, ref_words=9, norm_wer=1 / 9, wer_sub=1)])
    assert abs(a["norm_wer_micro"] - 0.2) < 1e-9, a["norm_wer_micro"]
    assert abs(a["norm_wer_macro"] - 0.5555555) < 1e-5, a["norm_wer_macro"]
    assert a["sub"] == 2 and a["n"] == 2

    # pooled RTF uses total time over total audio, not a mean of ratios
    b = _agg([_blank(audio_duration_s=1.0, asr_latency_s=1.0, asr_rtf=1.0),
              _blank(audio_duration_s=9.0, asr_latency_s=1.0, asr_rtf=1 / 9)])
    assert abs(b["rtf_pooled"] - 0.2) < 1e-9, b["rtf_pooled"]
    print("evaluate self-check ok")


if __name__ == "__main__":
    _self_check()
