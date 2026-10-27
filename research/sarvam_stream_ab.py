"""Does streaming actually cut Sarvam's latency, or was that caching?

Interleaves streaming and non-streaming calls on the same prompts so network drift
hits both arms equally, and uses a --nonce to defeat any server-side prompt cache.

    python research/sarvam_stream_ab.py --reps 3
"""
import argparse
import json
import math
import statistics
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from speechSys import llm  # noqa: E402

MODEL = "sarvam-105b"
# items not used in earlier probes, so nothing is warm
PROBES = [("hi", "nor_02"), ("hi", "cs_01"), ("hi", "lng_03"),
          ("ta", "nor_03"), ("ta", "cs_03"), ("en", "nor_04"),
          ("en", "lng_04"), ("pa", "nor_02")]


def percentile(v, p):
    if not v:
        return None
    s = sorted(v)
    return s[max(0, math.ceil(p / 100 * len(s)) - 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-new", type=int, default=512,
                    help="sarvam-105b reasons before answering; 48 can yield empty content")
    ap.add_argument("--nonce", action="store_true",
                    help="append a unique token to each prompt to defeat prompt caching")
    args = ap.parse_args()

    key = llm._sarvam_key(ROOT)
    qs = []
    for lang, iid in PROBES:
        rows = {json.loads(l)["item_id"]: json.loads(l)
                for l in (ROOT / "artifacts" / "bench" / f"items_{lang}.jsonl")
                .read_text(encoding="utf-8").splitlines() if l.strip()}
        qs.append((lang, iid, rows[iid]["reference_transcript"]))

    res = {True: {"total": [], "ttft": [], "tok": []}, False: {"total": [], "tok": []}}
    print(f"{'mode':10}{'lang':5}{'item':8}{'total_s':>9}{'ttft_s':>9}{'tok':>6}")
    for rep in range(args.reps):
        for lang, iid, q in qs:
            prompt = f"{q} [{uuid.uuid4().hex[:8]}]" if args.nonce else q
            for stream in (True, False):            # interleaved, same prompt
                txt, dt, n, ttft = llm.respond_api(MODEL, prompt, key, max_new_tokens=args.max_new, stream=stream)
                res[stream]["total"].append(dt)
                res[stream]["tok"].append(n)
                if stream:
                    res[stream]["ttft"].append(ttft if ttft is not None else dt)
                print(f"{'stream' if stream else 'blocking':10}{lang:5}{iid:8}"
                      f"{dt:9.3f}{(ttft if ttft is not None else float('nan')):9.3f}{n:>6}{'' if txt else '  EMPTY'}")
                time.sleep(0.2)                     # be polite to the endpoint

    print("\nSUMMARY")
    print(f"{'mode':10}{'n':>4}{'p50':>9}{'p95':>9}{'mean':>9}{'tok_p50':>9}")
    for stream, label in ((True, "stream"), (False, "blocking")):
        t = res[stream]["total"]
        print(f"{label:10}{len(t):>4}{percentile(t, 50):9.3f}{percentile(t, 95):9.3f}"
              f"{statistics.fmean(t):9.3f}{percentile(res[stream]['tok'], 50):>9}")
    tt = res[True]["ttft"]
    print(f"{'ttft':10}{len(tt):>4}{percentile(tt, 50):9.3f}{percentile(tt, 95):9.3f}"
          f"{statistics.fmean(tt):9.3f}")

    s50, b50 = percentile(res[True]["total"], 50), percentile(res[False]["total"], 50)
    print(f"\nstream p50 {s50:.3f}s vs blocking p50 {b50:.3f}s  "
          f"-> {'stream faster' if s50 < b50 else 'no gain'} by {abs(b50 - s50):.3f}s "
          f"({abs(b50 - s50) / b50 * 100:.1f}%)")
    print(f"ttft p50 {percentile(tt, 50):.3f}s = "
          f"{percentile(tt, 50) / s50 * 100:.0f}% of the streamed total "
          f"-> {'content arrives in one burst' if percentile(tt, 50) / s50 > 0.8 else 'genuinely incremental'}")


if __name__ == "__main__":
    main()
