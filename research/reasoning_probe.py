"""Does disabling sarvam-105b's reasoning make it deterministic (and cheaper)?

Throwaway. Repeats the same questions under several settings and reports how often
the answer comes back byte-identical, plus tokens and latency.

    python research/reasoning_probe.py --reps 3
"""
import argparse, json, os, sys, time, urllib.request, urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
URL = "https://api.sarvam.ai/v1/chat/completions"
SYSTEM = ("You are a voice assistant for Indian users. Answer in the SAME language and "
          "script as the question. Reply with only the answer itself - a name, a number, "
          "a phrase. No explanation, no preamble, no restating the question.")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def key():
    if os.environ.get("SARVAM_API_KEY"):
        return os.environ["SARVAM_API_KEY"]
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        k, _, v = line.partition("=")
        if k.strip() == "SARVAM_API_KEY":
            return v.strip().strip("\"'")
    sys.exit("no SARVAM_API_KEY")


def ask(q, k, extra):
    body = {"model": "sarvam-105b", "temperature": 0,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": q}]}
    body.update(extra)
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={
        "Content-Type": "application/json", "api-subscription-key": k,
        "Authorization": f"Bearer {k}"})
    t = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            d = json.loads(r.read())
    except urllib.error.HTTPError as e:
        return f"<HTTP {e.code}: {e.read().decode('utf-8','replace')[:120]}>", 0, 0
    msg = (d["choices"][0].get("message") or {}).get("content") or ""
    return msg.strip(), (d.get("usage") or {}).get("completion_tokens", 0), time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--items", nargs="*", default=["ent_04", "lng_02", "nor_04", "cs_03"])
    ap.add_argument("--lang", default="hi")
    a = ap.parse_args()

    bench = ROOT / "artifacts" / "bench"
    items = {i["item_id"]: i for i in
             (json.loads(l) for l in
              (bench / f"items_{a.lang}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())}
    qs = [(k, items[k]["reference_transcript"], items[k]["reference_answer"])
          for k in a.items if k in items]

    settings = [("reasoning low (default)", {}),
                ("reasoning off", {"reasoning_effort": None}),
                ("reasoning off + max_tokens 48",
                 {"reasoning_effort": None, "max_tokens": 48})]
    kk = key()
    for name, extra in settings:
        print(f"\n=== {name}")
        ident = tok = 0
        secs = 0.0
        for iid, q, gold in qs:
            outs = []
            for _ in range(a.reps):
                txt, n, dt = ask(q, kk, extra)
                outs.append(txt)
                tok += n
                secs += dt
            same = len(set(outs)) == 1
            ident += same
            print(f"  {iid:7} identical={same}  gold={gold[:18]!r}")
            for o in (outs if not same else outs[:1]):
                print(f"      {o[:70]!r}")
        print(f"  -> {ident}/{len(qs)} items byte-identical across {a.reps} reps, "
              f"{tok} tokens, {secs:.1f}s")


if __name__ == "__main__":
    main()
