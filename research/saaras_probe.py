"""Probe Sarvam Saaras STT in isolation: does it beat SraVaani on our clips?

Throwaway. Self-contained: reads the bench manifests as data, never imports
src/speechSys. Sends the bench clip as-is (24 kHz) - Saaras resamples server side,
so there is no 16 kHz step to match the pipeline's.

    python research/saaras_probe.py                 # 2 clips per language, clean
    python research/saaras_probe.py --n 25 --cond noisy_0db --save asr_only_noisy_0db_saaras_v4
    python research/saaras_probe.py --items ent_01 cs_03 --langs hi ta

With --save, records are appended as each clip completes and already-done items are
skipped on a rerun, so a rate limit or a crash never costs the calls already paid for.

Cost: Rs 30/hour of audio, billed per second rounded up. 125 clips ~ Rs 7.9.
"""
import argparse, json, math, os, sys, time, uuid, urllib.request, urllib.error, wave, contextlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
URL = "https://api.sarvam.ai/speech-to-text"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def api_key():
    if os.environ.get("SARVAM_API_KEY"):
        return os.environ["SARVAM_API_KEY"]
    f = ROOT / ".env"
    for line in f.read_text(encoding="utf-8").splitlines() if f.exists() else []:
        k, _, v = line.partition("=")
        if k.strip() == "SARVAM_API_KEY":
            return v.strip().strip("\"'")
    sys.exit("SARVAM_API_KEY not found in environment or .env")


def duration(path):
    with contextlib.closing(wave.open(str(path))) as w:
        return w.getnframes() / w.getframerate()


def multipart(fields, filename, blob):
    b = "----" + uuid.uuid4().hex
    out = b""
    for k, v in fields.items():
        out += (f"--{b}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n").encode()
    out += (f"--{b}\r\nContent-Disposition: form-data; name=\"file\"; "
            f"filename=\"{filename}\"\r\nContent-Type: audio/wav\r\n\r\n").encode()
    out += blob + b"\r\n" + f"--{b}--\r\n".encode()
    return b, out


def transcribe(clip, key, model, mode, retries=6):
    """(transcript, detected_lang, latency_s). Backs off on 429 and 5xx."""
    blob = clip.read_bytes()
    for attempt in range(retries):
        b, body = multipart({"model": model, "mode": mode}, clip.name, blob)
        req = urllib.request.Request(URL, data=body, headers={
            "api-subscription-key": key,
            "Content-Type": f"multipart/form-data; boundary={b}"})
        t = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                d = json.loads(r.read())
            return (d.get("transcript") or "").strip(), d.get("language_code"), time.perf_counter() - t
        except urllib.error.HTTPError as e:
            retryable = e.code == 429 or e.code >= 500
            if retryable and attempt < retries - 1:
                wait = 2 ** attempt
                print(f"   HTTP {e.code}, retrying in {wait}s")
                time.sleep(wait)
                continue
            raise SystemExit(f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:300]}")
    raise SystemExit("unreachable")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=2, help="clips per language")
    ap.add_argument("--langs", nargs="*")
    ap.add_argument("--items", nargs="*", help="specific item_ids, overrides --n")
    ap.add_argument("--cond", default="clean")
    ap.add_argument("--model", default="saaras:v4")
    ap.add_argument("--mode", default="transcribe")
    ap.add_argument("--gap", type=float, default=0.3, help="seconds between requests")
    ap.add_argument("--save", help="append pipeline-shaped records to "
                                   "artifacts/runs/<save>.jsonl for main.py eval")
    a = ap.parse_args()

    bench = ROOT / "artifacts" / "bench"
    langs = a.langs or sorted(p.stem.split("_", 1)[1] for p in bench.glob("items_*.jsonl"))
    # what SraVaani produced, for a side-by-side; absent is fine
    prev = {}
    for cand in (f"pipeline_{a.cond}_sravaani1.0_sarvam105b.jsonl",
                 f"asr_only_{a.cond}_sravaani1.0.jsonl"):
        run = ROOT / "artifacts/runs" / cand
        if run.exists():
            for line in run.read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                prev[(r["lang"], r["item_id"])] = r["asr_hypothesis"]
            break

    out = done = None
    if a.save:
        out = ROOT / "artifacts" / "runs" / f"{a.save}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        done = set()
        if out.exists():
            for line in out.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    done.add((r["lang"], r["item_id"]))

    picked = []
    for lang in langs:
        items = [json.loads(l) for l in
                 (bench / f"items_{lang}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        if a.items:
            items = [i for i in items if i["item_id"] in a.items]
        else:
            items = items[:a.n]
        for it in items:
            clip = bench / a.cond / lang / it["category"] / f"{it['item_id']}.wav"
            if not clip.exists():
                continue
            if done is not None and (lang, it["item_id"]) in done:
                continue
            picked.append((lang, it, clip))

    if not picked:
        sys.exit("nothing to do (all items already saved, or no clips matched)")
    audio = sum(duration(c) for _, _, c in picked)
    billed = sum(math.ceil(duration(c)) for _, _, c in picked)
    skipped = len(done) if done else 0
    print(f"{len(picked)} clips  {audio:.1f}s audio  billed {billed}s "
          f"~ Rs {billed / 3600 * 30:.2f}   model={a.model} mode={a.mode} cond={a.cond}"
          + (f"   (resuming, {skipped} already done)" if skipped else "") + "\n")

    key = api_key()
    fh = out.open("a", encoding="utf-8") if out else None
    over = 0
    try:
        for i, (lang, it, clip) in enumerate(picked, 1):
            d = duration(clip)
            if d > 30:
                print(f"{lang} {it['item_id']:7} SKIP {d:.1f}s > 30s sync limit")
                over += 1
                continue
            txt, det, dt = transcribe(clip, key, a.model, a.mode)
            print(f"[{i}/{len(picked)}] {lang} {it['item_id']:7} {it['category']:14} "
                  f"{d:5.1f}s  api {dt:5.2f}s  rtf {dt / d:.3f}  detected={det}")
            print(f"   ref     : {it['reference_transcript'][:100]}")
            print(f"   saaras  : {txt[:100]}")
            if (lang, it["item_id"]) in prev:
                print(f"   sravaani: {prev[(lang, it['item_id'])][:100]}")
            print()
            if fh:
                fh.write(json.dumps({
                    "run_id": a.save,
                    "item_id": it["item_id"], "lang": lang, "category": it["category"],
                    "acoustic_condition": a.cond,
                    "clip_path": str(clip.relative_to(ROOT)).replace("\\", "/"),
                    "audio_duration_s": round(d, 3),
                    "reference_transcript": it["reference_transcript"],
                    "reference_answer": it["reference_answer"],
                    "answer_aliases": it.get("answer_aliases") or [],
                    "reference_entities": it.get("reference_entities") or [],
                    "trap_answer": it.get("trap_answer"),
                    "asr_hypothesis": txt,
                    "asr_model": a.model,
                    "asr_params": {"mode": a.mode, "detected_language": det},
                    # hosted API: network round-trip, not compute. Not comparable to
                    # the local GPU runs, so never report it as RTF.
                    "asr_latency_s": round(dt, 4),
                    "machine": "sarvam hosted API",
                }, ensure_ascii=False) + "\n")
                fh.flush()
            if a.gap:
                time.sleep(a.gap)
    finally:
        if fh:
            fh.close()
    if over:
        print(f"{over} clip(s) over the 30s sync limit - those need the Batch API")
    if out:
        n = sum(1 for l in out.read_text(encoding="utf-8").splitlines() if l.strip())
        print(f"{n} records total -> {out}")


if __name__ == "__main__":
    main()
