"""Synthesize bench clips with Sarvam Bulbul v4 Flash.

    python research/sarvam_tts.py --item dis_01          # one item, all languages
    python research/sarvam_tts.py                        # everything

Writes artifacts/bench/clean/<lang>/<category>/<item_id>.wav at 24 kHz, and appends
the speaker actually used to artifacts/bench/tts_manifest.jsonl.
"""
import argparse, base64, json, os, random, sys, time, urllib.request, urllib.error, wave, contextlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
URL = "https://api.sarvam.ai/text-to-speech"
MODEL = "bulbul:v4-flash"
BCP47 = {"en": "en-IN", "hi": "hi-IN", "gu": "gu-IN", "ta": "ta-IN", "pa": "pa-IN"}

# Gender is NOT documented per voice; these groupings are inferred from the persona
# names. Gujarati has 3 voices total, Tamil and Punjabi 4 each, all apparently male.
VOICES = {
    "hi": {"m": ["aditya_hi_conversational", "rahul_hi_conversational", "amit_hi_conversational",
                 "kabir_hi_conversational", "sumit_hi_conversational", "tarun_hi_conversational",
                 "mohit_hi_conversational", "ashutosh_hi_conversational"],
           "f": ["ritu_hi_conversation", "kavya_hi_conversational", "shreya_hi_conversational",
                 "roopa_hi_conversational", "sanchita_hi_assistant", "ishita_hi_social",
                 "simran_hi_conversation", "shalini_hi_social"]},
    "en": {"m": ["dev_en_conversational", "deven_en_conversation", "rohan_en_recovery",
                 "ojas_en_social", "kalpit_en_edtech", "shubh_en_narration", "varun_en_ads",
                 "ashwin_en_sports"],
           "f": ["aparna_en_companion", "neha_en_customer", "ritu_en_edtech",
                 "roopa_en_conversational", "sanchita_en_companion", "simran_en_conversation",
                 "shalini_en_customer", "zarina_en_conversation"]},
    "gu": {"m": ["bhavik_gu_conversation"],
           "f": ["pooja_gu_conversational", "pooja_gu_customer"]},
    "ta": {"m": ["aravind_ta_ads", "gokul_ta_narration", "vijay_ta_narration",
                 "aravind_ta_suspense"], "f": []},
    "pa": {"m": ["anand_pa_conversation", "anand_pa_customer", "jaspal_pa_banking",
                 "harpreet_pa_narration"], "f": []},
}

# items whose first-person grammar is masculine, so the voice must be male
MALE_ONLY = {"lng_01", "lng_04"}


def api_key():
    if os.environ.get("SARVAM_API_KEY"):
        return os.environ["SARVAM_API_KEY"]
    f = ROOT / ".env"
    for line in f.read_text(encoding="utf-8").splitlines() if f.exists() else []:
        k, _, v = line.partition("=")
        if k.strip() == "SARVAM_API_KEY":
            return v.strip().strip("\"'")
    sys.exit("SARVAM_API_KEY not found in environment or .env")


def manifest(lang):
    f = ROOT / "artifacts" / "bench" / f"items_{lang}.jsonl"
    return [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]


def assign_speakers(lang, items, seed):
    """Round-robin over a seeded shuffle, so use is even and deterministic.

    shortcut: gu/ta/pa have 1-4 voices for 25 items, so speakers repeat within a
    language - unavoidable until Sarvam ships more Indic voices.
    """
    rng = random.Random(f"{seed}:{lang}")
    pools = {}
    for g in ("m", "f"):
        v = list(VOICES[lang][g])
        rng.shuffle(v)
        pools[g] = v
    out, counters = {}, {"m": 0, "f": 0}
    for it in items:
        g = "m" if it["item_id"] in MALE_ONLY or not pools["f"] else rng.choice("mf")
        if not pools[g]:
            g = "m" if g == "f" else "f"
        out[it["item_id"]] = pools[g][counters[g] % len(pools[g])]
        counters[g] += 1
    return out


def synthesize(key, text, lang, speaker):
    body = {"text": text, "language_code": BCP47[lang], "speaker": speaker, "model": MODEL,
            "speech_sample_rate": 24000, "output_audio_codec": "wav", "pace": 1.0}
    for field in ("language_code", "target_language_code"):  # docs disagree on the name
        payload = dict(body)
        if field != "language_code":
            payload[field] = payload.pop("language_code")
        req = urllib.request.Request(URL, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json",
                                              "api-subscription-key": key})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                if field != "language_code":
                    print(f"    (api accepted {field}, not language_code)")
                return base64.b64decode(json.loads(r.read())["audios"][0])
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:300]
            if e.code in (400, 422) and field == "language_code":
                continue  # try the other field name
            raise RuntimeError(f"HTTP {e.code} for {lang}/{speaker}: {detail}")
    raise RuntimeError("both language_code and target_language_code rejected")


def synthesize_retry(key, text, lang, speaker, tries=3):
    for i in range(tries):
        try:
            return synthesize(key, text, lang, speaker)
        except Exception as e:
            print(f"    attempt {i + 1}/{tries} failed: {e}")
            time.sleep(2 * (i + 1))
    return None


def wav_info(path):
    with contextlib.closing(wave.open(str(path))) as w:
        return w.getframerate(), w.getnchannels(), w.getnframes() / w.getframerate()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--item", help="only this item_id (across all languages)")
    ap.add_argument("--langs", nargs="+", default=["en", "hi", "gu", "ta", "pa"])
    ap.add_argument("--seed", default="vaani")
    ap.add_argument("--speaker", help="force this voice; writes to artifacts/trial/ as a probe")
    ap.add_argument("--to-bench", action="store_true",
                    help="with --speaker: write into the bench, not the probe folder")
    ap.add_argument("--skip-existing", action="store_true", help="leave clips already on disk alone")
    args = ap.parse_args()

    key = api_key()
    log = ROOT / "artifacts" / "bench" / "tts_manifest.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    n, failed = 0, []
    with log.open("a", encoding="utf-8") as fh:
        for lang in args.langs:
            items = manifest(lang)
            speakers = assign_speakers(lang, items, args.seed)
            for it in items:
                if args.item and it["item_id"] != args.item:
                    continue
                spk = args.speaker or speakers[it["item_id"]]
                probe = args.speaker and not args.to_bench
                out = (ROOT / "artifacts" / "trial" / f"probe_{lang}_{spk}.wav" if probe
                       else ROOT / "artifacts" / "bench" / "clean" / lang / it["category"] / f"{it['item_id']}.wav")
                if args.skip_existing and out.exists():
                    continue
                out.parent.mkdir(parents=True, exist_ok=True)
                t = time.perf_counter()
                audio = synthesize_retry(key, it["text"], lang, spk)
                if audio is None:
                    print(f"  FAILED {lang} {it['item_id']}")
                    failed.append(f"{lang}/{it['item_id']}")
                    continue
                out.write_bytes(audio)
                took = time.perf_counter() - t
                rate, ch, dur = wav_info(out)
                if probe:
                    print(f"  PROBE {lang} {spk:28} {rate}Hz/{ch}ch {dur:5.2f}s -> {out.name}")
                    n += 1
                    continue
                fh.write(json.dumps({"item_id": it["item_id"], "lang": lang,
                                     "category": it["category"], "speaker": spk,
                                     "model": MODEL, "sample_rate": rate, "pace": 1.0,
                                     "clip_path": str(out.relative_to(ROOT)).replace("\\", "/"),
                                     "duration_s": round(dur, 3),
                                     "synth_time_s": round(took, 2)}, ensure_ascii=False) + "\n")
                print(f"  {lang} {it['item_id']:7} {spk:28} {rate}Hz/{ch}ch {dur:5.2f}s  ({took:.1f}s)")
                n += 1
    print(f"\n{n} clips -> artifacts/bench/clean/<lang>/<category>/  | log: {log.name}")
    if failed:
        print(f"never succeeded ({len(failed)}): {', '.join(failed)}")


if __name__ == "__main__":
    main()
