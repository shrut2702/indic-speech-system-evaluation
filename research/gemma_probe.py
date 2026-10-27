"""Run Gemma 3 4B (4-bit) in isolation: load cost, VRAM, tok/s, and a text-only
answer-quality check against the bench references.

Text-only on purpose: feeds the manifest `text`, so the LLM is isolated from the ASR.

    python research/gemma_probe.py --bench
"""
import argparse, json, os, sys, time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("HF_HOME", "F:/hf_cache")

REPO = "unsloth/gemma-3-4b-it-bnb-4bit"
ROOT = Path(__file__).resolve().parent.parent
LANGS = ["en", "hi", "gu", "ta", "pa"]
ITEMS = ["ent_04", "nor_01", "cs_01", "lng_01"]

SYS = ("You are a voice assistant for Indian users. Answer in the SAME language and "
       "script as the question. Reply with only the answer itself - a name, a number, "
       "a phrase. No explanation, no preamble, no restating the question.")


def env_token():
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    f = ROOT / ".env"
    for line in f.read_text(encoding="utf-8").splitlines() if f.exists() else []:
        k, _, v = line.partition("=")
        if k.strip() == "HF_TOKEN":
            return v.strip().strip("\"'")
    return None


def manifest(lang):
    f = ROOT / "artifacts" / "bench" / f"items_{lang}.jsonl"
    rows = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
    return {r["item_id"]: r for r in rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true", help="run the 4-item x 5-lang check")
    ap.add_argument("--max-new", type=int, default=48)
    args = ap.parse_args()

    import torch
    from transformers import (AutoConfig, AutoTokenizer, AutoModelForCausalLM,
                              BitsAndBytesConfig)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"torch {torch.__version__} device={dev} repo={REPO}")

    t = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(REPO, token=env_token())
    # the checkpoint ships bnb_4bit_compute_dtype=bfloat16; Turing sm75 has no bf16
    q = dict(AutoConfig.from_pretrained(REPO, token=env_token()).quantization_config)
    q["bnb_4bit_compute_dtype"] = "float16"
    # transformers pre-reserves the whole model as one block; WDDM cannot hand out
    # 2.95 GiB contiguous on a 6 GB display card. It is only an optimisation.
    import transformers.modeling_utils as _mu
    _mu.caching_allocator_warmup = lambda *a, **k: None

    model = AutoModelForCausalLM.from_pretrained(
        REPO, device_map=dev, dtype=torch.float16, token=env_token(),
        quantization_config=BitsAndBytesConfig(
            **{k: v for k, v in q.items() if not k.startswith("_")})).eval()
    load_s = time.perf_counter() - t
    print(f"load {load_s:.1f}s  vram_alloc={torch.cuda.memory_allocated()/1e9:.2f}GB "
          f"reserved={torch.cuda.memory_reserved()/1e9:.2f}GB")

    def ask(q, max_new=args.max_new):
        msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": q}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True).to(dev)
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(**ids, max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        dt = time.perf_counter() - t0
        new = out[0][ids["input_ids"].shape[1]:]
        return tok.decode(new, skip_special_tokens=True).strip(), dt, len(new)

    txt, dt, n = ask("Say hello.")  # warm up
    print(f"warmup: {dt:.2f}s {n}tok -> {txt!r}")

    txt, dt, n = ask("List five Indian cities.", max_new=64)
    print(f"speed: {n} tok in {dt:.2f}s = {n/dt:.1f} tok/s")
    print(f"peak vram {torch.cuda.max_memory_allocated()/1e9:.2f}GB")

    if not args.bench:
        return

    print("\n" + "=" * 70)
    for lang in LANGS:
        rows = manifest(lang)
        for iid in ITEMS:
            r = rows[iid]
            resp, dt, n = ask(r["text"])
            refs = [r["reference_answer"]] + list(r.get("answer_aliases") or [])
            hit = any(a.lower() in resp.lower() for a in refs if a)
            print(f"\n[{lang} {iid} {r['category']}] {dt:.2f}s {n}tok  "
                  f"{'MATCH' if hit else 'NO-MATCH'}")
            print(f"  ref : {r['reference_answer']}  aliases={r.get('answer_aliases')}")
            print(f"  resp: {resp!r}")


if __name__ == "__main__":
    main()
