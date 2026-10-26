"""Run Meta Omnilingual ASR in isolation: check env, resolve lang codes, time a transcribe.

Needs WSL2 or Linux - fairseq2 has no native Windows build.
    pip install omnilingual-asr
"""
import argparse, time, wave, contextlib, sys

def duration(path):
    with contextlib.closing(wave.open(path)) as w:
        return w.getnframes() / w.getframerate()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", nargs="*", help="wav paths")
    ap.add_argument("--model", default="omniASR_CTC_300M")
    ap.add_argument("--lang", default="hin_Deva")
    ap.add_argument("--grep", help="print supported langs matching this and exit")
    args = ap.parse_args()

    import torch
    print(f"torch {torch.__version__} cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        print(f"gpu {torch.cuda.get_device_name()} sm{cap[0]}{cap[1]} "
              f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB "
              f"bf16={'yes' if cap[0] >= 8 else 'NO - use fp16'}")

    from omnilingual_asr.models.wav2vec2_llama.lang_ids import supported_langs
    print(f"{len(supported_langs)} languages supported")
    if args.grep:
        print([l for l in supported_langs if args.grep.lower() in l.lower()])
        return
    if args.lang not in supported_langs:
        sys.exit(f"{args.lang!r} not supported; try --grep hin")
    if not args.audio:
        sys.exit("no audio given")

    from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

    t = time.perf_counter()
    pipeline = ASRInferencePipeline(model_card=args.model)
    print(f"load {time.perf_counter() - t:.1f}s")

    t = time.perf_counter()
    out = pipeline.transcribe(args.audio, lang=[args.lang] * len(args.audio), batch_size=1)
    elapsed = time.perf_counter() - t

    audio_s = sum(duration(p) for p in args.audio)
    print(f"transcribe {elapsed:.2f}s for {audio_s:.2f}s audio  rtf={elapsed / audio_s:.3f}")
    for p, text in zip(args.audio, out):
        print(f"\n{p}\n  {text}")

if __name__ == "__main__":
    main()
