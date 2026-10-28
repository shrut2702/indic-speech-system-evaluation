"""Build a whole noisy bench condition at one SNR, clip by clip.

Reproduces exactly what add_noise.py --out does per clip (same seed derivation, so
the same bytes), just in-process instead of 125 subprocesses.

    python research/make_noisy_set.py --snr 0 --noise F:/noise_demand/PCAFETER

Writes artifacts/bench/noisy_<snr>db/<lang>/<category>/<item>.wav at the source's
own 24 kHz. Seeded per OUTPUT path, matching how noisy_5db was built - so a given
clip gets a different noise window in each condition. Across 125 items that adds
variance to a 5dB-vs-0dB comparison but no bias; it is not a controlled per-item pair.
"""
import argparse, json, sys, zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from add_noise import as_float, load_noise, mix, read_wav, snr_of, to_mono, write_wav

ROOT = Path(__file__).resolve().parent.parent
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def one(src, out, noise_dir, snr, seed):
    raw, in_rate, ch = read_wav(src)
    raw = to_mono(raw, in_rate, ch, in_rate)
    clean = as_float(raw)
    # identical to add_noise.py: the seed is derived from the output path
    rng = np.random.default_rng([seed, zlib.crc32(str(out).encode())])
    noise, noise_name = load_noise(noise_dir, len(clean), rng, in_rate)
    noisy, scaled = mix(clean, noise, snr)
    got = snr_of(clean, scaled)
    assert abs(got - snr) < 0.5, f"{src.name}: wanted {snr} dB, got {got:.2f}"
    peak = np.abs(noisy).max()
    clipped = peak > 1.0
    if clipped:
        noisy /= peak            # scaling signal and noise together keeps the SNR
    out.parent.mkdir(parents=True, exist_ok=True)
    write_wav(out, (noisy * 32767).astype(np.int16).tobytes(), in_rate)
    return got, noise_name, clipped


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snr", type=float, required=True)
    ap.add_argument("--noise", required=True, help="dir of noise .wav, or one file")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--src-cond", default="clean", help="condition to mix noise into")
    ap.add_argument("--name", help="output condition dir name (default noisy_<snr>db)")
    a = ap.parse_args()

    bench = ROOT / "artifacts" / "bench"
    cond = a.name or f"noisy_{a.snr:g}db"
    langs = sorted(p.stem.split("_", 1)[1] for p in bench.glob("items_*.jsonl"))
    n = clipped = 0
    worst = 0.0
    for lang in langs:
        items = [json.loads(l) for l in
                 (bench / f"items_{lang}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        for it in items:
            rel = Path(lang) / it["category"] / f"{it['item_id']}.wav"
            src = bench / a.src_cond / rel
            if not src.exists():
                print(f"  missing {src}")
                continue
            got, noise_name, cl = one(src, bench / cond / rel, a.noise, a.snr, a.seed)
            n += 1
            clipped += cl
            worst = max(worst, abs(got - a.snr))
            if n % 25 == 0:
                print(f"  [{n}] {lang} {it['item_id']} snr {got:.2f} noise {noise_name}")
    print(f"\n{n} clips -> {bench / cond}")
    print(f"max SNR error {worst:.3f} dB   peak-normalized {clipped}/{n}")


if __name__ == "__main__":
    main()
