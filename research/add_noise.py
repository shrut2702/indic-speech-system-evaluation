"""Make noisy variants of a clean clip at target SNRs.

Noise comes from a real corpus (DEMAND PCAFETER = cafeteria babble):
    python research/add_noise.py clip.wav --noise F:/noise_demand/PCAFETER/ch01.wav
    python research/add_noise.py clip.wav --noise F:/noise_demand/PCAFETER --out bench/x.wav

Without --out: probe mode, resamples to 16 kHz and writes beside the source.
With --out: bench mode, keeps the clip's own rate (24 kHz) and writes just that file.
"""
import argparse, audioop, wave, contextlib, sys, zlib
from pathlib import Path
import numpy as np

def read_wav(path):
    with contextlib.closing(wave.open(str(path))) as w:
        if w.getsampwidth() != 2:
            sys.exit(f"{path}: need 16-bit PCM, got {w.getsampwidth() * 8}-bit")
        return w.readframes(w.getnframes()), w.getframerate(), w.getnchannels()

def write_wav(path, raw, rate):
    with contextlib.closing(wave.open(str(path), "wb")) as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes(raw)

def to_mono(raw, rate, channels, target):
    if channels > 1:
        raw = audioop.tomono(raw, 2, 0.5, 0.5)
    if rate != target:
        # shortcut: audioop.ratecv is a basic resampler; only the noise bed goes
        # through it in bench mode, so it never touches the speech signal.
        raw, _ = audioop.ratecv(raw, 2, 1, rate, target, None)
    return raw

def as_float(raw):
    return np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0

def load_noise(path, n, rng, rate):
    """A random n-sample window of real noise, mono at `rate`. Tiles if too short."""
    p = Path(path)
    if p.is_dir():
        files = sorted(p.glob("*.wav"))
        if not files:
            sys.exit(f"{p}: no .wav files")
        p = files[rng.integers(len(files))]
    noise = as_float(to_mono(*read_wav(p), rate))
    if len(noise) < n:
        noise = np.tile(noise, -(-n // len(noise)))
    start = rng.integers(len(noise) - n + 1)
    return noise[start:start + n], p.name

def mix(clean, noise, snr_db):
    """Scales noise to hit snr_db against clean. Returns (mixed, scaled_noise)."""
    scaled = noise * np.sqrt(np.mean(clean**2) / (np.mean(noise**2) * 10 ** (snr_db / 10)))
    return clean + scaled, scaled

def snr_of(clean, noise):
    return 10 * np.log10(np.mean(clean**2) / np.mean(noise**2))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--noise", required=True, help="noise .wav, or a dir of them")
    ap.add_argument("--snr", type=float, nargs="+", default=[5])
    ap.add_argument("--seed", type=int, default=0, help="fixed so runs are reproducible")
    ap.add_argument("--out", help="write the single noisy clip here, at the source's own rate")
    args = ap.parse_args()
    if args.out and len(args.snr) != 1:
        sys.exit("--out takes exactly one --snr")

    src = Path(args.wav)
    raw, in_rate, ch = read_wav(src)
    rate = in_rate if args.out else 16000
    raw = to_mono(raw, in_rate, ch, rate)
    clean = as_float(raw)
    print(f"{src.name}: -> {rate}Hz/1ch, {len(clean) / rate:.2f}s")

    if not args.out:
        out = src.with_name(f"{src.stem}_clean16k.wav")
        write_wav(out, raw, rate)
        print(f"  wrote {out.name}")

    # seed per output clip, so every clip gets a different noise file + window
    rng = np.random.default_rng([args.seed, zlib.crc32(str(args.out or src).encode())])
    noise, noise_name = load_noise(args.noise, len(clean), rng, rate)
    print(f"  noise: {noise_name}")

    for snr in args.snr:
        noisy, scaled = mix(clean, noise, snr)
        got = snr_of(clean, scaled)
        assert abs(got - snr) < 0.5, f"SNR off: wanted {snr}, got {got:.2f}"
        peak = np.abs(noisy).max()
        if peak > 1.0:
            noisy /= peak  # scaling signal and noise together keeps the SNR
            print(f"  snr={snr:g}: peak {peak:.2f} -> normalized to avoid clipping")
        out = Path(args.out) if args.out else src.with_name(f"{src.stem}_noisy{snr:g}db.wav")
        out.parent.mkdir(parents=True, exist_ok=True)
        write_wav(out, (noisy * 32767).astype(np.int16).tobytes(), rate)
        print(f"  wrote {out.name}  (measured SNR {got:.2f} dB)")

if __name__ == "__main__":
    main()
