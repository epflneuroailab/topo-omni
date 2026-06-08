"""
Generate a bank of pure-tone exemplars for tonotopy mapping in Topo-Omni.

For each center frequency, renders N exemplars that vary nuisance parameters --
duration, level, ramp envelope, onset/offset silence, starting phase, and
optional tone-in-noise -- while holding center frequency fixed. These exemplars
are the within-frequency replicates needed for a valid per-unit tuning test
(one-way ANOVA across frequency) instead of pseudoreplicated time chunks.

Design note: the N exemplar parameter sets are drawn ONCE and reused at every
frequency, so the nuisance variation is matched across conditions and frequency
is the only systematic difference. exemplar_idx is therefore a crossed factor --
run a plain one-way ANOVA on frequency, or a 2-way (frequency x exemplar) ANOVA
to pull the exemplar main effect out of the error term for more power.

Caveat: if the Qwen2.5-Omni audio feature extractor normalizes each clip, the
absolute-level manipulation may be attenuated downstream. Duration, envelope,
onset position, phase, and especially tone-in-noise survive normalization and
do the real work of generating within-frequency variance.

Output: <out_dir>/freqXX_<hz>Hz_exYY.wav  +  <out_dir>/manifest.csv
"""

import os
import csv
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from dotenv import load_dotenv
load_dotenv()

STIMULI_DIR = os.getenv("STIMULI_DIR", "./stimuli")


# ----- noise -----------------------------------------------------------------

def _white_noise(n, rng):
    x = rng.standard_normal(n)
    return x / (x.std() + 1e-12)


def _pink_noise(n, rng):
    """1/f noise via spectral shaping of white noise."""
    white = rng.standard_normal(n)
    X = np.fft.rfft(white)
    f = np.fft.rfftfreq(n)
    f[0] = f[1] if len(f) > 1 else 1.0          # avoid div-by-zero at DC
    pink = np.fft.irfft(X / np.sqrt(f), n=n)
    return pink / (pink.std() + 1e-12)


def _make_noise(kind, n, rng):
    return _pink_noise(n, rng) if kind == "pink" else _white_noise(n, rng)


# ----- envelope --------------------------------------------------------------

def _ramp(shape, n):
    """Rising ramp of length n in [0, 1]; the offset ramp is its reverse."""
    if n <= 0:
        return np.ones(0)
    t = np.linspace(0.0, 1.0, n)
    if shape == "cosine":                        # raised-cosine, the auditory default
        return 0.5 * (1.0 - np.cos(np.pi * t))
    if shape == "exponential":
        return (np.exp(4.0 * t) - 1.0) / (np.exp(4.0) - 1.0)
    return t                                     # linear


# ----- single exemplar -------------------------------------------------------

def render_exemplar(freq_hz, sr, p, ref_rms=0.05):
    """Render one tone exemplar. `p` is a parameter dict from sample_params()."""
    n_tone = int(round(p["tone_duration_s"] * sr))
    t = np.arange(n_tone) / sr
    tone = np.sin(2.0 * np.pi * freq_hz * t + p["phase_rad"])

    # onset/offset ramp
    ramp_n = min(int(round(p["ramp_ms"] / 1000.0 * sr)), n_tone // 2)
    if ramp_n > 0:
        up = _ramp(p["ramp_shape"], ramp_n)
        env = np.ones(n_tone)
        env[:ramp_n] = up
        env[-ramp_n:] = up[::-1]
        tone *= env

    # level: normalize to reference RMS, then apply the per-exemplar dB offset
    rms = np.sqrt(np.mean(tone ** 2)) + 1e-12
    tone = tone / rms * ref_rms * 10.0 ** (p["level_db"] / 20.0)

    # optional tone-in-noise at a fixed SNR; noise spans the tone only.
    # seeded from p["seed"] so the noise realization is identical across
    # frequencies for a given exemplar (keeps the design matched).
    if p["noise_type"] is not None:
        rng = np.random.default_rng(p["seed"])
        noise = _make_noise(p["noise_type"], n_tone, rng)
        tone_rms = np.sqrt(np.mean(tone ** 2)) + 1e-12
        tone = tone + noise * (tone_rms / 10.0 ** (p["snr_db"] / 20.0))

    # leading / trailing silence so tone onset time within the clip varies
    lead = np.zeros(int(round(p["lead_silence_s"] * sr)))
    trail = np.zeros(int(round(p["trail_silence_s"] * sr)))
    x = np.concatenate([lead, tone, trail]).astype(np.float32)
    min_samples = int(round(2.0 * sr))
    if len(x) < min_samples:
        x = np.concatenate([x, np.zeros(min_samples - len(x), dtype=np.float32)])

    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 0.99:
        print(f"  warning: clipping guard fired (peak={peak:.2f}); lower ref_rms")
        x = x / peak * 0.99
    return x


# ----- parameter sampling ----------------------------------------------------

def sample_params(rng, idx):
    """Draw one exemplar's nuisance parameters (reused across all frequencies).

    Tone duration + silences may be under 2 s; render_exemplar pads with trailing
    silence so every clip is at least 2 seconds.
    """
    noisy = rng.random() < 0.5
    return {
        "exemplar_idx": idx,
        "tone_duration_s": float(rng.uniform(0.4, 1.4)),
        "lead_silence_s": float(rng.uniform(0.0, 0.25)),
        "trail_silence_s": float(rng.uniform(0.0, 0.25)),
        "level_db": float(rng.uniform(-12.0, 6.0)),
        "ramp_shape": str(rng.choice(["cosine", "linear", "exponential"])),
        "ramp_ms": float(rng.uniform(5.0, 50.0)),
        "phase_rad": float(rng.uniform(0.0, 2.0 * np.pi)),
        "noise_type": str(rng.choice(["white", "pink"])) if noisy else None,
        "snr_db": float(rng.uniform(0.0, 20.0)) if noisy else float("nan"),
        "seed": int(rng.integers(0, 2 ** 31)),
    }


# ----- bank generation -------------------------------------------------------

def generate_tone_bank(freqs_hz, out_dir, n_exemplars=16, sr=16000,
                        ref_rms=0.05, master_seed=0):
    """Render n_exemplars per frequency, write wavs + a manifest.csv.

    The manifest is what you group on downstream: load it, group rows by
    `freq_idx`, and the exemplars within a group are your replicates.
    """
    freqs_hz = np.asarray(freqs_hz, dtype=float)
    if np.any(freqs_hz > 0.45 * sr):
        print(f"warning: frequencies above {0.45 * sr:.0f} Hz are close to "
              f"Nyquist ({sr / 2:.0f} Hz) at sr={sr}; raise sr or lower max freq.")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # draw the exemplar parameter sets ONCE, reuse at every frequency
    rng = np.random.default_rng(master_seed)
    exemplar_params = [sample_params(rng, i) for i in range(n_exemplars)]

    rows = []
    for f_idx, f_hz in enumerate(freqs_hz):
        for p in exemplar_params:
            x = render_exemplar(f_hz, sr, p, ref_rms=ref_rms)
            fname = f"freq{f_idx:02d}_{f_hz:04.0f}Hz_ex{p['exemplar_idx']:02d}.wav"
            fpath = out_dir / fname
            wavfile.write(fpath, sr, (x * 32767).astype(np.int16))
            rows.append({
                "filepath": str(fpath),
                "freq_idx": f_idx,
                "freq_hz": round(float(f_hz), 3),
                "total_duration_s": round(len(x) / sr, 4),
                "sr": sr,
                **{k: p[k] for k in (
                    "exemplar_idx", "tone_duration_s", "lead_silence_s",
                    "trail_silence_s", "level_db", "ramp_shape", "ramp_ms",
                    "phase_rad", "noise_type", "snr_db", "seed")},
            })

    manifest = out_dir / "manifest.csv"
    with open(manifest, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} wav files ({len(freqs_hz)} freqs x {n_exemplars} "
          f"exemplars) + manifest -> {out_dir}")
    return manifest


if __name__ == "__main__":
    # 30 log-spaced frequencies; capped at 7 kHz to stay clear of Nyquist at 16 kHz.
    # For content up to 8 kHz, generate at sr=22050+ and let the processor resample.
    freqs_hz = np.logspace(np.log10(100), np.log10(7000), 30)
    generate_tone_bank(freqs_hz, out_dir=os.path.join(STIMULI_DIR, "tone_bank"), n_exemplars=16,
                       sr=16000, master_seed=0)