#!/usr/bin/env python3
"""A/B voice-clone evaluation for reference-audio normalization strategies.

Generates paired outputs from the same text/reference using two preprocessing paths:
- legacy_match_default: Studio-like RMS/peak matching to default voice profile
- safe_peak_only: peak limiting only (identity-preserving default)

Then reports spectral centroid and HF energy (>4 kHz) deltas.

Run from repo root:
    python BayanSynthTTS/scripts/ab_normalization_eval.py \
        --text "مَرْحَباً بِكُمْ" \
        --ref-audio BayanSynthTTS/voices/default.wav
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def _active_audio_region(audio: np.ndarray) -> np.ndarray:
    if audio.size == 0:
        return audio
    peak = float(np.max(np.abs(audio)))
    if peak <= 1e-6:
        return audio
    gate = max(peak * 0.1, 0.01)
    active = audio[np.abs(audio) >= gate]
    return active if active.size else audio


def _find_default_voice(repo_root: Path) -> Path | None:
    candidates = [
        repo_root / "BayanSynthTTS" / "voices" / "default.wav",
        repo_root / "demos" / "studio" / "voices" / "default.wav",
        repo_root / "asset" / "default.wav",
        repo_root / "BayanSynthTTS" / "asset" / "zero_shot_prompt.wav",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def _load_mono_24k(path: str) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = librosa.to_mono(audio.T)
    if sr != 24000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=24000)
    return np.asarray(audio, dtype=np.float32)


def _default_stats(repo_root: Path) -> dict[str, float] | None:
    default_voice = _find_default_voice(repo_root)
    if default_voice is None:
        return None
    audio = _load_mono_24k(str(default_voice))
    if audio.size == 0:
        return None
    active = _active_audio_region(audio)
    rms = float(np.sqrt(np.mean(np.square(active, dtype=np.float64)))) if active.size else 0.0
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if rms <= 1e-6 or peak <= 1e-6:
        return None
    return {"rms": rms, "peak": peak}


def normalize_legacy_match_default(audio: np.ndarray, stats: dict[str, float] | None) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32)
    if x.size == 0:
        return x

    active = _active_audio_region(x)
    current_rms = float(np.sqrt(np.mean(np.square(active, dtype=np.float64)))) if active.size else 0.0
    if stats and current_rms > 1e-6:
        x = x * (stats["rms"] / current_rms)

    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak <= 1e-6:
        return x

    ceiling = min(0.95, max(stats["peak"] * 1.05, 0.8)) if stats else 0.9
    if peak > ceiling:
        x = x * (ceiling / peak)
    elif not stats:
        x = x * (0.9 / peak)
    return np.asarray(x, dtype=np.float32)


def normalize_safe_peak_only(audio: np.ndarray) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32)
    if x.size == 0:
        return x
    peak = float(np.max(np.abs(x)))
    if peak <= 1e-6:
        return x
    if peak > 0.95:
        x = x * (0.95 / peak)
    return np.asarray(x, dtype=np.float32)


def audio_metrics(audio: np.ndarray, sr: int = 24000) -> dict[str, float]:
    x = np.asarray(audio, dtype=np.float32)
    if x.size == 0:
        return {"spectral_centroid_hz": 0.0, "hf_energy_pct": 0.0}

    centroid = float(np.mean(librosa.feature.spectral_centroid(y=x, sr=sr)))

    spectrum = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), d=1.0 / sr)
    power = np.abs(spectrum) ** 2
    total = float(np.sum(power))
    if total <= 0:
        hf_pct = 0.0
    else:
        hf_pct = float(np.sum(power[freqs >= 4000.0]) / total * 100.0)

    return {
        "spectral_centroid_hz": centroid,
        "hf_energy_pct": hf_pct,
    }


def synthesize_with_ref(text: str, ref_wav_path: str, seed: int, speed: float) -> np.ndarray:
    from bayansynthtts import BayanSynthTTS

    tts = BayanSynthTTS()
    audio = tts.synthesize(
        text,
        ref_audio=ref_wav_path,
        auto_tashkeel=False,
        seed=seed,
        speed=speed,
    )
    return np.asarray(audio, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="A/B evaluate normalization impact on clone quality")
    parser.add_argument("--text", required=True, help="Arabic synthesis text")
    parser.add_argument("--ref-audio", required=True, help="Reference audio path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--out-dir", default="BayanSynthTTS/samples/ab_norm_eval")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    out_dir = (repo_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_audio = _load_mono_24k(args.ref_audio)
    stats = _default_stats(repo_root)

    legacy_ref = normalize_legacy_match_default(ref_audio, stats)
    safe_ref = normalize_safe_peak_only(ref_audio)

    with tempfile.NamedTemporaryFile(suffix="_legacy.wav", delete=False) as f1, tempfile.NamedTemporaryFile(
        suffix="_safe.wav", delete=False
    ) as f2:
        legacy_ref_path = f1.name
        safe_ref_path = f2.name

    try:
        sf.write(legacy_ref_path, legacy_ref, 24000, subtype="PCM_16")
        sf.write(safe_ref_path, safe_ref, 24000, subtype="PCM_16")

        legacy_audio = synthesize_with_ref(args.text, legacy_ref_path, args.seed, args.speed)
        safe_audio = synthesize_with_ref(args.text, safe_ref_path, args.seed, args.speed)

        legacy_out = out_dir / "legacy_match_default.wav"
        safe_out = out_dir / "safe_peak_only.wav"
        sf.write(str(legacy_out), legacy_audio, 24000, subtype="PCM_16")
        sf.write(str(safe_out), safe_audio, 24000, subtype="PCM_16")

        m_legacy = audio_metrics(legacy_audio)
        m_safe = audio_metrics(safe_audio)

        report = {
            "text": args.text,
            "ref_audio": str(Path(args.ref_audio).resolve()),
            "seed": args.seed,
            "speed": args.speed,
            "metrics": {
                "legacy_match_default": m_legacy,
                "safe_peak_only": m_safe,
            },
            "delta": {
                "spectral_centroid_hz": m_safe["spectral_centroid_hz"] - m_legacy["spectral_centroid_hz"],
                "hf_energy_pct": m_safe["hf_energy_pct"] - m_legacy["hf_energy_pct"],
            },
            "outputs": {
                "legacy_match_default": str(legacy_out),
                "safe_peak_only": str(safe_out),
            },
        }

        report_path = out_dir / "report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

        print("A/B normalization evaluation complete")
        print(f"  Legacy output: {legacy_out}")
        print(f"  Safe output:   {safe_out}")
        print(f"  Report:        {report_path}")
        print("  Metrics:")
        print(
            "    legacy: centroid={:.2f} Hz, hf_energy={:.4f}%".format(
                m_legacy["spectral_centroid_hz"], m_legacy["hf_energy_pct"]
            )
        )
        print(
            "    safe:   centroid={:.2f} Hz, hf_energy={:.4f}%".format(
                m_safe["spectral_centroid_hz"], m_safe["hf_energy_pct"]
            )
        )
        print(
            "    delta(safe-legacy): centroid={:+.2f} Hz, hf_energy={:+.4f}%".format(
                report["delta"]["spectral_centroid_hz"], report["delta"]["hf_energy_pct"]
            )
        )
    finally:
        for p in (legacy_ref_path, safe_ref_path):
            if os.path.isfile(p):
                try:
                    os.unlink(p)
                except OSError:
                    pass


if __name__ == "__main__":
    main()
