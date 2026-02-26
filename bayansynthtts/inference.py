#!/usr/bin/env python3
"""BayanSynthTTS inference engine.

Single source of truth for model loading, LoRA injection, and synthesis.
Both the CLI and Gradio UI import from here.

Usage (programmatic)::

    from bayansynthtts import BayanSynthTTS

    tts = BayanSynthTTS()
    audio = tts.synthesize("مَرْحَباً بِكُمْ فِي اخْتِبَارِ نِظَامِ تَحْوِيلِ النَّصِّ إِلَى كَلَامٍ بِالْعَرَبِيِّ")
    tts.save_wav(audio, "output.wav")
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from typing import Generator, Optional, Union

import numpy as np

# Force UTF-8 on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# ── Paths ──────────────────────────────────────────────────────────────────
# BayanSynthTTS/ directory (one level up from bayansynthtts/inference.py)
BAYAN_DIR = str(Path(__file__).resolve().parent.parent)
# Repo root (one level up from BayanSynthTTS/) — where cosyvoice/ and third_party/ live
REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
_matcha = os.path.join(REPO_ROOT, "third_party", "Matcha-TTS")
if _matcha not in sys.path:
    sys.path.insert(0, _matcha)

DEFAULT_MODEL_DIR = os.path.join(REPO_ROOT, "pretrained_models", "CosyVoice3")

# LoRA checkpoints live in BayanSynthTTS/checkpoints/
DEFAULT_LLM_CKPT = os.path.join(BAYAN_DIR, "checkpoints", "llm", "epoch_28_whole.pt")

# Default reference voice
DEFAULT_PROMPT_WAV = os.path.join(BAYAN_DIR, "voices", "default.wav")
# Fallback to original asset if voices/default.wav not present
_ASSET_PROMPT_WAV = os.path.join(REPO_ROOT, "asset", "zero_shot_prompt.wav")

# Instruct prompt — MUST match training data
DEFAULT_INSTRUCT = "You are a helpful assistant.<|endofprompt|>"

# Default harakat text for quick tests
DEFAULT_TEXT_HARAKAT = (
    "مَرْحَباً بِكُمْ فِي اخْتِبَارِ نِظَامِ تَحْوِيلِ النَّصِّ إِلَى كَلَامٍ بِالْعَرَبِيِّ"
)

SAMPLE_RATE = 24000

# ── LLM LoRA config (must match training) ──────────────────────────────────
LLM_LORA_R = 8
LLM_LORA_ALPHA = 16
LLM_LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


# ═══════════════════════════════════════════════════════════════════════════
#  Model Configuration Loader
# ═══════════════════════════════════════════════════════════════════════════

def _resolve_path(p: str, base: str) -> str:
    """Resolve a path that may be absolute, relative-to-base, or relative-to-repo-root."""
    if os.path.isabs(p):
        return p
    candidate = os.path.normpath(os.path.join(base, p))
    if os.path.exists(candidate):
        return candidate
    candidate2 = os.path.normpath(os.path.join(base, "..", p))
    if os.path.exists(candidate2):
        return candidate2
    return candidate


def load_model_config(config_path: Optional[str] = None) -> dict:
    """Load BayanSynthTTS/conf/models.yaml and resolve all paths.

    Returns a flat dict with keys: model_dir, llm_checkpoint, llm_enabled,
    llm_r, llm_alpha, llm_targets, default_voice, instruct,
    auto_tashkeel, sample_rate.

    Falls back to hardcoded defaults if the config file is missing.
    """
    import yaml

    yaml_path = config_path or os.path.join(BAYAN_DIR, "conf", "models.yaml")

    cfg: dict = {}
    if os.path.isfile(yaml_path):
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    base = BAYAN_DIR

    def _p(section, key, fallback):
        val = cfg.get(section, {}).get(key, fallback)
        return _resolve_path(str(val), base) if val else fallback

    def _v(section, key, fallback):
        return cfg.get(section, {}).get(key, fallback)

    return {
        "model_dir":       _p("base_model", "model_dir", DEFAULT_MODEL_DIR),
        "llm_checkpoint":  _p("llm_lora",   "checkpoint", DEFAULT_LLM_CKPT),
        "llm_enabled":     _v("llm_lora",   "enabled",    True),
        "llm_r":           _v("llm_lora",   "r",          LLM_LORA_R),
        "llm_alpha":       _v("llm_lora",   "alpha",      LLM_LORA_ALPHA),
        "llm_targets":     _v("llm_lora",   "target_modules", LLM_LORA_TARGETS),
        "default_voice":   _p("defaults",   "voice",      DEFAULT_PROMPT_WAV),
        "instruct":        _v("defaults",   "instruct",   DEFAULT_INSTRUCT),
        "auto_tashkeel":   _v("defaults",   "auto_tashkeel", True),
        "sample_rate":     _v("defaults",   "sample_rate", SAMPLE_RATE),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  WAV I/O
# ═══════════════════════════════════════════════════════════════════════════

def save_wav(filepath: str, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    """Save a numpy audio array to a 16-bit PCM WAV file."""
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 2:
        if arr.shape[0] <= arr.shape[1]:
            arr = arr.T
    elif arr.ndim != 1:
        arr = arr.flatten()

    peak = np.abs(arr).max()
    if peak > 1.0:
        arr = arr / peak

    try:
        import soundfile as sf
        sf.write(filepath, arr, sample_rate, subtype="PCM_16")
        return
    except ImportError:
        pass
    try:
        from scipy.io import wavfile as scipy_wav
        int16 = (arr * 32767).astype(np.int16)
        if int16.ndim == 2:
            int16 = int16[:, 0]
        scipy_wav.write(filepath, sample_rate, int16)
        return
    except ImportError:
        pass
    import torch, torchaudio
    t = torch.from_numpy(arr.T if arr.ndim == 2 else arr.reshape(1, -1)).float()
    torchaudio.save(filepath, t, sample_rate)


def convert_audio_to_wav(src_path: str, target_sr: int = SAMPLE_RATE) -> str:
    """Convert any audio file to a temporary 24 kHz mono WAV.

    Supports mp3, ogg, m4a, flac, aac, opus, wav and any format that
    torchaudio (via ffmpeg backend) or soundfile can handle.

    Returns the path to the converted temporary WAV (delete when finished).
    Raises RuntimeError if all conversion backends fail.
    """
    import tempfile, shutil, subprocess
    import torch
    import torchaudio
    import torchaudio.functional as F

    wav, sr = None, None
    load_errors = []

    # 1. torchaudio + ffmpeg backend
    try:
        wav, sr = torchaudio.load(src_path, backend="ffmpeg")
    except Exception as e:
        load_errors.append(f"ffmpeg backend: {e}")

    # 2. soundfile
    if wav is None:
        try:
            import soundfile as sf
            data, sr = sf.read(src_path, dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)
            wav = torch.from_numpy(data).unsqueeze(0)
        except Exception as e:
            load_errors.append(f"soundfile: {e}")

    # 3. torchaudio + soundfile backend
    if wav is None:
        try:
            wav, sr = torchaudio.load(src_path, backend="soundfile")
        except Exception as e:
            load_errors.append(f"soundfile backend: {e}")

    # 4. subprocess ffmpeg
    if wav is None:
        if shutil.which("ffmpeg"):
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.close()
            try:
                subprocess.run(
                    ["ffmpeg", "-y", "-i", src_path, "-ac", "1",
                     "-ar", str(target_sr), "-sample_fmt", "s16", tmp.name],
                    check=True, capture_output=True,
                )
                return tmp.name
            except subprocess.CalledProcessError as e:
                os.unlink(tmp.name)
                load_errors.append(f"ffmpeg subprocess: {e.stderr.decode(errors='replace')[:200]}")
        else:
            load_errors.append("ffmpeg not in PATH")

    if wav is None:
        raise RuntimeError(
            f"[BayanSynthTTS] Cannot load audio '{src_path}'. Tried: {'; '.join(load_errors)}\n"
            "TIP: Install ffmpeg (https://ffmpeg.org/download.html) and ensure it is in PATH."
        )

    if wav.ndim == 2 and wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    elif wav.ndim == 1:
        wav = wav.unsqueeze(0)

    if sr != target_sr:
        wav = F.resample(wav, sr, target_sr)

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    torchaudio.save(tmp.name, wav, target_sr, encoding="PCM_S", bits_per_sample=16)
    return tmp.name


def load_ref_audio(path: str, target_sr: int = SAMPLE_RATE):
    """Load and resample a reference audio file. Returns (tensor_1d, sample_rate).

    Auto-converts mp3, ogg, m4a, flac, aac and any ffmpeg-supported format to
    a temporary 24 kHz WAV then loads it.
    """
    import torch

    ext = Path(path).suffix.lower()
    needs_conversion = ext not in (".wav",)

    tmp_path = None
    load_path = path

    if needs_conversion:
        tmp_path = convert_audio_to_wav(path, target_sr=target_sr)
        load_path = tmp_path

    try:
        try:
            import soundfile as sf
            data, sr = sf.read(load_path, dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)
            wav = torch.from_numpy(data)
        except Exception:
            import torchaudio
            wav, sr = torchaudio.load(load_path)
            wav = wav[0]

        if sr != target_sr:
            import torchaudio.functional as F
            wav = F.resample(wav, sr, target_sr)
    finally:
        if tmp_path and os.path.isfile(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    return wav, target_sr


# ═══════════════════════════════════════════════════════════════════════════
#  LoRA Injection
# ═══════════════════════════════════════════════════════════════════════════

def _inject_llm_lora(cosyvoice, checkpoint_path: str, *, device=None) -> int:
    """Inject LLM LoRA adapters and load checkpoint weights.

    Returns number of successfully loaded tensor keys.
    """
    import torch
    from peft import LoraConfig, inject_adapter_in_model

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sd = torch.load(checkpoint_path, map_location="cpu")
    has_lora = any("lora_" in k for k in sd.keys())

    if has_lora:
        lora_cfg = LoraConfig(
            r=LLM_LORA_R,
            lora_alpha=LLM_LORA_ALPHA,
            lora_dropout=0.0,
            target_modules=LLM_LORA_TARGETS,
            bias="none",
        )
        inject_adapter_in_model(lora_cfg, cosyvoice.model.llm.llm.model)

    tensor_sd = {k: v for k, v in sd.items() if isinstance(v, torch.Tensor)}

    if has_lora:
        model_keys = set(cosyvoice.model.llm.state_dict().keys())
        remapped = {}
        for k, v in tensor_sd.items():
            if k in model_keys:
                remapped[k] = v
            else:
                done = False
                for sfx in (".weight", ".bias"):
                    if k.endswith(sfx):
                        nk = k[: -len(sfx)] + ".base_layer" + sfx
                        if nk in model_keys and nk not in tensor_sd:
                            remapped[nk] = v
                            done = True
                            break
                if not done:
                    remapped[k] = v
        tensor_sd = remapped

    missing, unexpected = cosyvoice.model.llm.load_state_dict(tensor_sd, strict=False)
    n_loaded = len(tensor_sd) - len(unexpected)

    cosyvoice.model.llm.to(device).eval()
    return n_loaded


# ═══════════════════════════════════════════════════════════════════════════
#  BayanSynthTTS — Main inference class
# ═══════════════════════════════════════════════════════════════════════════

class BayanSynthTTS:
    """High-level Arabic TTS engine.

    Wraps CosyVoice3 with a LoRA-finetuned LLM for Arabic synthesis.

    All arguments are optional — defaults come from ``BayanSynthTTS/conf/models.yaml``.
    Pass explicit values to override without editing YAML (useful for A/B testing checkpoints).

    Example::

        tts = BayanSynthTTS()
        audio = tts.synthesize("مَرْحَباً بِكُمْ")
        tts.save_wav(audio, "hello.wav")

    Voice cloning::

        tts = BayanSynthTTS()
        audio = tts.synthesize("مَرْحَباً", ref_audio="my_voice.mp3")

    Swap LoRA at runtime::

        tts = BayanSynthTTS(llm_checkpoint="checkpoints/llm/my_epoch.pt")
    """

    def __init__(
        self,
        model_dir: Optional[str] = None,
        llm_checkpoint: Optional[str] = None,
        ref_audio: Optional[str] = None,
        instruct: Optional[str] = None,
        config_path: Optional[str] = None,
    ):
        import torch
        from cosyvoice.cli.cosyvoice import AutoModel

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._lock = threading.Lock()

        # ── Load config and apply explicit overrides ─────────────────────────
        cfg = load_model_config(config_path)

        _model_dir     = model_dir      or cfg["model_dir"]
        self._llm_ckpt = llm_checkpoint or (cfg["llm_checkpoint"] if cfg["llm_enabled"] else None)

        # Resolve default voice: voices/default.wav → asset fallback
        _default_voice = cfg["default_voice"]
        if ref_audio:
            self._ref_audio_path = ref_audio
        elif _default_voice and os.path.isfile(_default_voice):
            self._ref_audio_path = _default_voice
        elif os.path.isfile(DEFAULT_PROMPT_WAV):
            self._ref_audio_path = DEFAULT_PROMPT_WAV
        elif os.path.isfile(_ASSET_PROMPT_WAV):
            self._ref_audio_path = _ASSET_PROMPT_WAV
        else:
            self._ref_audio_path = None

        _raw_instruct = instruct or cfg["instruct"]
        self._instruct = _raw_instruct if "<|endofprompt|>" in _raw_instruct else _raw_instruct + "<|endofprompt|>"
        self._default_auto_tashkeel: bool = cfg["auto_tashkeel"]

        # ── Load base model ──────────────────────────────────────────────────
        print(f"[BayanSynthTTS] Loading CosyVoice3 from {_model_dir}")
        self.cosyvoice = AutoModel(model_dir=_model_dir)

        # ── Inject LLM LoRA ─────────────────────────────────────────────────
        if self._llm_ckpt and os.path.isfile(self._llm_ckpt):
            n = _inject_llm_lora(self.cosyvoice, self._llm_ckpt, device=self._device)
            print(f"[BayanSynthTTS] LLM LoRA loaded: {n} keys from {os.path.basename(self._llm_ckpt)}")
        else:
            print(f"[BayanSynthTTS] No LLM checkpoint at '{self._llm_ckpt}' — using pretrained base")

        self.sample_rate = SAMPLE_RATE

    # ── Public API ────────────────────────────────────────────────────────

    def synthesize(
        self,
        text: str,
        *,
        ref_audio: Optional[str] = None,
        instruct: Optional[str] = None,
        speed: float = 1.0,
        stream: bool = False,
        seed: Optional[int] = None,
        auto_tashkeel: Optional[bool] = None,
    ) -> Union[np.ndarray, Generator[np.ndarray, None, None]]:
        """Synthesize Arabic speech from text.

        Args:
            text: Arabic text. Auto-diacritization is ON by default (configure in
                conf/models.yaml or pass auto_tashkeel=False to disable).
            ref_audio: Optional path to reference voice (any format: wav/mp3/ogg/flac/m4a).
                       Overrides the default voice set in conf/models.yaml.
            instruct: Optional override for instruct prompt.
            speed: Playback speed multiplier (0.5–2.0).
            stream: If True, yields numpy chunks instead of returning full array.
            seed: Random seed for reproducibility.
            auto_tashkeel: True/False to override; None = use conf/models.yaml default (True).

        Returns:
            numpy array (samples,) at 24 kHz, or a generator if stream=True.
        """
        import torch
        from cosyvoice.utils.common import set_all_random_seed

        _do_tashkeel = auto_tashkeel if auto_tashkeel is not None else self._default_auto_tashkeel
        if _do_tashkeel:
            from bayansynthtts.tashkeel import auto_diacritize
            text = auto_diacritize(text)

        if seed is not None:
            set_all_random_seed(seed)

        prompt = ref_audio or self._ref_audio_path
        inst = instruct or self._instruct
        if "<|endofprompt|>" not in inst:
            inst = inst + "<|endofprompt|>"

        with self._lock:
            gen = self._generate(text, prompt, inst, speed, stream)
            if stream:
                return gen
            chunks = list(gen)

        if not chunks:
            return np.zeros(1, dtype=np.float32)

        audio = np.concatenate(chunks, axis=-1)
        peak = np.abs(audio).max()
        if peak > 0:
            audio = audio / peak * 0.9
        return audio

    def synthesize_to_file(
        self, text: str, output_path: str, **kwargs
    ) -> float:
        """Synthesize and save to WAV. Returns duration in seconds."""
        audio = self.synthesize(text, **kwargs)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        save_wav(output_path, audio, self.sample_rate)
        return len(audio) / self.sample_rate

    @staticmethod
    def save_wav(audio: np.ndarray, path: str, sample_rate: int = SAMPLE_RATE) -> None:
        """Save audio to WAV file (convenience wrapper)."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        save_wav(path, audio, sample_rate)

    def list_voices(self) -> list[str]:
        """List available voice files in BayanSynthTTS/voices/."""
        voices_dir = os.path.join(BAYAN_DIR, "voices")
        if not os.path.isdir(voices_dir):
            return []
        return sorted(
            f for f in os.listdir(voices_dir)
            if f.lower().endswith((".wav", ".mp3", ".flac", ".ogg"))
        )

    def get_voice_path(self, name: str) -> str:
        """Get full path for a named voice file."""
        return os.path.join(BAYAN_DIR, "voices", name)

    # ── Private implementation ────────────────────────────────────────────

    def _generate(
        self, text: str, prompt, instruct: str, speed: float, stream: bool
    ) -> Generator[np.ndarray, None, None]:
        """Core synthesis generator. Tries instruct2 → zero_shot → cross_lingual."""
        import tempfile
        import torch

        tmp_wav_path = None

        def _ensure_wav_path(p):
            nonlocal tmp_wav_path
            if p is None:
                return None
            if isinstance(p, torch.Tensor):
                import torchaudio
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                tmp.close()
                wav = p.unsqueeze(0) if p.ndim == 1 else p
                torchaudio.save(tmp.name, wav.float().cpu(), SAMPLE_RATE,
                                encoding="PCM_S", bits_per_sample=16)
                tmp_wav_path = tmp.name
                return tmp.name
            ext = os.path.splitext(p)[1].lower()
            if ext != ".wav":
                converted = convert_audio_to_wav(p, target_sr=SAMPLE_RATE)
                tmp_wav_path = converted
                return converted
            return p

        prompt_path = None
        try:
            prompt_path = _ensure_wav_path(prompt)
        except Exception as conv_err:
            print(f"[BayanSynthTTS] WARNING: Could not prepare ref audio: {conv_err}")

        try:
            # ── Mode 1: instruct2 ─────────────────────────────────────────
            if prompt_path and os.path.isfile(prompt_path):
                try:
                    for chunk in self.cosyvoice.inference_instruct2(
                        text, instruct, prompt_path,
                        stream=stream, speed=speed, text_frontend=False,
                    ):
                        yield chunk["tts_speech"].cpu().numpy().flatten()
                    return
                except Exception as e:
                    print(f"[BayanSynthTTS] instruct2 failed ({type(e).__name__}: {e}), trying zero_shot …")
            else:
                print("[BayanSynthTTS] No ref audio — skipping instruct2, trying zero_shot/cross_lingual")

            # ── Mode 2: zero_shot ─────────────────────────────────────────
            if prompt_path and os.path.isfile(prompt_path):
                try:
                    for chunk in self.cosyvoice.inference_zero_shot(
                        text, "", prompt_path, stream=stream, speed=speed, text_frontend=False
                    ):
                        yield chunk["tts_speech"].cpu().numpy().flatten()
                    return
                except Exception as e:
                    print(f"[BayanSynthTTS] zero_shot failed ({type(e).__name__}: {e}), trying cross_lingual …")

            # ── Mode 3: cross_lingual ─────────────────────────────────────
            if prompt_path and os.path.isfile(prompt_path):
                try:
                    for chunk in self.cosyvoice.inference_cross_lingual(
                        text, prompt_path, stream=stream, text_frontend=False
                    ):
                        yield chunk["tts_speech"].cpu().numpy().flatten()
                    return
                except Exception as e:
                    print(f"[BayanSynthTTS] cross_lingual failed ({type(e).__name__}: {e})")

            # ── Final fallback: asset prompt wav ──────────────────────────
            fallback = _ASSET_PROMPT_WAV if os.path.isfile(_ASSET_PROMPT_WAV) else None
            if fallback:
                print(f"[BayanSynthTTS] Using asset fallback wav: {fallback}")
                try:
                    for chunk in self.cosyvoice.inference_cross_lingual(
                        text, fallback, stream=stream, text_frontend=False
                    ):
                        yield chunk["tts_speech"].cpu().numpy().flatten()
                    return
                except Exception as e:
                    print(f"[BayanSynthTTS] Asset fallback also failed: {e}")

            print("[BayanSynthTTS] ERROR: All inference modes failed. No audio generated.")
        finally:
            if tmp_wav_path and os.path.isfile(tmp_wav_path):
                try:
                    os.unlink(tmp_wav_path)
                except Exception:
                    pass


# ═══════════════════════════════════════════════════════════════════════════
#  CLI entry point
# ═══════════════════════════════════════════════════════════════════════════

def _cli_main():
    """Command-line interface for BayanSynthTTS."""
    import argparse

    parser = argparse.ArgumentParser(
        description="BayanSynthTTS — Arabic TTS powered by CosyVoice3 + LoRA"
    )
    parser.add_argument("text", nargs="?", default=DEFAULT_TEXT_HARAKAT,
                        help="Arabic text to synthesize")
    parser.add_argument("-o", "--output", default="output.wav",
                        help="Output WAV file path")
    parser.add_argument("--voice", default=None,
                        help="Path to reference voice file (any format)")
    parser.add_argument("--llm", default=None,
                        help="Override LLM LoRA checkpoint path")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Speed multiplier (0.5–2.0)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--no-tashkeel", action="store_true",
                        help="Disable auto-diacritization")
    parser.add_argument("--config", default=None,
                        help="Path to a custom models.yaml config file")
    args = parser.parse_args()

    tts = BayanSynthTTS(
        llm_checkpoint=args.llm,
        config_path=args.config,
    )
    dur = tts.synthesize_to_file(
        args.text,
        args.output,
        ref_audio=args.voice,
        speed=args.speed,
        seed=args.seed,
        auto_tashkeel=(not args.no_tashkeel),
    )
    print(f"Saved {dur:.1f}s audio to {args.output}")


if __name__ == "__main__":
    _cli_main()
