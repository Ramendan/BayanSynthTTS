#!/usr/bin/env python3
"""BayanSynthTTS setup script.

Downloads the CosyVoice3 base model and LoRA checkpoints.

Usage:
    # Full setup (base model + checkpoints)
    python scripts/setup_models.py

    # Skip base model download (already have it)
    python scripts/setup_models.py --skip-base

    # Skip checkpoint download (only check what is present)
    python scripts/setup_models.py --skip-checkpoints
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import sys
from pathlib import Path

# Paths
SCRIPT_DIR  = Path(__file__).resolve().parent          # BayanSynthTTS/scripts/
BAYAN_DIR   = SCRIPT_DIR.parent                        # BayanSynthTTS/

DEFAULT_MODEL_DIR = BAYAN_DIR / "pretrained_models" / "CosyVoice3"
DEFAULT_LLM_CKPT  = BAYAN_DIR / "checkpoints" / "llm" / "epoch_28_whole.pt"
DEFAULT_VOICE     = BAYAN_DIR / "voices" / "default.wav"
ASSET_PROMPT_WAV  = BAYAN_DIR / "asset" / "zero_shot_prompt.wav"

# Base CosyVoice3 model weights (Hugging Face)
HF_BASE_REPO_ID = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"

# LoRA checkpoints (Hugging Face)
HF_CKPT_REPO_ID = "Ramendan/BayanSynthTTS-checkpoints"

# Files to download (filename -> destination relative to BAYAN_DIR)
CHECKPOINT_FILES = {
    "epoch_28_whole.pt": "checkpoints/llm/epoch_28_whole.pt",
}
# SHA-256 checksums for verification after download
CHECKPOINT_SHA256 = {
    "epoch_28_whole.pt": "805441555f4d829517e6bb79ba74ac23b65c40c8382802362b433d7e91ff8ca2",
}


def _verify_sha256(path: Path, expected: str) -> bool:
    """Return True if file matches expected SHA-256 hex digest."""
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if sha256 != expected:
        print(f"[setup] SHA-256 mismatch for {path.name}!")
        print(f"         Expected: {expected}")
        print(f"         Got:      {sha256}")
        return False
    print(f"[setup] SHA-256 verified: {path.name}")
    return True


def download_base_model(model_dir: Path, force: bool = False) -> None:
    """Download CosyVoice3 weights from Hugging Face Hub."""
    if model_dir.exists() and not force:
        print(f"[setup] Base model already exists at: {model_dir}")
        return

    print(f"[setup] Downloading {HF_BASE_REPO_ID} -> {model_dir}")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=HF_BASE_REPO_ID,
            local_dir=str(model_dir),
            ignore_patterns=["*.msgpack", "flax_model*", "tf_model*"],
        )
        print(f"[setup] Base model downloaded to {model_dir}")
    except Exception as e:
        print(f"[setup] ERROR downloading base model: {e}")
        print("       Install huggingface_hub: pip install huggingface_hub")
        sys.exit(1)


def download_checkpoints(force: bool = False) -> None:
    """Download LoRA checkpoint files from Hugging Face Hub."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[setup] ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    for filename, rel_dest in CHECKPOINT_FILES.items():
        dest = BAYAN_DIR / rel_dest
        if dest.exists() and not force:
            size_mb = dest.stat().st_size / 1_048_576
            print(f"[setup] {filename} already present  ({size_mb:.0f} MB)")
            continue

        print(f"[setup] Downloading {filename} from {HF_CKPT_REPO_ID} ...")
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            hf_hub_download(
                repo_id=HF_CKPT_REPO_ID,
                filename=filename,
                local_dir=str(dest.parent),
            )
            size_mb = dest.stat().st_size / 1_048_576
            print(f"[setup] Downloaded {filename}  ({size_mb:.0f} MB)")
            expected_sha = CHECKPOINT_SHA256.get(filename)
            if expected_sha:
                if not _verify_sha256(dest, expected_sha):
                    dest.unlink()
        except Exception as e:
            print(f"[setup] Download failed: {e}")
            print(f"         Manual download:")
            print(f"         https://huggingface.co/{HF_CKPT_REPO_ID}/resolve/main/{filename}")
            print(f"         Save to: {dest}")


def check_checkpoints() -> bool:
    """Check whether the LLM LoRA checkpoint is present. Returns True if found."""
    if DEFAULT_LLM_CKPT.exists():
        size_mb = DEFAULT_LLM_CKPT.stat().st_size / 1_048_576
        print(f"[setup] LLM LoRA: {DEFAULT_LLM_CKPT.name}  ({size_mb:.0f} MB)")
        return True
    else:
        print(f"[setup] LLM LoRA: not found at {DEFAULT_LLM_CKPT}")
        return False


def ensure_default_voice() -> None:
    """Copy asset/zero_shot_prompt.wav -> voices/default.wav if missing."""
    if DEFAULT_VOICE.exists():
        print(f"[setup] Default voice: {DEFAULT_VOICE.name}")
        return
    if ASSET_PROMPT_WAV.exists():
        DEFAULT_VOICE.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ASSET_PROMPT_WAV, DEFAULT_VOICE)
        print("[setup] Copied default voice -> voices/default.wav")
    else:
        print(f"[setup] No default voice. Add a WAV file to: {DEFAULT_VOICE}")


def update_models_yaml(model_dir: Path) -> None:
    """Update conf/models.yaml model_dir to the resolved relative path."""
    yaml_path = BAYAN_DIR / "conf" / "models.yaml"
    if not yaml_path.exists():
        return
    text = yaml_path.read_text(encoding="utf-8")
    rel = os.path.relpath(model_dir, BAYAN_DIR).replace("\\", "/")
    updated = re.sub(r'(model_dir:\s*)(".+?")', f'model_dir: "{rel}"', text)
    if updated != text:
        yaml_path.write_text(updated, encoding="utf-8")
        print(f"[setup] Updated conf/models.yaml model_dir -> {rel}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BayanSynthTTS -- one-time setup (base model + LoRA checkpoints)"
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR,
                        help="CosyVoice3 base model directory")
    parser.add_argument("--skip-base", action="store_true",
                        help="Skip base model download")
    parser.add_argument("--skip-checkpoints", action="store_true",
                        help="Skip checkpoint download (only check what is present)")
    parser.add_argument("--force", action="store_true",
                        help="Re-download everything even if already present")
    args = parser.parse_args()

    print("=" * 62)
    print("  BayanSynthTTS -- Setup")
    print("=" * 62)

    # Step 1: Base model
    if not args.skip_base:
        print("\n[1/3] CosyVoice3 base model")
        download_base_model(args.model_dir, force=args.force)
        update_models_yaml(args.model_dir)
    else:
        print("\n[1/3] Skipping base model download")

    # Step 2: LoRA checkpoints
    print("\n[2/3] LoRA checkpoints")
    if args.skip_checkpoints:
        check_checkpoints()
    else:
        download_checkpoints(force=args.force)

    # Step 3: Default voice
    print("\n[3/3] Default voice")
    ensure_default_voice()

    # Summary
    print()
    llm_ok = DEFAULT_LLM_CKPT.exists()
    print("=" * 62)
    if llm_ok:
        print("  Setup complete!")
        print()
        print("  Quick test:")
        print('  python -c "from bayansynthtts import BayanSynthTTS; BayanSynthTTS()"')
        print()
        print("  Launch UI:")
        print("  scripts\\run_ui.bat   (or: python bayansynthtts/app.py)")
    else:
        print("  Setup incomplete -- LLM LoRA checkpoint missing.")
        print()
        print("  The model will still run using the CosyVoice3 base (lower quality).")
        print()
        print("  To add the checkpoint manually:")
        url = f"https://huggingface.co/{HF_CKPT_REPO_ID}/resolve/main/epoch_28_whole.pt"
        print(f"    Download: {url}")
        print(f"    Save to:  {DEFAULT_LLM_CKPT}")
    print("=" * 62)


if __name__ == "__main__":
    main()