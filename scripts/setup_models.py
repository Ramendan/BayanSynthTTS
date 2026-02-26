#!/usr/bin/env python3
"""BayanSynthTTS setup script.

Downloads the CosyVoice3 base model and LoRA checkpoints.

LoRA checkpoints are too large for git (100MB–1GB) and are distributed
via GitHub Releases.  This script downloads them automatically.

Usage:
    # Full setup (base model + checkpoints from GitHub Releases)
    python BayanSynthTTS/scripts/setup_models.py

    # Custom GitHub release URL
    python BayanSynthTTS/scripts/setup_models.py --release-url https://github.com/YOU/BayanSynthTTS/releases/download/v1.0

    # Skip base model download (already have it)
    python BayanSynthTTS/scripts/setup_models.py --skip-base

    # Skip checkpoint download (only check what's present)
    python BayanSynthTTS/scripts/setup_models.py --skip-checkpoints
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import urllib.request
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent          # BayanSynthTTS/scripts/
BAYAN_DIR   = SCRIPT_DIR.parent                         # BayanSynthTTS/
REPO_ROOT   = BAYAN_DIR.parent                          # CosyVoice/ (repo root)

DEFAULT_MODEL_DIR = REPO_ROOT / "pretrained_models" / "CosyVoice3"
DEFAULT_LLM_CKPT  = BAYAN_DIR / "checkpoints" / "llm" / "epoch_28_whole.pt"
DEFAULT_FLOW_CKPT = BAYAN_DIR / "checkpoints" / "flow" / "epoch_15_step_49000.pt"
DEFAULT_VOICE     = BAYAN_DIR / "voices" / "default.wav"
ASSET_PROMPT_WAV  = REPO_ROOT / "asset" / "zero_shot_prompt.wav"

HF_REPO_ID = "FunAudioLLM/CosyVoice3-300M-Instruct"

# GitHub Releases base URL for LoRA checkpoints.
# Set to your own repo's release URL once you publish.
# Format: https://github.com/OWNER/REPO/releases/download/TAG
GITHUB_RELEASE_URL = "https://github.com/Ramendan/BayanSynthTTS/releases/download/v1.0"

# Files to download from the release (filename → destination relative to BAYAN_DIR)
CHECKPOINT_FILES = {
    "epoch_28_whole.pt":        "checkpoints/llm/epoch_28_whole.pt",
    "epoch_15_step_49000.pt":   "checkpoints/flow/epoch_15_step_49000.pt",
}


# ── Helpers ────────────────────────────────────────────────────────────────

def _download_file(url: str, dest: Path) -> bool:
    """Download a file with a progress indicator. Returns True on success."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    try:
        print(f"[setup] Downloading {url}")
        print(f"         → {dest}")

        def _progress(block_count, block_size, total):
            if total > 0:
                pct = min(100, block_count * block_size * 100 // total)
                mb_done = block_count * block_size / 1_048_576
                mb_total = total / 1_048_576
                print(f"\r         {pct:3d}%  {mb_done:.0f}/{mb_total:.0f} MB", end="", flush=True)

        urllib.request.urlretrieve(url, tmp, reporthook=_progress)
        print()  # newline after progress
        tmp.rename(dest)
        size_mb = dest.stat().st_size / 1_048_576
        print(f"[setup] ✓  Downloaded {dest.name}  ({size_mb:.0f} MB)")
        return True
    except Exception as e:
        print(f"\n[setup] ✗  Download failed: {e}")
        if tmp.exists():
            tmp.unlink()
        return False


# ── Main tasks ─────────────────────────────────────────────────────────────

def download_base_model(model_dir: Path, force: bool = False) -> None:
    """Download CosyVoice3 weights from Hugging Face Hub."""
    if model_dir.exists() and not force:
        print(f"[setup] ✓  Base model already exists at: {model_dir}")
        return

    print(f"[setup] Downloading {HF_REPO_ID} → {model_dir}")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=HF_REPO_ID,
            local_dir=str(model_dir),
            ignore_patterns=["*.msgpack", "flax_model*", "tf_model*"],
        )
        print(f"[setup] ✓  Base model downloaded to {model_dir}")
    except Exception as e:
        print(f"[setup] ERROR downloading base model: {e}")
        print("       Install huggingface_hub: pip install huggingface_hub")
        sys.exit(1)


def download_checkpoints(release_url: str, force: bool = False) -> None:
    """Download LoRA checkpoint .pt files from a GitHub Release."""
    release_url = release_url.rstrip("/")
    any_downloaded = False

    for filename, rel_dest in CHECKPOINT_FILES.items():
        dest = BAYAN_DIR / rel_dest
        if dest.exists() and not force:
            size_mb = dest.stat().st_size / 1_048_576
            print(f"[setup] ✓  {filename} already present  ({size_mb:.0f} MB)")
            continue

        url = f"{release_url}/{filename}"
        ok = _download_file(url, dest)
        if ok:
            any_downloaded = True
        else:
            print(f"         Manual download: {url}")
            print(f"         Save to:         {dest}")

    if not any_downloaded:
        print("[setup] All checkpoints already present.")


def check_checkpoints() -> bool:
    """Check which checkpoints are present. Returns True if LLM LoRA found."""
    checks = [
        (DEFAULT_LLM_CKPT,  "LLM LoRA (required)",  True),
        (DEFAULT_FLOW_CKPT, "Flow LoRA (optional)",  False),
    ]
    llm_ok = False
    for path, name, required in checks:
        if path.exists():
            size_mb = path.stat().st_size / 1_048_576
            print(f"[setup] ✓  {name}: {path.name}  ({size_mb:.0f} MB)")
            if required:
                llm_ok = True
        else:
            marker = "✗" if required else "–"
            print(f"[setup] {marker}  {name}: not found at {path}")
    return llm_ok


def ensure_default_voice() -> None:
    """Copy asset/zero_shot_prompt.wav → voices/default.wav if missing."""
    if DEFAULT_VOICE.exists():
        print(f"[setup] ✓  Default voice: {DEFAULT_VOICE.name}")
        return
    if ASSET_PROMPT_WAV.exists():
        DEFAULT_VOICE.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ASSET_PROMPT_WAV, DEFAULT_VOICE)
        print(f"[setup] ✓  Copied default voice → voices/default.wav")
    else:
        print(f"[setup] ⚠  No default voice. Add a WAV file to: {DEFAULT_VOICE}")


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
        print(f"[setup] Updated conf/models.yaml model_dir → {rel}")


# ── Entry point ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="BayanSynthTTS — one-time setup (base model + LoRA checkpoints)"
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR,
                        help="CosyVoice3 base model directory")
    parser.add_argument("--release-url", default=GITHUB_RELEASE_URL,
                        help="GitHub Releases base URL for checkpoint downloads")
    parser.add_argument("--skip-base", action="store_true",
                        help="Skip base model download")
    parser.add_argument("--skip-checkpoints", action="store_true",
                        help="Skip checkpoint download (only check what's present)")
    parser.add_argument("--force", action="store_true",
                        help="Re-download everything even if already present")
    args = parser.parse_args()

    print("=" * 62)
    print("  BayanSynthTTS — Setup")
    print("=" * 62)

    # ── Step 1: Base model ───────────────────────────────────────────
    if not args.skip_base:
        print("\n[1/3] CosyVoice3 base model")
        download_base_model(args.model_dir, force=args.force)
        update_models_yaml(args.model_dir)
    else:
        print("\n[1/3] Skipping base model download")

    # ── Step 2: LoRA checkpoints ─────────────────────────────────────
    print("\n[2/3] LoRA checkpoints")
    if args.skip_checkpoints:
        check_checkpoints()
    else:
        if args.release_url == GITHUB_RELEASE_URL and "Ramendan" in GITHUB_RELEASE_URL:
            # Default URL — try download but don't hard-fail if release not published yet
            print(f"      Release URL: {args.release_url}")
            download_checkpoints(args.release_url, force=args.force)
        else:
            download_checkpoints(args.release_url, force=args.force)

    # ── Step 3: Default voice ────────────────────────────────────────
    print("\n[3/3] Default voice")
    ensure_default_voice()

    # ── Summary ──────────────────────────────────────────────────────
    print()
    llm_ok = DEFAULT_LLM_CKPT.exists()
    print("=" * 62)
    if llm_ok:
        print("  ✅  Setup complete!")
        print()
        print("  Quick test:")
        print('  python -c "from bayansynthtts import BayanSynthTTS; BayanSynthTTS()"')
        print()
        print("  Launch UI:")
        print("  scripts\\run_ui.bat   (or: python bayansynthtts/app.py)")
    else:
        print("  ⚠  Setup incomplete — LLM LoRA checkpoint missing.")
        print()
        print("  The model will still run using the CosyVoice3 base (lower quality).")
        print()
        print("  To add checkpoints manually:")
        print(f"    Copy epoch_28_whole.pt → {DEFAULT_LLM_CKPT}")
        print()
        print("  Or download from GitHub Releases:")
        print(f"    python scripts/setup_models.py --release-url YOUR_RELEASE_URL")
    print("=" * 62)


if __name__ == "__main__":
    main()
