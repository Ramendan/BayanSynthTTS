#!/usr/bin/env python3
"""BayanSynthTTS — Simple Gradio UI for Arabic TTS.

Lightweight, inference-only interface. Text in, audio out.

Usage:
    python -m bayansynthtts.app
    python BayanSynthTTS/bayansynthtts/app.py --port 7865 --share
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import gradio as gr

# Ensure bundled cosyvoice + matcha packages are importable when running from
# the repo root without `pip install -e .`
BAYAN_DIR = str(Path(__file__).resolve().parent.parent)  # BayanSynthTTS/
_pkg_root = BAYAN_DIR
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

# ── Lazy model loading ────────────────────────────────────────────────────
_TTS_INSTANCE = None


def _get_tts():
    global _TTS_INSTANCE
    if _TTS_INSTANCE is None:
        from bayansynthtts.inference import BayanSynthTTS
        _TTS_INSTANCE = BayanSynthTTS()
    return _TTS_INSTANCE


def _list_voices() -> list[tuple[str, str]]:
    """Scan voices/ directory for available voice files."""
    voices_dir = os.path.join(BAYAN_DIR, "voices")
    choices = []
    if os.path.isdir(voices_dir):
        for f in sorted(os.listdir(voices_dir)):
            if f.lower().endswith((".wav", ".mp3", ".flac", ".ogg")):
                choices.append((Path(f).stem, os.path.join(voices_dir, f)))
    # Fallback to asset prompt if voices folder is empty
    if not choices:
        asset = os.path.join(BAYAN_DIR, "asset", "zero_shot_prompt.wav")
        if os.path.isfile(asset):
            choices.append(("Default Voice", asset))
    return choices


def synthesize(
    text: str,
    voice_choice: str,
    voice_upload,
    instruct: str,
    speed: float,
    auto_tashkeel: bool,
    seed: int,
):
    if not text.strip():
        return None, "Please enter some Arabic text."

    # Upload takes priority over dropdown
    ref_audio = None
    if voice_upload:
        ref_audio = voice_upload
    elif voice_choice and os.path.isfile(voice_choice):
        ref_audio = voice_choice

    try:
        tts = _get_tts()
    except Exception as e:
        return None, f"Model load error: {e}"

    instruct_text = instruct.strip() if instruct and instruct.strip() else None

    try:
        audio = tts.synthesize(
            text,
            ref_audio=ref_audio,
            instruct=instruct_text,
            speed=speed,
            seed=int(seed),
            auto_tashkeel=auto_tashkeel,
        )
    except Exception as e:
        return None, f"Synthesis error: {e}"

    if audio is None or len(audio) == 0:
        return None, "No audio generated. Check the console for details."

    return (tts.sample_rate, audio.astype("float32")), "Done"


def build_ui() -> gr.Blocks:
    voice_choices = _list_voices()
    default_voice = voice_choices[0][1] if voice_choices else None

    with gr.Blocks(title="BayanSynthTTS — Arabic TTS", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "#BayanSynthTTS — Arabic Text-to-Speech\n"
            "Powered by **CosyVoice3** with Arabic LoRA fine-tuning. "
            "Type Arabic text (with or without harakat) and press **Generate**."
        )

        with gr.Row():
            with gr.Column(scale=3):
                text_input = gr.Textbox(
                    label="Arabic Text",
                    placeholder="اكتب النص العربي هنا…",
                    lines=4,
                    rtl=True,
                )
                auto_tashkeel = gr.Checkbox(
                    label="Auto-Diacritize (Tashkeel) — recommended for plain Arabic text",
                    value=True,
                )
                generate_btn = gr.Button("Generate Speech", variant="primary", size="lg")
                status = gr.Textbox(label="Status", interactive=False, max_lines=2)
                audio_output = gr.Audio(label="Output Audio", type="numpy")

            with gr.Column(scale=2):
                gr.Markdown("### Voice Settings")

                voice_dropdown = gr.Dropdown(
                    choices=[(label, val) for label, val in voice_choices],
                    value=default_voice,
                    label="Reference Voice",
                    info="Voice from the voices/ folder",
                )
                voice_upload = gr.Audio(
                    label="Upload Your Own Voice (5–15s, any format)",
                    type="filepath",
                    sources=["upload"],
                )

                with gr.Accordion("Advanced", open=False):
                    speed = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="Speed")
                    seed = gr.Number(value=42, label="Seed (for reproducibility)")
                    instruct = gr.Textbox(
                        label="Instruct Prompt (leave blank for default)",
                        placeholder="You are a helpful assistant.",
                        lines=2,
                    )

        generate_btn.click(
            fn=synthesize,
            inputs=[text_input, voice_dropdown, voice_upload,
                    instruct, speed, auto_tashkeel, seed],
            outputs=[audio_output, status],
        )

        gr.Markdown(
            "---\n"
            "**Tips:**\n"
            "- Enable *Auto-Diacritize* when inputting plain Arabic without harakat.\n"
            "- Upload any 5–15 second clean voice clip for tone cloning.\n"
            "- Change `voices/default.wav` to set a permanent default voice.\n"
            "- Swap LoRA checkpoints in `conf/models.yaml` — no code changes needed."
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="BayanSynthTTS Gradio UI")
    parser.add_argument("--port", type=int, default=7865)
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    parser.add_argument("--no-preload", action="store_true",
                        help="Lazy-load model on first request instead of at startup")
    args = parser.parse_args()

    if not args.no_preload:
        print("[BayanSynthTTS] Pre-loading model…")
        _get_tts()

    demo = build_ui()
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
