# BayanSynthTTS

**Arabic Text-to-Speech powered by CosyVoice3 with LoRA fine-tuning.**

> Just text in, speech out.  No training code, no data processing — inference only.

## Features

| Feature | Details |
|---------|---------|
| 🇸🇦 Arabic TTS | Natural-sounding Modern Standard Arabic |
| 🔤 Auto-Tashkeel | Automatic diacritization via mishkal (always on) |
| 🎙 Voice Cloning | Clone any voice from a 5–15s clip (WAV/MP3/OGG/M4A/FLAC) |
| 🔄 LoRA Swapping | Change checkpoints via `conf/models.yaml` — no code edits |
| 🌊 Streaming | Chunk-by-chunk audio generation |
| 🖥 Gradio UI | Simple web interface included |
| 💻 CLI | One-liner inference from terminal |

---

## Quick Start

### 1. Prerequisites

This library lives inside the [CosyVoice-Arabic](https://github.com/Ramendan/CosyVoice-Arabic) repo.
Clone the main repo first, then install dependencies:

```bash
git clone https://github.com/Ramendan/CosyVoice-Arabic
cd CosyVoice-Arabic
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r BayanSynthTTS/requirements.txt
```

### 2. Download the base model + set up checkpoints

```bash
python BayanSynthTTS/scripts/setup_models.py
```

This will:
- Download CosyVoice3 weights from Hugging Face Hub
- Check if LoRA checkpoints exist in `BayanSynthTTS/checkpoints/`
- Copy the default reference voice

### 3. Add LoRA checkpoints

Copy your trained `.pt` files to:
```
BayanSynthTTS/
└── checkpoints/
    ├── llm/
    │   └── epoch_28_whole.pt       ← LLM LoRA (required for best Arabic quality)
    └── flow/
        └── epoch_15_step_49000.pt  ← Flow LoRA (optional, disabled by default)
```

### 4. Run

**Web UI:**
```bash
scripts\run_ui.bat            # Windows
python bayansynthtts/app.py   # Cross-platform
```

**Python API:**
```python
from bayansynthtts import BayanSynthTTS

tts = BayanSynthTTS()
audio = tts.synthesize("مرحبا بكم في اختبار النظام")
tts.save_wav(audio, "output.wav")
```

---

## Python API

### Basic synthesis (auto-tashkeel on by default)

```python
from bayansynthtts import BayanSynthTTS

tts = BayanSynthTTS()

# Plain Arabic — diacritics added automatically
audio = tts.synthesize("مرحبا بكم")

# Pre-diacritized text
audio = tts.synthesize("مَرْحَباً بِكُمْ")

tts.save_wav(audio, "output.wav")
```

### Voice cloning

```python
# Any audio format — auto-converted to 24 kHz mono WAV internally
audio = tts.synthesize("مَرْحَباً", ref_audio="my_voice.mp3")
```

### Streaming output

```python
for chunk in tts.synthesize("النص العربي الطويل...", stream=True):
    # chunk is a numpy array; feed to audio device or accumulate
    pass
```

### Save directly to file

```python
duration = tts.synthesize_to_file("مَرْحَباً بِكُمْ", "output.wav")
print(f"Generated {duration:.1f}s of audio")
```

### Disable tashkeel (text already diacritized)

```python
audio = tts.synthesize("مَرْحَباً بِكُمْ", auto_tashkeel=False)
```

### List voices

```python
print(tts.list_voices())   # e.g. ['default.wav', 'speaker2.wav']
audio = tts.synthesize("مَرْحَباً", ref_audio=tts.get_voice_path("speaker2.wav"))
```

---

## Swapping the LoRA Checkpoint

### Via `conf/models.yaml` (recommended — no code changes)

```yaml
llm_lora:
  enabled: true
  checkpoint: "checkpoints/llm/my_new_epoch.pt"   # ← change this line only
```

### Via Python constructor (for A/B testing at runtime)

```python
tts = BayanSynthTTS(llm_checkpoint="checkpoints/llm/epoch_40.pt")
```

### Via CLI flag

```bash
bayansynthtts "مَرْحَباً" --llm checkpoints/llm/epoch_40.pt
```

### Enable Flow LoRA

```yaml
flow_lora:
  enabled: true
  checkpoint: "checkpoints/flow/my_flow.pt"
  scale: 0.6   # tune 0.3–1.0
```

---

## Changing the Default Voice

**Option A — Replace the file:**
```bash
cp my_voice.wav BayanSynthTTS/voices/default.wav
```

**Option B — Edit `conf/models.yaml`:**
```yaml
defaults:
  voice: "voices/my_voice.wav"
```

**Option C — Pass at runtime:**
```python
audio = tts.synthesize("مَرْحَباً", ref_audio="path/to/my_voice.mp3")
```

**Voice quality guidelines:** 5–15 seconds, quiet room, clear Arabic speech, no music.

---

## CLI Reference

```bash
bayansynthtts "مَرْحَباً بِكُمْ"                       # basic synthesis → output.wav
bayansynthtts "مَرْحَباً" -o hello.wav                  # custom output path
bayansynthtts "مَرْحَباً" --voice voices/speaker2.wav   # use specific voice
bayansynthtts "مَرْحَباً" --llm checkpoints/llm/new.pt  # override LLM LoRA
bayansynthtts "مَرْحَباً" --enable-flow                 # enable flow LoRA
bayansynthtts "مَرْحَباً" --speed 0.85                  # slower speech
bayansynthtts "مَرْحَباً" --no-tashkeel                 # skip auto-diacritize
bayansynthtts "مَرْحَباً" --seed 123                    # reproducible output
bayansynthtts --help
```

---

## Project Layout

```
BayanSynthTTS/
├── bayansynthtts/          # Python package (inference only)
│   ├── __init__.py         # Public API exports
│   ├── inference.py        # Core TTS engine + LoRA injection
│   ├── tashkeel.py         # Arabic diacritization (mishkal + tashkeel)
│   └── app.py              # Gradio web UI
├── checkpoints/            # LoRA checkpoints (not tracked in git)
│   ├── llm/                # LLM LoRA .pt files
│   └── flow/               # Flow LoRA .pt files
├── conf/
│   └── models.yaml         # ← Edit this to swap models / defaults
├── voices/
│   ├── default.wav         # Default reference voice (replace with your own)
│   └── README.md
├── scripts/
│   ├── setup_models.py     # One-time setup (download base model, check deps)
│   ├── setup_models.bat    # Windows wrapper
│   ├── run_ui.bat          # Launch Gradio UI (Windows)
│   └── infer.bat           # CLI inference (Windows)
├── pyproject.toml          # Package definition (pip install -e .)
├── requirements.txt        # Dependency list
└── README.md
```

---

## API Reference

### `BayanSynthTTS`

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `model_dir` | `str` | from YAML | CosyVoice3 weights directory |
| `llm_checkpoint` | `str` | from YAML | LLM LoRA `.pt` path |
| `flow_checkpoint` | `str` | from YAML | Flow LoRA `.pt` path |
| `flow_lora_scale` | `float` | `0.6` | Flow LoRA scale (0.3–1.0) |
| `disable_flow_lora` | `bool` | `True` | Skip flow LoRA injection |
| `ref_audio` | `str` | from YAML | Default reference voice path |
| `instruct` | `str` | from YAML | Instruct prompt text |
| `config_path` | `str` | `conf/models.yaml` | Custom config file path |

### `synthesize(text, *, ...)`

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `text` | `str` | — | Arabic text (plain or diacritized) |
| `ref_audio` | `str` | default voice | Voice clone source (any format) |
| `instruct` | `str` | from config | Instruct prompt override |
| `speed` | `float` | `1.0` | Speed multiplier (0.5–2.0) |
| `stream` | `bool` | `False` | Yield chunks vs return full array |
| `seed` | `int` | `None` | Random seed for reproducibility |
| `auto_tashkeel` | `bool` | `True` | Auto-diacritize input text |

### Tashkeel utilities

```python
from bayansynthtts import auto_diacritize, has_harakat, strip_harakat, list_available_backends

auto_diacritize("مرحبا بكم")          # → "مَرْحَباً بِكُمْ"
has_harakat("مَرْحَباً")              # → True
strip_harakat("مَرْحَباً")            # → "مرحبا"
list_available_backends()              # → ['mishkal']  (or ['tashkeel', 'mishkal'])
```

---

## GitHub Repo Setup

### Step 1 — Push the code (no checkpoints in git)

```bash
cd BayanSynthTTS

git init
git add .
git commit -m "Initial BayanSynthTTS library — inference only"

# Create a new repo on GitHub (e.g. https://github.com/YOUR_USER/BayanSynthTTS)
git remote add origin https://github.com/YOUR_USER/BayanSynthTTS
git branch -M main
git push -u origin main
```

> **Why no checkpoints in git?** `.pt` files are 100MB–1GB.  
> GitHub blocks files >100MB and git history bloats permanently with large binaries.  
> The `.gitignore` already excludes `checkpoints/` and `*.pt`.

### Step 2 — Publish checkpoints as a GitHub Release asset

```bash
# Tag the release
git tag v1.0
git push origin v1.0
```

Then on GitHub: **Releases → Draft a new release → tag v1.0**  
Attach these files as release assets:
- `checkpoints/llm/epoch_28_whole.pt`
- `checkpoints/flow/epoch_15_step_49000.pt` (optional)

Or use the GitHub CLI:
```bash
gh release create v1.0 --title "BayanSynthTTS v1.0"
gh release upload v1.0 checkpoints/llm/epoch_28_whole.pt
```

### Step 3 — Update the release URL in setup_models.py

In [scripts/setup_models.py](scripts/setup_models.py), update:
```python
GITHUB_RELEASE_URL = "https://github.com/YOUR_USER/BayanSynthTTS/releases/download/v1.0"
```

Commit and push. Users now get everything with:
```bash
git clone https://github.com/YOUR_USER/BayanSynthTTS
cd BayanSynthTTS
pip install -r requirements.txt
python scripts/setup_models.py   # downloads base model + LoRA checkpoints
scripts\run_ui.bat
```

---

## Troubleshooting

| Problem | Solution |
|---------|---------|
| `No module named 'cosyvoice'` | Run from the CosyVoice-Arabic repo root; add it to `sys.path` |
| `No LLM checkpoint found` | Copy `.pt` to `BayanSynthTTS/checkpoints/llm/` |
| `mishkal not found` | `pip install mishkal` |
| No audio generated | Check console for the specific mode that failed; verify `voices/default.wav` exists |
| Bad audio quality | Disable flow LoRA (`flow_lora.enabled: false`); try different seed |
| MP3/M4A upload fails | Install ffmpeg: `winget install ffmpeg` |
| `huggingface_hub>=1.0` error | `pip install "huggingface_hub<1.0"` — do NOT upgrade to ≥1.0 |

---

## License

Apache 2.0 — See [LICENSE](../LICENSE)

The underlying CosyVoice3 model is subject to its own license.  
LoRA checkpoints trained on Common Voice Arabic data are released under CC‑BY 4.0.
