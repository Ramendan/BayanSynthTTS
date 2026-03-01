# BayanSynthTTS

**Arabic Text-to-Speech powered by CosyVoice3 with LoRA fine-tuning.**

> Just text in, speech out.  No training code, no data processing — inference only.

## Features

| Feature | Details |
|---------|---------|
| Arabic TTS | Natural-sounding Modern Standard Arabic |
| Auto-Tashkeel | Automatic diacritization via mishkal (always on) |
| Voice Cloning | Clone any voice from a 5–15s clip (WAV/MP3/OGG/M4A/FLAC) |
| LoRA Swapping | Change checkpoints via `conf/models.yaml` — no code edits |
| Streaming | Chunk-by-chunk audio generation |
| Gradio UI | Simple web interface included |
| CLI | One-liner inference from terminal |
| Multilingual base model | CosyVoice3 supports many languages; this package ships with an Arabic LoRA by default |


---

> **Multilingual note:** the underlying CosyVoice3 base model is trained for zero-shot
> synthesis across a wide range of languages. BayanSynthTTS currently defaults to an
> Arabic-conditioned LoRA checkpoint and delivers the best results in Modern Standard
> Arabic. You are free to plug in other LoRA files (not provided here) for additional
> languages, though quality may vary.


## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/Ramendan/BayanSynthTTS
cd BayanSynthTTS
python -m venv .venv
.venv\Scripts\activate         # Windows
# source .venv/bin/activate   # Linux / macOS
pip install -r requirements.txt
pip install -e .               # installs bayansynthtts + bundled packages into the venv
```

> The CosyVoice3 inference engine and Matcha-TTS decoder are **bundled directly in this repo** — no external private repos required.
>
> **First-run note:** when Python loads the model for the first time it downloads a small Chinese/English text normalizer (~30 MB) to your local cache (`~/.cache/modelscope`). This is a one-time download and is only needed internally by the audio engine — your Arabic text goes through a separate pipeline.

### 2. Download models

```bash
python scripts/setup_models.py
```

This downloads everything automatically:
- CosyVoice3 base weights (~2 GB) from Hugging Face → `pretrained_models/CosyVoice3/`
- Arabic LoRA checkpoint from GitHub Releases → `checkpoints/llm/epoch_28_whole.pt`
- Verifies the checkpoint SHA-256

> On Windows you can also double-click `scripts\setup_models.bat`.

### 3. Run

**Web UI:**
```bash
scripts\run_ui.bat            # Windows GUI launcher
python bayansynthtts/app.py   # Cross-platform (run from inside BayanSynthTTS/)
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
├── cosyvoice/              # Bundled CosyVoice3 engine (Apache 2.0)
├── matcha/                 # Bundled Matcha-TTS / HiFi-GAN decoder (Apache 2.0)
├── checkpoints/            # LoRA checkpoints (not tracked in git)
│   └── llm/                # LLM LoRA .pt files
├── pretrained_models/      # CosyVoice3 base weights (~2 GB, not tracked in git)
│   └── CosyVoice3/
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

## Hosting Checkpoints on Hugging Face (Faster Downloads)

Checkpoints hosted on **Hugging Face Hub** benefit from global CDN, resume-on-failure, and are often faster to download than GitHub Releases (especially outside the US).

### Upload your LoRA checkpoint to HF

```bash
pip install huggingface_hub
huggingface-cli login              # enter your HF token (write access)

# Create a new HF model repo (one time)
huggingface-cli repo create BayanSynthTTS-checkpoints --type model

# Upload the checkpoint
huggingface-cli upload Ramendan/BayanSynthTTS-checkpoints \
    checkpoints/llm/epoch_28_whole.pt epoch_28_whole.pt
```

### Point setup_models.py at HF instead of GitHub Releases

In [scripts/setup_models.py](scripts/setup_models.py), replace the `download_checkpoints` call with a HF download:

```python
from huggingface_hub import hf_hub_download

def download_checkpoints_hf(repo_id: str, force: bool = False) -> None:
    for filename, rel_dest in CHECKPOINT_FILES.items():
        dest = BAYAN_DIR / rel_dest
        if dest.exists() and not force:
            print(f"[setup] {filename} already present")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(dest.parent))
        print(f"[setup] Downloaded {filename} from HF")
```

Then call `download_checkpoints_hf("Ramendan/BayanSynthTTS-checkpoints")` instead.

> This is optional — GitHub Releases works fine for most users. HF is recommended if you find downloads are slow or unreliable.

---

## GitHub Repo Setup

### Step 1 — Push the code (no checkpoints in git)

```bash
cd BayanSynthTTS

git init
git add .
git commit -m "Initial BayanSynthTTS library — inference only"

# Create a new repo on GitHub (e.g. https://github.com/Ramendan/BayanSynthTTS)
git remote add origin https://github.com/Ramendan/BayanSynthTTS
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

Or use the GitHub CLI:
```bash
gh release create v1.0 --title "BayanSynthTTS v1.0"
gh release upload v1.0 checkpoints/llm/epoch_28_whole.pt
```

### Step 3 — Update the release URL in setup_models.py

In [scripts/setup_models.py](scripts/setup_models.py), update:
```python
GITHUB_RELEASE_URL = "https://github.com/Ramendan/BayanSynthTTS/releases/download/v1.0"
```

Commit and push. Users now get everything with:
```bash
git clone https://github.com/Ramendan/BayanSynthTTS
cd BayanSynthTTS
pip install -r requirements.txt
python scripts/setup_models.py   # downloads base model + LoRA checkpoints
scripts\run_ui.bat
```

---

## Troubleshooting

| Problem | Solution |
|---------|---------|
| `No module named 'cosyvoice'` | Run `pip install -e .` from inside `BayanSynthTTS/`. The `.bat` scripts need the venv **inside** `BayanSynthTTS/` (i.e. `BayanSynthTTS/.venv/`). |
| `No LLM checkpoint found` | Run `python scripts/setup_models.py` or manually copy `.pt` to `checkpoints/llm/` |
| `mishkal not found` | `pip install mishkal` |
| No audio generated | Check console for the specific mode that failed; verify `voices/default.wav` exists |
| MP3/M4A upload fails | Install ffmpeg: `winget install ffmpeg` (Windows) or `sudo apt install ffmpeg` (Linux) |
| `ONNX CUDAExecutionProvider not available` | Expected on CPU-only machines or when `onnxruntime-gpu` is not installed. Inference still works on CPU. |
| First-run downloads from modelscope.cn | Expected — the `wetext` text normalizer downloads its model (~30 MB) once to `~/.cache/modelscope`. It's cached after the first run. |
| `scripts\run_ui.bat` says "venv not found" | Make sure you created the venv **inside** `BayanSynthTTS/` with `python -m venv .venv` and ran `pip install -r requirements.txt && pip install -e .` from there. |
| `huggingface_hub` version conflict | Keep `huggingface_hub<1.0` as pinned. If you see errors, run `pip install "huggingface_hub<1.0"` |

---

## License

Apache 2.0 — See [LICENSE](../LICENSE)

The underlying CosyVoice3 model is subject to its own license.  
LoRA checkpoints trained on Common Voice Arabic data are released under CC‑BY 4.0.
