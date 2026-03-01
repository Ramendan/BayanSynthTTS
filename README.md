# BayanSynthTTS

**Arabic Text-to-Speech powered by CosyVoice3 with LoRA fine-tuning.**

> Just text in, speech out. No training code or data processing required. Inference only.

## Features

| Feature | Details |
|---------|---------|
| Arabic TTS | Natural-sounding Modern Standard Arabic |
| Auto-Tashkeel | Automatic diacritization via mishkal (always on by default) |
| Voice Cloning | Clone any voice from a 5-15 s clip (WAV/MP3/OGG/M4A/FLAC) |
| Example voices | Two reference voices (`default.wav` and `muffled-talking.wav`) are included; add your own to `voices/` |
| Speed control | Slow down or speed up synthesis (0.5–2.0×) |
| Streaming | Chunk-by-chunk audio generation |
| Gradio UI | Simple web interface included |
| CLI | One-liner inference from terminal |
| Multilingual base | CosyVoice3 supports many languages; Arabic LoRA ships by default |

---

> **Multilingual note:** the underlying CosyVoice3 base model is trained for zero-shot
> synthesis across a wide range of languages. BayanSynthTTS currently defaults to an
> Arabic-conditioned LoRA checkpoint and delivers the best results in Modern Standard
> Arabic. You are free to plug in other LoRA files (not provided here) for additional
> languages, though quality may vary.

---

## Audio Demos

All clips below were generated with this library. No post-processing applied.

> GitHub does not support embedded audio players in Markdown files.  
> Click any link below to open the file page with a built-in player.

| # | Description | Duration | Listen |
|---|-------------|-----|--------|
| 1 | Basic synthesis, auto-tashkeel | ~6 s | [01_basic.wav](samples/01_basic.wav) |
| 2 | Pre-diacritized text, mishkal off | ~7 s | [02_prediacritized.wav](samples/02_prediacritized.wav) |
| 3 | Voice cloning from muffled reference | ~10 s | [03_voice_cloning.wav](samples/03_voice_cloning.wav) |
|   ↳ | Reference clip used above | 10 s | 🎤 [ref_voice_muffled.wav](samples/ref_voice_muffled.wav) |
| 4 | Longer passage, AI topic, 3 sentences | ~15 s | [04_long_text.wav](samples/04_long_text.wav) |
| 5 | Slow speed (0.80x) | ~10 s | [05_slow_speed.wav](samples/05_slow_speed.wav) |
| 6 | Fast speed (1.20x) | ~5 s | [06_fast_speed.wav](samples/06_fast_speed.wav) |
| 7 | Phonetics test: halqiyyat, tanwin, shaddah | ~7 s | [07_phonetics.wav](samples/07_phonetics.wav) |
| 8 | Flow and rhythm, connected speech | ~10 s | [08_flow.wav](samples/08_flow.wav) |
| 9 | Challenge: identical root, different diacritics | ~5 s | [09_challenge.wav](samples/09_challenge.wav) |
| 10 | Phonetics, alternate seed (seed=99) | ~4 s | [10_phonetics_s2.wav](samples/10_phonetics_s2.wav) |
| 11 | Flow, alternate seed (seed=99) | ~9 s | [11_flow_s2.wav](samples/11_flow_s2.wav) |

---

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

> The CosyVoice3 inference engine and Matcha-TTS decoder are **bundled directly in this repo**. No external private repos required.
>
> **Example voices:** two reference clips (`default.wav` and `muffled-talking.wav`) live in `voices/`. Drop additional 5-15 s recordings there and they automatically appear in the CLI/UI dropdown.

### 2. Download models

```bash
python scripts/setup_models.py
```

This downloads everything automatically:
- CosyVoice3 base weights (~2 GB) from Hugging Face → `pretrained_models/CosyVoice3/`
- Arabic LoRA checkpoint from Hugging Face → `checkpoints/llm/epoch_28_whole.pt`
- Verifies the checkpoint SHA-256

> On Windows you can also double-click `scripts\setup_models.bat`.

### 3. Run

**Web UI:**
```bash
scripts\run_ui.bat            # Windows GUI launcher
python bayansynthtts/app.py   # Cross-platform (run from inside BayanSynthTTS/)
```

---

## Python API

### Basic synthesis (auto-tashkeel on by default)

```python
from bayansynthtts import BayanSynthTTS

tts = BayanSynthTTS()

# Plain Arabic - mishkal automatically adds full diacritics before synthesis
audio = tts.synthesize("مرحباً أنا بيان سينث، نظام لتوليد الكلام العربي")
tts.save_wav(audio, "output.wav")
```

> **auto_tashkeel=True** (default) runs the text through **mishkal** before inference.  
> Plain unvocalized Arabic produces natural, correctly-stressed speech.  
> Pass `auto_tashkeel=False` only when your text is already fully diacritized.

**[Listen: 01_basic.wav](samples/01_basic.wav)**

---

### Pre-diacritized text (skip auto-tashkeel)

```python
# When text is already fully vowelled, disable mishkal to preserve exact pronunciation
audio = tts.synthesize(
    "إِنَّ اللُّغَةَ الْعَرَبِيَّةَ كَنْزٌ مِنَ الثَّقَافَةِ وَالتُّرَاثِ، "
    "وَهِيَ لُغَةُ الْقُرْآنِ الْكَرِيمِ وَالشِّعْرِ الْعَرَبِيِّ الْعَرِيقِ.",
    auto_tashkeel=False,
)
tts.save_wav(audio, "output.wav")
```

**[Listen: 02_prediacritized.wav](samples/02_prediacritized.wav)**

---

### Voice cloning

```python
# Use the bundled muffled-talking voice (trim it to 5-15 s first for best results)
import soundfile as sf
import numpy as np

# Trim reference to 10 s
data, sr = sf.read("voices/muffled-talking.wav", dtype="float32")
sf.write("voices/muffled_trim.wav", data[:sr * 10], sr, subtype="PCM_16")

audio = tts.synthesize(
    "هَذَا الصَّوْتُ مُسْتَنْسَخٌ مِنْ مَقْطَعٍ صَوْتِيٍّ قَصِيرٍ. "
    "يُمْكِنُكَ اسْتِخْدَامُ أَيِّ مَقْطَعٍ بِمُدَّةِ خَمْسٍ إِلَى خَمْسَ عَشَرَةَ ثَانِيَةً.",
    ref_audio="voices/muffled_trim.wav",
    auto_tashkeel=False,
)
tts.save_wav(audio, "output.wav")

# Or clone any clip in any format (mp3/m4a/ogg/flac, auto-converted internally)
audio = tts.synthesize("مَرْحَباً", ref_audio="my_voice.mp3")
```

> **Tip:** Keep reference clips to **5-15 seconds**: single speaker, quiet room, no music.  
> Longer clips (>15 s) cause `instruct2` to fail silently and fall back to a lower-quality mode.

**Reference clip used above:** [ref_voice_muffled.wav](samples/ref_voice_muffled.wav) *(muffled-talking.wav trimmed to 10 s)*  
**[Listen: 03_voice_cloning.wav](samples/03_voice_cloning.wav)**

---

### Longer text

```python
audio = tts.synthesize(
    "الذكاء الاصطناعي هو أحد أبرز التطورات التكنولوجية في عصرنا الحديث. "
    "يعتمد على تحليل كميات ضخمة من البيانات لاستخلاص أنماط معقدة. "
    "ومن أبرز تطبيقاته نظم التعرف على الصوت وترجمة اللغات وتوليد النصوص.",
    auto_tashkeel=True,
)
tts.save_wav(audio, "output.wav")
```

**[Listen: 04_long_text.wav](samples/04_long_text.wav)**

---

### Phonetics test: halqiyyat, tanwin, shaddah

Designed to exercise pharyngeal/velar consonants, gemination, and nunation at once:

```python
audio = tts.synthesize(
    "الْجَوْدَةُ الْعَالِيَةُ لِتَقْنِيَّاتِ الذَّكَاءِ الاصْطِنَاعِيِّ "
    "تُسَاهِمُ فِي بِنَاءِ مُسْتَقْبَلٍ بَاهِرٍ لِلْأَجْيَالِ.",
    auto_tashkeel=False,
)
tts.save_wav(audio, "output.wav")
```

**[Listen: 07_phonetics.wav](samples/07_phonetics.wav)** *(seed=42)*  
**[Listen: 10_phonetics_s2.wav](samples/10_phonetics_s2.wav)** *(seed=99, different prosody variation)*

---

### Flow and Rhythm test: connected speech

Tests natural sandhi, liaison, and intonation across a multi-clause sentence:

```python
audio = tts.synthesize(
    "إِنَّ نِظَامَ بَيَانِ سِينْث يَهْدِفُ إِلَى تَقْدِيمِ تَجْرِبَةٍ صَوْتِيَّةٍ فَرِيدَةٍ، "
    "تَجْمَعُ بَيْنَ دِقَّةِ النُّطْقِ وَجَمَالِ الْأَدَاءِ.",
    auto_tashkeel=False,
)
tts.save_wav(audio, "output.wav")
```

**[Listen: 08_flow.wav](samples/08_flow.wav)** *(seed=42)*  
**[Listen: 11_flow_s2.wav](samples/11_flow_s2.wav)** *(seed=99, different prosody variation)*

---

### Challenge test: tashkeel disambiguation

All five ع-rooted words differ **only** by their diacritics; correct rendering proves the model reads harakat accurately:

```python
# عَلِم (he knew) vs عَالِم (scholar) vs عَلَم (flag) vs عِلْم (knowledge)
audio = tts.synthesize(
    "عَلِمَ الْعَالِمُ أَنَّ الْعَلَمَ يَعْلُو بِالْعِلْمِ، "
    "فَاسْتَعْلَمَ عَنْ عُلُومِ الْأَوَّلِينَ.",
    auto_tashkeel=False,
)
tts.save_wav(audio, "output.wav")
```

**[Listen: 09_challenge.wav](samples/09_challenge.wav)**

---

### Speed control

```python
TEXT = "مَرْحَباً بِكُمْ فِي بَيَانْ سِينْثِ. هَذَا تَوْلِيدٌ بِسُرْعَةٍ مُخَفَّضَةٍ لِلتَّوْضِيحِ."

# Slower speech (0.80×)
audio = tts.synthesize(TEXT, speed=0.80, auto_tashkeel=False)
tts.save_wav(audio, "slow.wav")

# Faster speech (1.20×)
audio = tts.synthesize(
    "مَرْحَباً بِكُمْ فِي بَيَانْ سِينْثِ. هَذَا تَوْلِيدٌ بِسُرْعَةٍ مُرْتَفَعَةٍ لِلتَّوْضِيحِ.",
    speed=1.20,
    auto_tashkeel=False,
)
tts.save_wav(audio, "fast.wav")
```

**[Listen slow: 05_slow_speed.wav](samples/05_slow_speed.wav)**  
**[Listen fast: 06_fast_speed.wav](samples/06_fast_speed.wav)**

---

### Streaming output

```python
for chunk in tts.synthesize("النص العربي الطويل...", stream=True):
    # chunk is a numpy array; feed to audio device or accumulate
    pass
```

---

### Save directly to file

```python
duration = tts.synthesize_to_file("مَرْحَباً بِكُمْ", "output.wav")
print(f"Generated {duration:.1f}s of audio")
```

---

### List available voices

```python
print(tts.list_voices())   # e.g. ['default.wav', 'muffled-talking.wav']
audio = tts.synthesize("مَرْحَباً", ref_audio=tts.get_voice_path("muffled-talking.wav"))
```

---

## Swapping the LoRA Checkpoint

### Via `conf/models.yaml` (recommended, no code changes)

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

## Adding Your Own Voices

Drop any 5-15 second Arabic clip into `voices/`. Supported formats: WAV, MP3, FLAC, OGG, M4A. Non-WAV files are auto-converted at runtime.

New files are picked up automatically. No configuration changes needed.

```python
from bayansynthtts import BayanSynthTTS

tts = BayanSynthTTS()
print(tts.list_voices())  # e.g. ['default.wav', 'muffled-talking.wav', 'my_voice.wav']
```

```bash
bayansynthtts "مرحبا" --voice voices/my_voice.wav
```

---

## Changing the Default Voice

**Option A: Replace the file:**
```bash
cp my_voice.wav BayanSynthTTS/voices/default.wav
```

**Option B: Edit `conf/models.yaml`:**
```yaml
defaults:
  voice: "voices/my_voice.wav"
```

**Option C: Pass at runtime:**
```python
audio = tts.synthesize("مَرْحَباً", ref_audio="path/to/my_voice.mp3")
```

**Voice quality guidelines:** 5-15 seconds, quiet room, clear Arabic speech, no music.

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
│   ├── muffled-talking.wav # Additional bundled voice
│   └── README.md
├── samples/                # Pre-generated audio demos (tracked in git)
│   ├── 01_basic.wav                #  "مرحباً أنا بيان سينث" - auto-tashkeel
│   ├── 02_prediacritized.wav       # fully-vowelled classical Arabic
│   ├── 03_voice_cloning.wav        # voice-cloned from muffled-talking.wav
│   ├── 04_long_text.wav            # ~15 s multi-sentence AI topic
│   ├── 05_slow_speed.wav           # speed=0.80
│   ├── 06_fast_speed.wav           # speed=1.20
│   ├── 07_phonetics.wav            # حلقيات / tanwin / shaddah test (seed=42)
│   ├── 08_flow.wav                 # flow & rhythm test (seed=42)
│   ├── 09_challenge.wav            # عَلِم/عَالِم/عَلَم tashkeel disambiguation
│   ├── 10_phonetics_s2.wav         # same as 07, seed=99 (prosody variation)
│   ├── 11_flow_s2.wav              # same as 08, seed=99 (prosody variation)
│   └── ref_voice_muffled.wav       # reference voice clip used for 03 (10 s)
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
| `text` | `str` | required | Arabic text (plain or diacritized) |
| `ref_audio` | `str` | default voice | Voice clone source (any format) |
| `instruct` | `str` | from config | Instruct prompt override |
| `speed` | `float` | `1.0` | Speed multiplier (0.5-2.0) |
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

## Troubleshooting

| Problem | Solution |
|---------|---------|
| `No module named 'cosyvoice'` | Run `pip install -e .` from inside `BayanSynthTTS/`. The `.bat` scripts need the venv **inside** `BayanSynthTTS/` (i.e. `BayanSynthTTS/.venv/`). If you encounter an error about `setuptools.backends`, install/upgrade setuptools first (`pip install --upgrade setuptools`). |
| `ModuleNotFoundError: No module named 'whisper'` | The bundled CosyVoice engine uses the `whisper` package; install it with `pip install openai-whisper` or `pip install whisper`. |
| `No LLM checkpoint found` | Run `python scripts/setup_models.py` or manually copy `.pt` to `checkpoints/llm/` |
| `mishkal not found` | `pip install mishkal` |
| No audio generated | Check console for the specific mode that failed; verify `voices/default.wav` exists |
| MP3/M4A upload fails | Install ffmpeg: `winget install ffmpeg` (Windows) or `sudo apt install ffmpeg` (Linux) |
| `ONNX CUDAExecutionProvider not available` | Expected on CPU-only machines or when `onnxruntime-gpu` is not installed. Inference still works on CPU. |
| First-run downloads from modelscope.cn | Expected. The `wetext` text normalizer downloads its model (~30 MB) once to `~/.cache/modelscope`. It is cached after the first run. |
| `scripts\run_ui.bat` says "venv not found" | Make sure you created the venv **inside** `BayanSynthTTS/` with `python -m venv .venv` and ran `pip install -r requirements.txt && pip install -e .` from there. |
| `huggingface_hub` version conflict | Keep `huggingface_hub<1.0` as pinned. If you see errors, run `pip install "huggingface_hub<1.0"` |

---

## License

Apache 2.0. See [LICENSE](../LICENSE)

The underlying CosyVoice3 model is subject to its own license.  
LoRA checkpoints trained on Common Voice Arabic data are released under CC‑BY 4.0.
