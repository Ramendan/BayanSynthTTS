---
license: apache-2.0
language:
  - ar
tags:
  - tts
  - arabic
  - cosyvoice
  - lora
  - speech-synthesis
---

# BayanSynthTTS

**Arabic Text-to-Speech powered by CosyVoice3 with LoRA fine-tuning.**

> Text in. Speech out. Inference only without training or preprocessing.

**GitHub:** [Ramendan/BayanSynthTTS](https://github.com/Ramendan/BayanSynthTTS)

## Features

| Feature | Details |
|---------|---------|
| Arabic TTS | Natural-sounding Modern Standard Arabic |
| Auto-Tashkeel | Automatic diacritization via mishkal (always on by default) |
| Voice Cloning | Clone any voice from a 5-15 s clip (WAV/MP3/OGG/M4A/FLAC) |
| Example voices | Two reference voices (`default.wav` and `muffled-talking.wav`) are included; add your own to `voices/` |
| Speed control | Slow down or speed up synthesis (0.5–2.0×) |
| LoRA Swapping | Change checkpoints via `conf/models.yaml` no code edits |
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

All samples were generated with this library. No post-processing applied.

| # | Description | Duration |
|---|-------------|----------|
| 1 | Basic synthesis, pre-diacritized | ~5 s |
| 2 | Pre-diacritized text, mishkal off | ~4 s |
| 3 | Voice cloning from muffled reference | ~10 s |
| 4 | Longer passage, AI topic, 3 sentences | ~17 s |
| 5 | Slow speed (0.80x) | ~10 s |
| 6 | Fast speed (1.20x) | ~5 s |
| 7 | Phonetics test: halqiyyat, tanwin, shaddah | ~7 s |
| 8 | Flow and rhythm, connected speech | ~9 s |
| 9 | Challenge: identical root, different diacritics | ~5 s |
| 10 | Phonetics, alternate seed (seed=17) | ~9 s |
| 11 | Flow, alternate seed (seed=99) | ~10 s |
| 12 | Instruct prompt: warm newsreader style | ~8 s |

---

### 1. Basic synthesis

> مَرْحَبًا، أَنَا بَيَانْسِينْث، نِظَامٌ لِتَوْلِيدِ الْكَلَامِ الْعَرَبِيِّ.
>
> *Hello, I am BayanSynth, a system for generating Arabic speech.*

```python
from bayansynthtts import BayanSynthTTS
tts = BayanSynthTTS()
audio = tts.synthesize(
    "مَرْحَبًا، أَنَا بَيَانْسِينْث، نِظَامٌ لِتَوْلِيدِ الْكَلَامِ الْعَرَبِيِّ.",
    auto_tashkeel=False,
)
tts.save_wav(audio, "output.wav")
```

<audio controls src="https://huggingface.co/Ramendan/BayanSynthTTS-checkpoints/resolve/main/samples/01_basic.wav"></audio>

---

### 2. Pre-diacritized text (mishkal off)

> إِنَّ اللُّغَةَ الْعَرَبِيَّةَ كَنْزٌ مِنَ الثَّقَافَةِ وَالتُّرَاثِ.
>
> *The Arabic language is a treasure of culture and heritage.*

```python
audio = tts.synthesize(
    "إِنَّ اللُّغَةَ الْعَرَبِيَّةَ كَنْزٌ مِنَ الثَّقَافَةِ وَالتُّرَاثِ.",
    auto_tashkeel=False,
)
```

<audio controls src="https://huggingface.co/Ramendan/BayanSynthTTS-checkpoints/resolve/main/samples/02_prediacritized.wav"></audio>

---

### 3. Voice cloning

> هَذَا الصَّوْتُ مُسْتَنْسَخٌ مِنْ مَقْطَعٍ صَوْتِيٍّ قَصِيرٍ. يُمْكِنُكَ اسْتِخْدَامُ أَيِّ مَقْطَعٍ بِمُدَّةِ خَمْسٍ إِلَى خَمْسَ عَشَرَةَ ثَانِيَةً.
>
> *This voice is cloned from a short audio clip. You can use any clip between five and fifteen seconds.*

```python
audio = tts.synthesize(
    "هَذَا الصَّوْتُ مُسْتَنْسَخٌ مِنْ مَقْطَعٍ صَوْتِيٍّ قَصِيرٍ. "
    "يُمْكِنُكَ اسْتِخْدَامُ أَيِّ مَقْطَعٍ بِمُدَّةِ خَمْسٍ إِلَى خَمْسَ عَشَرَةَ ثَانِيَةً.",
    ref_audio="voices/muffled_trim.wav",
    auto_tashkeel=False,
)
```

**Reference clip:** <audio controls src="https://huggingface.co/Ramendan/BayanSynthTTS-checkpoints/resolve/main/samples/ref_voice_muffled.wav"></audio>

**Result:** <audio controls src="https://huggingface.co/Ramendan/BayanSynthTTS-checkpoints/resolve/main/samples/03_voice_cloning.wav"></audio>

---

### 4. Longer passage (auto-tashkeel, speed 0.88)

> الذكاء الاصطناعي هو أحد أبرز التطورات التكنولوجية في عصرنا الحديث. يعتمد على تحليل كميات ضخمة من البيانات لاستخلاص أنماط معقدة. ومن أبرز تطبيقاته نظم التعرف على الصوت وترجمة اللغات وتوليد النصوص.
>
> *Artificial intelligence is one of the most prominent technological advances of our era. It relies on analyzing massive amounts of data to extract complex patterns. Among its most notable applications: speech recognition, language translation, and text generation.*

```python
audio = tts.synthesize(
    "الذكاء الاصطناعي هو أحد أبرز التطورات التكنولوجية في عصرنا الحديث. "
    "يعتمد على تحليل كميات ضخمة من البيانات لاستخلاص أنماط معقدة. "
    "ومن أبرز تطبيقاته نظم التعرف على الصوت وترجمة اللغات وتوليد النصوص.",
    auto_tashkeel=True,
    speed=0.88,
)
```

<audio controls src="https://huggingface.co/Ramendan/BayanSynthTTS-checkpoints/resolve/main/samples/04_long_text.wav"></audio>

---

### 5. Speed control

> مَرْحَباً بِكُمْ فِي بَيَانْسِينْثِ. هَذَا تَوْلِيدٌ بِسُرْعَةٍ مُخَفَّضَةٍ لِلتَّوْضِيحِ.
>
> *Welcome to BayanSynth. This is synthesis at reduced speed for demonstration.*

```python
TEXT = "مَرْحَباً بِكُمْ فِي بَيَانْسِينْثِ. هَذَا تَوْلِيدٌ بِسُرْعَةٍ مُخَفَّضَةٍ لِلتَّوْضِيحِ."
audio = tts.synthesize(TEXT, speed=0.80, auto_tashkeel=False)
```

**Slow (0.80×):** <audio controls src="https://huggingface.co/Ramendan/BayanSynthTTS-checkpoints/resolve/main/samples/05_slow_speed.wav"></audio>

**Fast (1.20×):** <audio controls src="https://huggingface.co/Ramendan/BayanSynthTTS-checkpoints/resolve/main/samples/06_fast_speed.wav"></audio>

---

### 6. Phonetics test: halqiyyat, tanwin, shaddah

Designed to exercise pharyngeal/velar consonants, gemination, and nunation at once:

> الْجَوْدَةُ الْعَالِيَةُ لِتَقْنِيَّاتِ الذَّكَاءِ الاصْطِنَاعِيِّ تُسَاهِمُ فِي بِنَاءِ مُسْتَقْبَلٍ بَاهِرٍ لِلْأَجْيَالِ.
>
> *The high quality of AI technologies contributes to building a brilliant future for generations to come.*

```python
audio = tts.synthesize(
    "الْجَوْدَةُ الْعَالِيَةُ لِتَقْنِيَّاتِ الذَّكَاءِ الاصْطِنَاعِيِّ "
    "تُسَاهِمُ فِي بِنَاءِ مُسْتَقْبَلٍ بَاهِرٍ لِلْأَجْيَالِ.",
    auto_tashkeel=False,
)
```

**seed=42:** <audio controls src="https://huggingface.co/Ramendan/BayanSynthTTS-checkpoints/resolve/main/samples/07_phonetics.wav"></audio>

**seed=17 (different prosody):** <audio controls src="https://huggingface.co/Ramendan/BayanSynthTTS-checkpoints/resolve/main/samples/10_phonetics_s2.wav"></audio>

---

### 7. Flow & rhythm test: connected speech

Tests natural sandhi, liaison, and intonation across a multi-clause sentence:

> إِنَّ نِظَامَ بَيَانِسِينْث يَهْدِفُ إِلَى تَقْدِيمِ تَجْرِبَةٍ صَوْتِيَّةٍ فَرِيدَةٍ، تَجْمَعُ بَيْنَ دِقَّةِ النُّطْقِ وَجَمَالِ الْأَدَاءِ.
>
> *BayanSynth aims to deliver a unique voice experience that combines precise pronunciation with beauty of delivery.*

```python
audio = tts.synthesize(
    "إِنَّ نِظَامَ بَيَانِسِينْث يَهْدِفُ إِلَى تَقْدِيمِ تَجْرِبَةٍ صَوْتِيَّةٍ فَرِيدَةٍ، "
    "تَجْمَعُ بَيْنَ دِقَّةِ النُّطْقِ وَجَمَالِ الْأَدَاءِ.",
    auto_tashkeel=False,
)
```

**seed=42:** <audio controls src="https://huggingface.co/Ramendan/BayanSynthTTS-checkpoints/resolve/main/samples/08_flow.wav"></audio>

**seed=99 (different prosody):** <audio controls src="https://huggingface.co/Ramendan/BayanSynthTTS-checkpoints/resolve/main/samples/11_flow_s2.wav"></audio>

---

### 8. Challenge: tashkeel disambiguation

All five ع-rooted words differ **only** by their diacritics; correct rendering proves the model reads harakat accurately:

> عَلِمَ الْعَالِمُ أَنَّ الْعَلَمَ يَعْلُو بِالْعِلْمِ، فَاسْتَعْلَمَ عَنْ عُلُومِ الْأَوَّلِينَ.
>
> *The scholar knew that the flag rises with knowledge, so he inquired about the sciences of the ancients.*

```python
audio = tts.synthesize(
    "عَلِمَ الْعَالِمُ أَنَّ الْعَلَمَ يَعْلُو بِالْعِلْمِ، "
    "فَاسْتَعْلَمَ عَنْ عُلُومِ الْأَوَّلِينَ.",
    auto_tashkeel=False,
)
```

<audio controls src="https://huggingface.co/Ramendan/BayanSynthTTS-checkpoints/resolve/main/samples/09_challenge.wav"></audio>

---

### 9. Instruct prompt: warm newsreader style

Pass a free-text style directive alongside the synthesis text to steer the speaker's tone, register, or delivery:

> مَرْحَباً بِكُمْ. هَذَا مِثَالٌ عَلَى اسْتِخْدَامِ التَّوْجِيهِ لِضَبْطِ أُسْلُوبِ الصَّوْتِ.
>
> *Welcome. This is an example of using an instruct prompt to control voice style.*

```python
audio = tts.synthesize(
    "مَرْحَباً بِكُمْ. هَذَا مِثَالٌ عَلَى اسْتِخْدَامِ التَّوْجِيهِ لِضَبْطِ أُسْلُوبِ الصَّوْتِ.",
    instruct="Speak in a warm, clear newsreader style with careful diction.",
    auto_tashkeel=False,
    seed=42,
)
```

<audio controls src="https://huggingface.co/Ramendan/BayanSynthTTS-checkpoints/resolve/main/samples/12_instruct.wav"></audio>

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

## Files in this repo

| File | Description |
|------|-------------|
| `epoch_28_whole.pt` | LoRA weights (LLM, 629 keys) — main checkpoint |
| `samples/*.wav` | Pre-generated audio demos |

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

```python
from bayansynthtts import BayanSynthTTS
tts = BayanSynthTTS()
print(tts.list_voices())  # e.g. ['default.wav', 'muffled-talking.wav', 'my_voice.wav']
```

```bash
bayansynthtts "مرحبا" --voice voices/my_voice.wav
```

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
| `No module named 'cosyvoice'` | Run `pip install -e .` from inside `BayanSynthTTS/` |
| `No LLM checkpoint found` | Run `python scripts/setup_models.py` |
| `mishkal not found` | `pip install mishkal` |
| No audio generated | Check console for the specific mode that failed; verify `voices/default.wav` exists |
| MP3/M4A upload fails | Install ffmpeg: `winget install ffmpeg` (Windows) or `sudo apt install ffmpeg` (Linux) |

---

## License

Apache 2.0.

The underlying CosyVoice3 model is subject to its own license.
LoRA checkpoints trained on Common Voice Arabic data are released under CC-BY 4.0.
