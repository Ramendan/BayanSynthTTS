# BayanSynthTTS Voices

This directory contains reference voice files used for synthesis and voice cloning.

## Default Voice

`default.wav` is the default reference voice.  
**Replace it with your own Arabic voice recording for best results.**

## Requirements for a Good Voice

| Property | Value |
|----------|-------|
| Duration | 5–15 seconds |
| Format   | WAV, MP3, FLAC, OGG, M4A (auto-converted) |
| Content  | Natural Arabic speech, clear pronunciation |
| Room     | Quiet room, no echo or background noise |
| Mic      | Any decent microphone or phone |

## How to Change the Default Voice

1. Record a clean 5–15 second clip of the voice you want
2. Save it anywhere you like
3. **Option A** — Replace the file directly:
   ```
   cp my_voice.wav BayanSynthTTS/voices/default.wav
   ```
4. **Option B** — Edit `conf/models.yaml`:
   ```yaml
   defaults:
     voice: "voices/my_new_voice.wav"
   ```
5. **Option C** — Pass at runtime:
   ```python
   audio = tts.synthesize("مَرْحَباً", ref_audio="my_voice.mp3")
   ```

## Free Arabic Voice Sources

- [Common Voice Arabic](https://commonvoice.mozilla.org/ar) — Public domain
- [OpenSLR Arabic](https://www.openslr.org/resources.php) — Various licenses
- Record your own with `BayanSynthTTS/scripts/record_my_voice.bat`

## Multiple Voices

You can keep multiple voice files and switch between them:
```python
tts.synthesize("مَرْحَباً", ref_audio=tts.get_voice_path("speaker2.wav"))
```

Or list all available:
```python
print(tts.list_voices())
```
