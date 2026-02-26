"""BayanSynthTTS — Arabic Text-to-Speech built on CosyVoice3.

Quick start::

    from bayansynthtts import BayanSynthTTS

    tts = BayanSynthTTS()                       # loads best checkpoints automatically
    audio = tts.synthesize("مَرْحَباً بِكُمْ")  # returns numpy array at 24 kHz
    tts.save_wav(audio, "output.wav")

Voice cloning (any audio format)::

    audio = tts.synthesize("مَرْحَباً", ref_audio="my_voice.mp3")

Auto-diacritization::

    from bayansynthtts import BayanSynthTTS, auto_diacritize

    text = auto_diacritize("مرحبا بكم")
    audio = tts.synthesize(text)

Audio utilities::

    from bayansynthtts import save_wav, load_ref_audio, convert_audio_to_wav

    # Convert any format to a temporary WAV path
    wav_path = convert_audio_to_wav("voice.mp3")

    # Load as a numpy-compatible tensor
    tensor, sr = load_ref_audio("voice.mp3")
"""

from bayansynthtts.inference import (
    BayanSynthTTS,
    save_wav,
    load_ref_audio,
    convert_audio_to_wav,
    SAMPLE_RATE,
)

from bayansynthtts.tashkeel import (
    auto_diacritize,
    has_harakat,
    strip_harakat,
    detect_diacritization_ratio,
    list_available_backends,
)

__version__ = "1.0.0"
__all__ = [
    # Core TTS
    "BayanSynthTTS",
    # Audio I/O utilities
    "save_wav",
    "load_ref_audio",
    "convert_audio_to_wav",
    "SAMPLE_RATE",
    # Tashkeel (Arabic diacritization)
    "auto_diacritize",
    "has_harakat",
    "strip_harakat",
    "detect_diacritization_ratio",
    "list_available_backends",
]
