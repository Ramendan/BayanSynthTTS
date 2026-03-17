from bayansynthtts import BayanSynthTTS
import soundfile as sf

# Correctly pre-diacritized text (mishkal mis-diacritizes مرحباً as مُرْحِبًا)
text = "مَرْحَبًا، أَنَا بَيَانْسِينْث، نِظَامٌ لِتَوْلِيدِ الْكَلَامِ الْعَرَبِيِّ."
print("text:", text)
tts = BayanSynthTTS()
audio = tts.synthesize(text, auto_tashkeel=False, speed=0.95)
print(f"duration {len(audio)/22050:.2f}s")
sf.write("samples/01_basic.wav", audio, 22050)
print("wrote samples/01_basic.wav")
