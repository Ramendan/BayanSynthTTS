"""Upload WAV samples + model card to HuggingFace."""
from huggingface_hub import HfApi
import os

REPO = "Ramendan/BayanSynthTTS-checkpoints"
SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "samples")

api = HfApi()

# Upload WAV files
wavs = [
    "01_basic.wav",
    "02_prediacritized.wav",
    "03_voice_cloning.wav",
    "04_long_text.wav",
    "05_slow_speed.wav",
    "06_fast_speed.wav",
    "07_phonetics.wav",
    "08_flow.wav",
    "09_challenge.wav",
    "10_phonetics_s2.wav",
    "11_flow_s2.wav",
    "12_instruct.wav",
    "ref_voice_muffled.wav",
]

for wav in wavs:
    local = os.path.join(SAMPLES_DIR, wav)
    if not os.path.exists(local):
        print(f"SKIP (missing): {wav}")
        continue
    print(f"Uploading {wav} ...")
    api.upload_file(
        path_or_fileobj=local,
        path_in_repo=f"samples/{wav}",
        repo_id=REPO,
        repo_type="model",
    )
    print(f"  done: {wav}")

# Upload model card
readme = os.path.join(os.path.dirname(__file__), "_hf_readme.md")
print("Uploading README.md (model card) ...")
api.upload_file(
    path_or_fileobj=readme,
    path_in_repo="README.md",
    repo_id=REPO,
    repo_type="model",
)
print("done: README.md")
print("All uploads complete!")
