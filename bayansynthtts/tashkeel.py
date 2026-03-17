#!/usr/bin/env python3
"""BayanSynthTTS Auto-Tashkeel Module.

Provides automatic Arabic diacritization (tashkeel/harakat) with a
multi-backend fallback chain:

  1. tashkeel (https://github.com/Wikipedia-Pronunciation/Tashkeel) — neural (best quality)
  2. mishkal  (https://github.com/linuxscout/mishkal) — rule-based (reliable, always included)
  3. Pass-through (return text as-is if all backends fail)

Usage::

    from bayansynthtts.tashkeel import auto_diacritize
    diacritized = auto_diacritize("مرحبا بكم في اختبار النظام")
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

# Arabic diacritics (harakat) Unicode range
HARAKAT = set("\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0670\u0657\u0658")

_BACKEND_CACHE: dict[str, object] = {}
_AR_WORD_RE = re.compile(r"[\u0600-\u06FF]+")
_AR_LETTER_RE = re.compile(r"[\u0621-\u064A]")

# Letters that can carry harakat when acting as consonants.
# Bare alef ا (U+0627) is excluded because it almost always acts as a long
# vowel in un-vowelled text and adding fatha to it produces wrong readings.
_VOCALIZE_LETTERS: frozenset[str] = frozenset(
    "\u0621"  # ء
    "\u0622"  # آ
    "\u0623"  # أ
    "\u0624"  # ؤ
    "\u0625"  # إ
    "\u0626"  # ئ
    "\u0628\u062A\u062B\u062C\u062D\u062E"  # ب ت ث ج ح خ
    "\u062F\u0630\u0631\u0632\u0633\u0634"  # د ذ ر ز س ش
    "\u0635\u0636\u0637\u0638\u0639\u063A"  # ص ض ط ظ ع غ
    "\u0641\u0642\u0643\u0644\u0645\u0646"  # ف ق ك ل م ن
    "\u0647\u0648\u064A"                    # ه و ي
)

# Word-level overrides for words that mishkal consistently fails to diacritize
# correctly.  Keys are the bare (no-harakat) form; values are the diacritized
# form.  Only applied when mishkal produces ZERO harakat for that word.
_WORD_OVERRIDES: dict[str, str] = {
    # Greetings / common expressions
    "اهلا":   "أَهْلًا",
    "اهلاً":  "أَهْلًا",
    "وسهلا":  "وَسَهْلًا",
    "وسهلاً": "وَسَهْلًا",
    "ياهلا":  "يَاهْلًا",
    "شكرا":   "شُكْرًا",
    "شكراً":  "شُكْرًا",
    "عفوا":   "عَفْوًا",
    "عفواً":  "عَفْوًا",
    "اسف":    "آسِف",
    "اسفا":   "آسِفًا",
    "اسفاً":  "آسِفًا",
    "صباحا":  "صَبَاحًا",
    "مساءا":  "مَسَاءً",
    "تفضل":   "تَفَضَّل",
    "تفضلي":  "تَفَضَّلِي",
    "تفضلوا": "تَفَضَّلُوا",
}


def has_harakat(text: str) -> bool:
    """Check if text already contains Arabic diacritics."""
    return any(c in HARAKAT for c in text)


def strip_harakat(text: str) -> str:
    """Remove Arabic diacritics from text."""
    return "".join(c for c in text if c not in HARAKAT)


def detect_diacritization_ratio(text: str) -> float:
    """Return the fraction of Arabic words that have at least one diacritic."""
    words = _AR_WORD_RE.findall(text)
    if not words:
        return 0.0
    diac_count = sum(1 for w in words if has_harakat(w))
    return diac_count / len(words)


def detect_letter_diacritization_ratio(text: str) -> float:
    """Return a letter-level diacritization score for Arabic text.

    Score = number of harakat marks / number of Arabic letters.
    """
    letters = _AR_LETTER_RE.findall(text)
    if not letters:
        return 0.0
    marks = sum(1 for c in text if c in HARAKAT)
    return marks / len(letters)


def _cleanup_diacritized_text(text: str) -> str:
    """Normalize known backend artefacts without changing intended wording."""
    out = unicodedata.normalize("NFC", text)
    # Fix broken words where backend inserts newlines or long gaps inside Arabic words.
    out = re.sub(r"(?<=[\u0621-\u064A])(?:\s*\n\s*|[ \t]{2,})(?=[\u0621-\u064A])", "", out)
    # Normalize any remaining whitespace.
    out = re.sub(r"[ \t]{2,}", " ", out)
    out = re.sub(r"\s*\n\s*", " ", out)
    return out.strip()


def _apply_fallback_harakat(word: str) -> str:
    """Apply basic fatha/sukun to a completely unvocalized Arabic word.

    This is a last-resort fallback: adds fatha after each vocalize-able
    consonant, and sukun after the final consonant.  Bare alef ا is
    skipped (it acts as a long vowel and adding fatha to it sounds wrong).

    The result is always better for TTS than zero diacritics.
    """
    if has_harakat(word):
        return word

    chars = list(word)
    consonant_idxs = [i for i, c in enumerate(chars) if c in _VOCALIZE_LETTERS]
    if not consonant_idxs:
        return word

    last_idx = consonant_idxs[-1]
    result: list[str] = []
    for i, c in enumerate(chars):
        result.append(c)
        if i in set(consonant_idxs):
            result.append("\u0652" if i == last_idx else "\u064E")  # sukun / fatha
    return "".join(result)


def _apply_word_fallbacks(diacritized: str, original: str) -> str:
    """For every Arabic word that still has zero harakat after tashkeel,
    check the override dictionary first, then fall back to the automatic
    fatha rule.  Words that were successfully vocalized are left alone.
    """
    orig_words = _AR_WORD_RE.findall(original)
    if not orig_words:
        return diacritized

    # Build a map of original-word → fallback replacement
    fallback_map: dict[str, str] = {}
    for ow in orig_words:
        # Only act on words that are still un-vowelled in the diacritized string.
        # We detect that by scanning the diacritized text for the same bare word.
        bare_ow = strip_harakat(ow)
        if bare_ow in fallback_map:
            continue
        override = _WORD_OVERRIDES.get(bare_ow)
        if override:
            fallback_map[bare_ow] = override
        else:
            fallback_map[bare_ow] = _apply_fallback_harakat(bare_ow)

    # Walk through diacritized text and replace un-vowelled Arabic words
    def _replace(match: re.Match) -> str:
        word = match.group(0)
        bare = strip_harakat(word)
        # Override dict always wins — these are cases we know mishkal gets wrong.
        if bare in _WORD_OVERRIDES:
            return _WORD_OVERRIDES[bare]
        # For words that still have zero harakat, apply fallback rule.
        if not has_harakat(word):
            return fallback_map.get(bare, word)
        return word

    return _AR_WORD_RE.sub(_replace, diacritized)


def _word_density(word: str) -> float:
    letters = _AR_LETTER_RE.findall(word)
    if not letters:
        return 0.0
    marks = sum(1 for c in word if c in HARAKAT)
    return marks / len(letters)


def _replace_arabic_words(template: str, replacement_words: list[str]) -> str:
    i = 0

    def _repl(match):
        nonlocal i
        if i < len(replacement_words):
            w = replacement_words[i]
            i += 1
            return w
        return match.group(0)

    return _AR_WORD_RE.sub(_repl, template)


def _build_hybrid_candidate(original: str, candidates: list[str]) -> Optional[str]:
    """Merge candidates per word by choosing highest letter-level diacritic density."""
    if not candidates:
        return None

    orig_words = _AR_WORD_RE.findall(original)
    if not orig_words:
        return None

    candidate_words = [_AR_WORD_RE.findall(c) for c in candidates]
    if not candidate_words:
        return None

    merged = []
    for idx, ow in enumerate(orig_words):
        variants = []
        for words in candidate_words:
            if idx < len(words):
                variants.append(words[idx])
        if not variants:
            merged.append(ow)
            continue
        best = max(variants, key=lambda w: (_word_density(w), len(w)))
        merged.append(best)

    base = candidates[0]
    return _cleanup_diacritized_text(_replace_arabic_words(base, merged))


# ── Backend 1: tashkeel (neural) ─────────────────────────────────────────
def _init_tashkeel():
    if "tashkeel" in _BACKEND_CACHE:
        return _BACKEND_CACHE["tashkeel"]
    try:
        from tashkeel.tashkeel import TashkeelModel
        model = TashkeelModel()
        model.load()
        _BACKEND_CACHE["tashkeel"] = model
        return model
    except Exception:
        _BACKEND_CACHE["tashkeel"] = None
        return None


def _diacritize_tashkeel(text: str) -> Optional[str]:
    model = _init_tashkeel()
    if model is None:
        return None
    try:
        return model.do_tashkeel(text)
    except Exception:
        return None


# ── Backend 2: mishkal (rule-based) ──────────────────────────────────────
def _init_mishkal():
    if "mishkal" in _BACKEND_CACHE:
        return _BACKEND_CACHE["mishkal"]
    try:
        import mishkal.tashkeel as mtashkeel
        vocalizer = mtashkeel.TashkeelClass()
        _BACKEND_CACHE["mishkal"] = vocalizer
        return vocalizer
    except Exception:
        _BACKEND_CACHE["mishkal"] = None
        return None


def _diacritize_mishkal(text: str) -> Optional[str]:
    vocalizer = _init_mishkal()
    if vocalizer is None:
        return None
    try:
        return vocalizer.tashkeel(text)
    except Exception:
        return None


# ── Public API ────────────────────────────────────────────────────────────

BACKENDS = [
    ("tashkeel", _diacritize_tashkeel),
    ("mishkal",  _diacritize_mishkal),
]


def list_available_backends() -> list[str]:
    """Return names of currently available tashkeel backends."""
    available = []
    for name, _ in BACKENDS:
        if name == "tashkeel" and _init_tashkeel() is not None:
            available.append(name)
        elif name == "mishkal" and _init_mishkal() is not None:
            available.append(name)
    return available


def auto_diacritize(
    text: str,
    backend: Optional[str] = None,
    skip_if_diacritized: bool = True,
    min_diac_ratio: float = 0.5,
) -> str:
    """Automatically add diacritics (tashkeel) to Arabic text.

    Args:
        text: Arabic text to diacritize.
        backend: Force a specific backend ("tashkeel" or "mishkal"),
                 or None for automatic fallback (tashkeel → mishkal).
        skip_if_diacritized: Skip if text already has sufficient diacritics.
        min_diac_ratio: Minimum diacritization ratio considered "already done".

    Returns:
        Diacritized text, or original if all backends fail.
    """
    text = unicodedata.normalize("NFC", text.strip())

    if not text:
        return text

    if skip_if_diacritized and detect_diacritization_ratio(text) >= min_diac_ratio:
        return text

    if backend:
        for name, func in BACKENDS:
            if name == backend:
                result = func(text)
                if not result:
                    return text
                cleaned = _cleanup_diacritized_text(result)
                return _apply_word_fallbacks(cleaned, text)
        return text

    candidates: list[str] = []
    for _, func in BACKENDS:
        result = func(text)
        if result:
            candidates.append(_cleanup_diacritized_text(result))

    if not candidates:
        return text

    hybrid = _build_hybrid_candidate(text, candidates)
    if hybrid:
        candidates.append(hybrid)

    # Prefer candidates with stronger letter-level coverage; use word-level ratio as tie-break.
    best = max(candidates, key=lambda t: (detect_letter_diacritization_ratio(t), detect_diacritization_ratio(t)))
    # Final pass: fill in any words that still have zero harakat.
    return _apply_word_fallbacks(best, text)


def get_backend_info() -> dict:
    """Return info about available backends for display in UI."""
    info = {"backends": [], "active": None}
    descriptions = {
        "tashkeel": "Neural (best quality)",
        "mishkal":  "Rule-based (reliable)",
    }
    for name, _ in BACKENDS:
        available = False
        if name == "tashkeel":
            available = _init_tashkeel() is not None
        elif name == "mishkal":
            available = _init_mishkal() is not None

        info["backends"].append({
            "name": name,
            "available": available,
            "type": descriptions.get(name, "unknown"),
        })
        if available and info["active"] is None:
            info["active"] = name

    return info


if __name__ == "__main__":
    tests = [
        "مرحبا بكم في اختبار النظام",
        "الذكاء الاصطناعي يغير العالم",
        "مَرْحَباً بِكُمْ فِي اخْتِبَارِ النِّظَامِ",
    ]
    print("Available backends:", list_available_backends())
    for t in tests:
        print(f"Input:  {t}")
        print(f"Output: {auto_diacritize(t)}")
        print()
