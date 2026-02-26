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


def has_harakat(text: str) -> bool:
    """Check if text already contains Arabic diacritics."""
    return any(c in HARAKAT for c in text)


def strip_harakat(text: str) -> str:
    """Remove Arabic diacritics from text."""
    return "".join(c for c in text if c not in HARAKAT)


def detect_diacritization_ratio(text: str) -> float:
    """Return the fraction of Arabic words that have at least one diacritic."""
    word_re = re.compile(r"[\u0600-\u06FF]+")
    words = word_re.findall(text)
    if not words:
        return 0.0
    diac_count = sum(1 for w in words if has_harakat(w))
    return diac_count / len(words)


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
                return result if result else text
        return text

    for _, func in BACKENDS:
        result = func(text)
        if result is not None:
            return result

    return text


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
