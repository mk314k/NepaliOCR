"""
Linguistic post-processing and Unicode normalization for Devanagari OCR output.
"""
import re
from typing import List, Tuple

_VOWEL_MATRAS = set("ािीुूृेैोौ")
_MODIFIERS = set("ंँः")
_HALANT = "्"

_PREETI_DIGIT_TOKENS: Tuple[Tuple[str, str], ...] = (
    ("ण्", "०"),
    ("ज्ञ", "१"),
    ("द्द", "२"),
    ("द्ध", "४"),
    ("घ", "३"),
    ("छ", "५"),
    ("ट", "६"),
    ("ठ", "७"),
    ("ड", "८"),
    ("ढ", "९"),
)

_PREETI_NUMERIC_RE = re.compile(
    r"^(?:ण्|ज्ञ|द्द|द्ध|घ|छ|ट|ठ|ड|ढ|[०-९0-9])+(?:[।\.](?:ण्|ज्ञ|द्द|द्ध|घ|छ|ट|ठ|ड|ढ|[०-९0-9])+)*$"
)


def _recover_preeti_number(token: str) -> str:
    if not _PREETI_NUMERIC_RE.match(token):
        return token
    out = token
    for src, dst in _PREETI_DIGIT_TOKENS:
        out = out.replace(src, dst)
    out = out.replace("।", ".")
    return out


def postprocess_devanagari(text: str) -> str:
    if not text:
        return text

    s = _recover_preeti_number(text.strip())
    s = s.replace("र्इ", "ई")
    s = (
        s.replace("अाे", "ओ")
        .replace("आे", "ओ")
        .replace("अो", "ओ")
        .replace("अाै", "औ")
        .replace("आै", "औ")
        .replace("अौ", "औ")
        .replace("अा", "आ")
        .replace("एे", "ऐ")
        .replace("एै", "ऐ")
        .replace("ाे", "ो")
        .replace("ाै", "ौ")
        .replace("्ा", "")
    )

    dedup_chars: List[str] = []
    for ch in s:
        if dedup_chars:
            prev = dedup_chars[-1]
            if ch in _VOWEL_MATRAS and prev in _VOWEL_MATRAS:
                continue
            if ch == prev and (ch in _MODIFIERS or ch == _HALANT):
                continue
        dedup_chars.append(ch)
    return "".join(dedup_chars)


linguistic_postprocess = postprocess_devanagari
