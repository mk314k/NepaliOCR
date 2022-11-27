"""
Linguistic post-processing and Unicode normalization for Devanagari OCR output.
"""
import re
from typing import List, Tuple

_CONSONANTS = set(
    "कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह"
)
_VOWEL_MATRAS = set("ािीुूृेैोौ")
_MODIFIERS = set("ंँः")
_HALANT = "्"

# In Preeti encoding, unshifted number keys '0'..'9' produce these consonant/conjunct
# strings and '.' produces '।'. Numeric tokens in Preeti ground truth therefore map to
# sequences of these strings.
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
    """
    Convert tokens composed of Preeti unshifted number-row glyphs (e.g. `'द्द।द्द'`
    or `'द्दद्द'`) into Unicode Devanagari digits (`'२.२'`, `'२२'`).
    """
    if not _PREETI_NUMERIC_RE.match(token):
        return token
    out = token
    for src, dst in _PREETI_DIGIT_TOKENS:
        out = out.replace(src, dst)
    out = out.replace("।", ".")
    return out


def _reorder_visual_ikar_and_modifiers(token: str) -> str:
    """
    Enforce canonical Unicode Devanagari codepoint ordering within a word:
      1. Move any leading/pre-consonant `'ि'` (visual order) to immediately follow
         its consonant or conjunct cluster (`C1 + ् + C2 + ि`).
      2. Ensure vowel matras precede anusvara/chandrabindu/visarga (`ं`, `ँ`, `ः`).
    """
    chars: List[str] = list(token)
    i = 0
    n = len(chars)
    out: List[str] = []

    while i < n:
        ch = chars[i]
        # Case 1: visual-order 'ि' appearing before a consonant cluster
        if ch == "ि" and (not out or out[-1] not in _CONSONANTS):
            if i + 1 < n and chars[i + 1] in _CONSONANTS:
                cluster: List[str] = [chars[i + 1]]
                j = i + 2
                while (
                    j + 1 < n
                    and chars[j] == _HALANT
                    and chars[j + 1] in _CONSONANTS
                ):
                    cluster.append(chars[j])
                    cluster.append(chars[j + 1])
                    j += 2
                out.extend(cluster)
                out.append("ि")
                i = j
                continue

        # Case 2: modifier ('ं', 'ँ', 'ः') emitted before a vowel matra
        if ch in _VOWEL_MATRAS and out and out[-1] in _MODIFIERS:
            mod = out.pop()
            out.append(ch)
            out.append(mod)
            i += 1
            continue

        out.append(ch)
        i += 1

    return "".join(out)


def postprocess_devanagari(text: str) -> str:
    """
    Apply Devanagari linguistic and orthographic normalization to raw CTC text:
      - Convert Preeti number-row glyph tokens (`'द्द।द्द'` -> `'२.२'`, `'द्दद्द'` -> `'२२'`)
      - Fuse Preeti `'र्इ'` (`र + ् + इ`) into `'ई'`
      - Combine split vowel signs (`'अ + ा'` -> `'आ'`, `'ा + े'` -> `'ो'`, `'ा + ै'` -> `'ौ'`)
      - Drop invalid second consecutive vowel matras on the same consonant
      - Reorder visual-order `'ि'` and nasal modifiers into canonical Unicode order
    """
    if not text:
        return text

    # 1. Convert Preeti number-row glyph tokens
    s = _recover_preeti_number(text.strip())

    # 2. Fuse Preeti 'र्इ' ('र' + '्' + 'इ') -> 'ई'
    s = s.replace("र्इ", "ई")

    # 3. Fuse composite vowels and two-part matras
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

    # 4. Collapse consecutive dependent vowel signs (a single consonant cannot take
    # two vowel matras simultaneously) and duplicate modifiers/halants
    dedup_chars: List[str] = []
    for ch in s:
        if dedup_chars:
            prev = dedup_chars[-1]
            if ch in _VOWEL_MATRAS and prev in _VOWEL_MATRAS:
                continue
            if ch == prev and (ch in _MODIFIERS or ch == _HALANT):
                continue
        dedup_chars.append(ch)
    s = "".join(dedup_chars)

    # 5. Reorder visual-order pre-base 'ि' and modifiers
    s = _reorder_visual_ikar_and_modifiers(s)

    return s


linguistic_postprocess = postprocess_devanagari
