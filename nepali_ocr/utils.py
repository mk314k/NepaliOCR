"""
Shared constants, image/color helpers, Preeti-to-Unicode conversion, and
character vocabulary mappings for Devanagari/English OCR.
"""
import os
from typing import Dict, List, Sequence, Tuple
import cv2
import numpy as np
from nepali_ocr.spatial.rect import Rect

# Rectangle corner indices
lowerLeft = 0
upperLeft = 1
upperRight = 2
lowerRight = 3

# Layout detection thresholds
REJECTWIDTHFRAC = 0.8
REJECTHEIGHTFRAC = 0.8
DETECTREGION = 190
AREAFACTOR = 0.1
TABLEUNIQUECOLOR = 5
COLORREDUCER = 96
IMGMAXSIZE = 1024

CRNNModelPath = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "crnn_nepali_ocr.pt",
)


def l1_distance(p1p2: Tuple[float, float]) -> float:
    return abs(p1p2[1] - p1p2[0])


def l1_distance_frac(p1p2: Tuple[float, float]) -> float:
    return l1_distance(p1p2) / max(1e-9, abs(p1p2[0]))


L1Distance = l1_distance
L1DistanceFrac = l1_distance_frac


def show_image(name: str, img: np.ndarray) -> None:
    """Display an image resized to `(750, 1000)` when running in an interactive GUI session."""
    if os.environ.get("NEPALI_OCR_HEADLESS", "1") == "1":
        return
    imgr = cv2.resize(img, (750, 1000))
    cv2.imshow(name, imgr)


showImage = show_image


def color_spread(img8: np.ndarray, point_set: np.ndarray) -> float:
    """Compute standard deviation of primary channel intensities along a contour."""
    points = point_set.flatten()
    if len(points) < 2:
        return 0.0
    h, w = img8.shape[:2]
    colors = []
    for i in range(0, len(points), 2):
        px = min(max(0, int(points[i])), w - 1)
        py = min(max(0, int(points[i + 1])), h - 1)
        clr = img8[py, px, 0] if img8.ndim == 3 else img8[py, px]
        colors.append(float(clr))
    return float(np.std(np.array(colors)))


colorSpread = color_spread


def color_count(rect: Rect, imgc: np.ndarray) -> Tuple[int, float]:
    """Return `(unique_color_count, std_dev)` inside `rect` on quantized image `imgc`."""
    crop = rect.crop_image(imgc)
    if crop.size == 0:
        return 0, 0.0
    flat = crop.reshape(-1, crop.shape[-1]) if crop.ndim == 3 else crop.flatten()
    u = np.unique(flat, axis=0)
    return len(u), float(np.std(flat))


colorCount = color_count


def cascading(funcs, img: np.ndarray, repeat: int = 1) -> np.ndarray:
    """Apply a sequence of image transforms `funcs` repeatedly `repeat` times."""
    imgp = np.array(img)
    while repeat > 0:
        for func in funcs:
            imgp = func(imgp)
        repeat -= 1
    return imgp


def bounding_object(contour: np.ndarray) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Compute tight `(min_x, min_y)` and `(max_x, max_y)` bounds of a contour."""
    pts = contour.reshape(-1, 2)
    min_xy = pts.min(axis=0)
    max_xy = pts.max(axis=0)
    return (int(min_xy[0]), int(min_xy[1])), (int(max_xy[0]), int(max_xy[1]))


boundingObject = bounding_object

nepaliKeyBoardPreeti: Dict[str, str] = {
    "a": "ब", "b": "द", "c": "अ", "d": "म", "e": "भ", "f": "ा", "g": "न", "h": "ज", "i": "ष्",
    "j": "व", "k": "प", "l": "ि", "m": "फ", "n": "ल", "o": "य", "p": "उ", "q": "त्र", "r": "च",
    "s": "क", "t": "त", "u": "ग", "v": "ख", "w": "ध", "x": "ह", "y": "थ", "z": "श", "A": "ब्",
    "B": "ध", "C": "ऋ", "D": "म्", "E": "भ्", "F": "ँ", "G": "न्", "H": "ज्", "I": "क्ष्", "J": "व्",
    "K": "प्", "L": "ी", "M": "ः", "N": "ल्", "O": "इ", "P": "ए", "Q": "त्त", "R": "च्", "S": "क्",
    "T": "त्", "U": "ग्", "V": "ख्", "W": "ध्", "X": "ह्", "Y": "थ्", "Z": "श्", "0": "ण्", "1": "ज्ञ",
    "2": "द्द", "3": "घ", "4": "द्ध", "5": "छ", "6": "ट", "7": "ठ", "8": "ड", "9": "ढ",
    "~": "ञ्", "`": "ञ", "!": "१", "@": "२", "#": "३", "$": "४", "%": "५", "^": "६", "&": "७",
    "*": "८", "(": "९", ")": "०", "-": "(", "_": ")", "+": "ं", "[": "ृ", "{": "र्", "]": "े",
    "}": "ै", "\\": "्", "|": "्र", ";": "स", ":": "स्", "'": "ु", '"': "ू", ",": ",", "<": "?",
    ".": "।", ">": "श्र", "/": "र", "?": "रु", "=": ".", "ˆ": "फ्", "Î": "ङ्ख", "å": "द्व", "÷": "/",
    "«": "्र", "»": "्र", "¿": "रू", "°": "ंग", "±": "ं", "³": "ं", "µ": "ं", "¶": "ठ्ठ",
    "·": "ः", "¸": "्र", "º": "ं", "¼": "्र", "½": "्र", "¾": "्र",
}

_PREETI_MULTI_REPLACEMENTS: Tuple[Tuple[str, str], ...] = (
    ("j|m", "क्र"),
    ("k|m", "फ्र"),
    ("km", "फ"),
    ("Qm", "क्त"),
    ("qm", "स्र"),
    ("Iff", "क्षा"),
    ("If", "क्ष"),
    ("1f", "ज्ञा"),
    ("0f", "ण"),
    ("˜", "ऽ"),
)


def _preeti_token_to_uni(token: str) -> str:
    """
    Convert a single Preeti-encoded word token to Unicode Devanagari.

    Handles multi-keystroke ligatures (`j|m -> क्र`, `km -> फ`), half-forms + `f` (`ा`)
    collapsing to full consonants (`Kf -> प`), pre-base `'l'` (`ि`) moving after the
    following consonant cluster, and post-typed `'{'` (*reph* `र्`) moving before the
    preceding consonant cluster.
    """
    s = token
    for src, dst in _PREETI_MULTI_REPLACEMENTS:
        s = s.replace(src, dst)

    out_chars: List[str] = []
    pending_ikar = False
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        mapped = nepaliKeyBoardPreeti.get(ch, ch)

        if ch == "l":
            pending_ikar = True
            i += 1
            continue

        # In Preeti, a half-consonant followed by 'f' ('ा') completes the full consonant
        if mapped.endswith("्") and i + 1 < n and s[i + 1] == "f":
            full_cons = mapped[:-1]
            out_chars.append(full_cons)
            i += 2
            if pending_ikar and not (i < n and s[i] == "\\"):
                out_chars.append("ि")
                pending_ikar = False
            continue

        # Post-typed '{' (reph 'र्') attaches before the preceding consonant cluster
        if ch == "{":
            insert_pos = len(out_chars)
            while insert_pos > 0 and out_chars[insert_pos - 1] in (
                "ा", "ि", "ी", "ु", "ू", "ृ", "े", "ै", "ो", "ौ", "ं", "ँ",
            ):
                insert_pos -= 1
            if insert_pos > 0:
                insert_pos -= 1
                while insert_pos > 0 and out_chars[insert_pos - 1] == "्":
                    insert_pos = max(0, insert_pos - 2)
            out_chars.insert(insert_pos, "र्")
            i += 1
            continue

        out_chars.append(mapped)
        is_half = mapped.endswith("्") or (i + 1 < n and s[i + 1] == "\\")
        if (
            pending_ikar
            and not is_half
            and mapped
            not in ("ा", "ी", "ु", "ू", "ृ", "े", "ै", "ं", "ँ", "ः")
        ):
            out_chars.append("ि")
            pending_ikar = False
        i += 1

    if pending_ikar:
        out_chars.append("ि")

    res = "".join(out_chars)
    res = (
        res.replace("अाे", "ओ")
        .replace("अाै", "औ")
        .replace("अा", "आ")
        .replace("एे", "ऐ")
        .replace("ाे", "ो")
        .replace("ाै", "ौ")
        .replace("्ा", "")
    )
    return res


def preeti_to_unicode(text: str) -> str:
    """Convert Preeti-encoded text to Unicode Devanagari."""
    lines = text.split("\n")
    converted_lines: List[str] = []
    for line in lines:
        tokens = line.split(" ")
        conv_tokens = [_preeti_token_to_uni(tok) if tok else "" for tok in tokens]
        converted_lines.append(" ".join(conv_tokens))
    return "\n".join(converted_lines)


preeti2Uni = preeti_to_unicode

recognizableCharacters: List[str] = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
    "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "(", ")", ".", ",", "-", "?",
    "ा", "ि", "ी", "े", "ै", "ो", "ौ", "ु", "ू", "ृ", "ं", "ँ", "ः", "्", "।",
    "अ", "आ", "उ", "ऊ", "ए", "ऐ", "इ", "ई", "ओ", "औ", "ऋ", "ॐ",
    "क", "ख", "ग", "घ", "ङ", "च", "छ", "ज", "झ", "ञ",
    "ट", "ठ", "ड", "ढ", "ण", "त", "थ", "द", "ध", "न",
    "प", "फ", "ब", "भ", "म", "य", "र", "ल", "व", "श",
    "ष", "स", "ह",
    "०", "१", "२", "३", "४", "५", "६", "७", "८", "९", "॰", " ",
]
unicodes: List[str] = recognizableCharacters

char_to_idx: Dict[str, int] = {ch: idx for idx, ch in enumerate(unicodes)}
idx_to_char: Dict[int, str] = {idx: ch for idx, ch in enumerate(unicodes)}
CTC_BLANK_INDEX: int = len(unicodes)


def encode_text(text: str) -> List[int]:
    """Map a Unicode string to vocabulary indices (skipping unknown characters)."""
    return [char_to_idx[ch] for ch in text if ch in char_to_idx]


def decode_indices(indices: Sequence[int]) -> str:
    """Convert non-blank vocabulary indices back into a Unicode string."""
    return "".join(
        idx_to_char[i]
        for i in indices
        if i in idx_to_char and i != CTC_BLANK_INDEX
    )


nepaliKeyBoard: Dict[str, str] = {
    "a": "ा", "i": "ि", "I": "ी", "e": "े", "E": "ै", "u": "ु", "U": "ू", "R": "ृ",
    "A": "अ", "F": "आ", "W": "इ", "o": "उ", "O": "ए", "L": "ऋ", "P": "ॐ",
    "k": "क", "K": "ख", "g": "ग", "G": "घ", "M": "ङ",
    "c": "च", "C": "छ", "j": "ज", "J": "झ", "Y": "ञ",
    "T": "ट", "Q": "ठ", "D": "ड", "V": "ढ", "N": "ण",
    "t": "त", "q": "थ", "d": "द", "v": "ध", "n": "न",
    "p": "प", "f": "फ", "b": "ब", "B": "भ", "m": "म",
    "y": "य", "r": "र", "l": "ल", "w": "व", "S": "श", "H": "ष", "s": "स", "h": "ह",
    "0": "०", "1": "१", "2": "२", "3": "३", "4": "४", "5": "५", "6": "६", "7": "७", "8": "८", "9": "९",
    ",": "्", ".": "।", "z": "ः", "Z": "्र", "x": "ं", "X": "ँ",
}
