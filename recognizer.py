"""
Image preprocessing, CTC greedy decoding, and top-level word recognition API.
"""
import os
from typing import List, Optional, Tuple
import cv2
import numpy as np

from learn import CRNN
from utility import CTC_BLANK_INDEX, CRNNModelPath, idx_to_char

_CACHED_MODEL: Optional[CRNN] = None
_CACHED_MODEL_PATH: Optional[str] = None


def preProcess(
    img: np.ndarray,
    target_shape: Tuple[int, int] = (32, 192),
    dynamic_width: bool = True,
) -> np.ndarray:
    """
    Resize a cropped word or line image to height `32` while preserving aspect ratio,
    and normalize foreground ink intensity to `[0.0, 1.0]`.

    When `dynamic_width=True` and the aspect-ratio-scaled width exceeds `target_w`,
    the output canvas expands to the next multiple of 4 so wide multi-word crops are
    not horizontally squashed.
    """
    target_h, target_w = target_shape
    if img is None or img.size == 0:
        return np.zeros((1, target_h, target_w, 1), dtype=np.float32)

    gray = (
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img.ndim == 3
        else img.copy()
    )
    h, w = gray.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((1, target_h, target_w, 1), dtype=np.float32)

    scale = target_h / float(h)
    raw_new_w = max(1, int(round(w * scale)))
    if dynamic_width and raw_new_w > target_w:
        new_w = ((raw_new_w + 3) // 4) * 4
        out_w = new_w
    else:
        new_w = min(target_w, raw_new_w)
        out_w = target_w

    interp = cv2.INTER_CUBIC if scale >= 1.0 else cv2.INTER_AREA
    resized = cv2.resize(gray, (new_w, target_h), interpolation=interp).astype(
        np.float32
    )

    # Invert so ink is positive (1.0) and paper background is 0.0, using percentile
    # stretching rather than hard thresholding to keep thin anti-aliased matras intact.
    inv = 255.0 - resized
    lo = float(np.percentile(inv, 10))
    hi = float(np.percentile(inv, 99))
    if hi - lo > 10.0:
        normed = np.clip((inv - lo) / (hi - lo), 0.0, 1.0)
    else:
        normed = inv / 255.0

    canvas = np.zeros((target_h, out_w), dtype=np.float32)
    canvas[:, :new_w] = normed
    return canvas[None, :, :, None]


def postProcess(pred: np.ndarray) -> str:
    """Greedy CTC decode: collapse repeated adjacent indices and drop blank tokens."""
    probs = np.asarray(pred)
    if probs.ndim == 3:
        probs = probs[0]
    best_path = np.argmax(probs, axis=-1).tolist()

    decoded_chars: List[str] = []
    prev_idx = -1
    for idx in best_path:
        if idx != prev_idx and idx != CTC_BLANK_INDEX and idx in idx_to_char:
            decoded_chars.append(idx_to_char[idx])
        prev_idx = idx
    return "".join(decoded_chars)


def getRecognizerModel(modelPath: str = CRNNModelPath) -> CRNN:
    """Return a cached `CRNN` instance loaded with `modelPath`."""
    global _CACHED_MODEL, _CACHED_MODEL_PATH
    if _CACHED_MODEL is None or _CACHED_MODEL_PATH != modelPath:
        model = CRNN()
        if modelPath and os.path.isfile(modelPath):
            model.loadWeights(modelPath)
        _CACHED_MODEL = model
        _CACHED_MODEL_PATH = modelPath
    return _CACHED_MODEL


def recognize(img: np.ndarray, modelPath: str = CRNNModelPath) -> str:
    """Recognize the Devanagari/English text in a cropped word or line image."""
    imgp = preProcess(img)
    recogModel = getRecognizerModel(modelPath)
    pred = recogModel.predict(imgp)
    return postProcess(pred)
