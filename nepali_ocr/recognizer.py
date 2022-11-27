"""
Image preprocessing, CTC greedy decoding, and top-level word recognition API.
"""
import os
from typing import List, Optional, Tuple
import cv2
import numpy as np

from nepali_ocr.model import CRNN
from nepali_ocr.postprocess import postprocess_devanagari
from nepali_ocr.utils import CTC_BLANK_INDEX, CRNNModelPath, idx_to_char

_CACHED_MODEL: Optional[CRNN] = None
_CACHED_MODEL_PATH: Optional[str] = None


def preprocess_crop(
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


preProcess = preprocess_crop


def ctc_greedy_decode(pred: np.ndarray, apply_postprocess: bool = True) -> str:
    """
    Greedy CTC decode: collapse repeated adjacent indices, drop blank tokens,
    and apply Devanagari linguistic normalization when `apply_postprocess=True`.
    """
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
    raw = "".join(decoded_chars)
    return postprocess_devanagari(raw) if apply_postprocess else raw


postProcess = ctc_greedy_decode


def get_recognizer_model(model_path: str = CRNNModelPath) -> CRNN:
    """Return a cached `CRNN` instance loaded with `model_path`."""
    global _CACHED_MODEL, _CACHED_MODEL_PATH
    if _CACHED_MODEL is None or _CACHED_MODEL_PATH != model_path:
        model = CRNN()
        if model_path and os.path.isfile(model_path):
            model.load_weights(model_path)
        _CACHED_MODEL = model
        _CACHED_MODEL_PATH = model_path
    return _CACHED_MODEL


getRecognizerModel = get_recognizer_model


def recognize(img: np.ndarray, model_path: str = CRNNModelPath, modelPath: Optional[str] = None) -> str:
    """Recognize the Devanagari/English text in a cropped word or line image."""
    if modelPath is not None:
        model_path = modelPath
    imgp = preprocess_crop(img)
    recog_model = get_recognizer_model(model_path)
    pred = recog_model.predict(imgp)
    return ctc_greedy_decode(pred)
