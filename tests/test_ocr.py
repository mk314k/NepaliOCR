import os
import tempfile
import unittest
import cv2
import numpy as np

from nepali_ocr import (
    CRNN,
    DynamicArray,
    Num,
    Point,
    Rect,
    RectSet,
    annotate_detected_regions,
    ctc_greedy_decode,
    decode_indices,
    detect_lines,
    detect_words_by_line,
    double_contour_detect,
    encode_text,
    postprocess_devanagari,
    preeti_to_unicode,
    preprocess_crop,
    unicodes,
)
from nepali_ocr.utils import CTC_BLANK_INDEX, char_to_idx


class TestSpatialStructures(unittest.TestCase):
    def test_point_operations_and_bounds(self):
        p1 = Point(10, 20)
        p2 = Point(30, 5)
        self.assertEqual(p1 + p2, Point(40, 25))
        self.assertEqual(p2 - p1, Point(20, -15))
        self.assertEqual(p1.min_xy(p2), Point(10, 5))
        self.assertEqual(p1.max_xy(p2), Point(30, 20))
        self.assertTrue(Point(30, 25).is_upper_right_to(p1))
        with self.assertRaises(IndexError):
            _ = p1[2]

    def test_rect_normalization_iou_and_padded_cropping(self):
        # Inverted coordinates should normalize to positive width and height
        r_inv = Rect(50, 60, 10, 20)
        self.assertEqual(r_inv["lowerLeft"], (10, 20))
        self.assertEqual(r_inv["upperRight"], (50, 60))
        self.assertEqual(r_inv.width(), 40)
        self.assertEqual(r_inv.height(), 40)

        # IoU invariants: identity = 1.0, symmetry, disjoint = 0.0
        r1 = Rect(10, 10, width=20, height=20)
        r2 = Rect(20, 20, width=20, height=20)
        r_disjoint = Rect(100, 100, width=10, height=10)
        self.assertAlmostEqual(r1.iou(r1), 1.0)
        self.assertAlmostEqual(r1.iou(r2), r2.iou(r1))
        self.assertAlmostEqual(r1.iou(r2), 100.0 / 700.0)
        self.assertEqual(r1.iou(r_disjoint), 0.0)

        # Padded overlap bridges small gaps only when padding > 0
        a = Rect(10, 10, width=10, height=10)
        b = Rect(22, 10, width=10, height=10)
        self.assertFalse(a.is_overlapping(b, padding=0))
        self.assertTrue(a.is_overlapping(b, padding=(1, 0)))

        # Padded crop clamps safely at image boundaries
        img = np.zeros((30, 40, 3), dtype=np.uint8)
        edge_rect = Rect(0, 0, width=10, height=10)
        crop = edge_rect.crop_image(img, pad=3)
        self.assertEqual(crop.shape, (13, 13, 3))

    def test_dynamic_array_slot_recycling_and_type_safety(self):
        arr = DynamicArray(Num, init_len=2)
        i0 = arr.append(Num(10))
        i1 = arr.append(Num(20))
        i2 = arr.append(Num(30))
        self.assertEqual((i0, i1, i2), (0, 1, 2))
        self.assertEqual(len(arr), 3)

        # Removing slot 1 and appending again must reuse slot 1 without growing size()
        arr.remove_index(i1)
        self.assertEqual(len(arr), 2)
        self.assertEqual(arr.size(), 3)
        i_reused = arr.append(Num(99))
        self.assertEqual(i_reused, i1)
        self.assertEqual(arr[i1], Num(99))

        with self.assertRaises(TypeError):
            arr.append("invalid_type")

    def test_rectset_transitive_merging_and_reading_order(self):
        # Block mode and non-block mode must produce identical merged rectangles
        rs_grid = RectSet((500, 500), block_size=100, block_mode=True)
        rs_flat = RectSet((500, 500), block_size=100, block_mode=False)

        # A and B are disjoint; C bridges A and B transitively into a single box
        boxes = [
            Rect(20, 40, width=30, height=20, type="text"),
            Rect(60, 40, width=30, height=20, type="text"),
            Rect(45, 42, width=20, height=18, type="text"),
            # Table box overlapping the merged text box must remain unmerged
            Rect(15, 35, width=100, height=40, type="Table"),
        ]
        for b in boxes:
            rs_grid.add_rect(Rect(b.x1, b.y1, b.x2, b.y2, type=b.type))
            rs_flat.add_rect(Rect(b.x1, b.y1, b.x2, b.y2, type=b.type))

        self.assertEqual(len(rs_grid), 2)
        self.assertEqual(rs_grid, rs_flat)

        # Reading order clusters boxes on the same line despite minor vertical jitter
        rs_order = RectSet((400, 400))
        rs_order.add_rect(Rect(180, 106, width=40, height=18))  # Line 1, right
        rs_order.add_rect(Rect(30, 100, width=40, height=22))   # Line 1, left
        rs_order.add_rect(Rect(100, 104, width=40, height=20))  # Line 1, middle
        rs_order.add_rect(Rect(40, 160, width=50, height=20))   # Line 2, left

        ordered = rs_order.reading_order(line_tolerance=14)
        xs_line1 = [r["lowerLeft"][0] for r in ordered[:3]]
        self.assertEqual(xs_line1, [30, 100, 180])
        self.assertEqual(ordered[3]["lowerLeft"], (40, 160))


class TestTextAndLinguisticProcessing(unittest.TestCase):
    def test_preeti_structural_rules(self):
        # Pre-base 'l' ('ि') moves after single consonants and full conjunct clusters
        self.assertEqual(preeti_to_unicode("lstfa"), "किताब")
        self.assertEqual(preeti_to_unicode("l:ylt"), "स्थिति")
        # Post-typed '{' (reph 'र्') moves before the preceding consonant
        self.assertEqual(preeti_to_unicode("ubf{"), "गर्दा")
        # Half-consonant + 'f' ('ा') collapses halant
        self.assertEqual(preeti_to_unicode("Kf"), "प")

    def test_devanagari_postprocess_invariants(self):
        # 1. Visual-order leading 'ि' reorders after consonant or conjunct cluster
        self.assertEqual(postprocess_devanagari("िक"), "कि")
        self.assertEqual(postprocess_devanagari("िप्र"), "प्रि")
        # 2. Nasal modifier before vowel matra reorders to matra + modifier
        self.assertEqual(postprocess_devanagari("हंा"), "हां")
        # 3. Consecutive vowel matras on a single consonant collapse to the first
        self.assertEqual(postprocess_devanagari("कृु"), "कृ")
        # 4. Split composite vowels fuse to canonical codepoints
        self.assertEqual(postprocess_devanagari("अा"), "आ")
        self.assertEqual(postprocess_devanagari("काे"), "को")
        self.assertEqual(postprocess_devanagari("काै"), "कौ")

    def test_vocab_encode_decode_bijection(self):
        all_vocab_str = "".join(unicodes)
        encoded = encode_text(all_vocab_str)
        self.assertEqual(len(encoded), len(unicodes))
        self.assertEqual(decode_indices(encoded), all_vocab_str)


class TestDetectorAndCRNNBehavior(unittest.TestCase):
    def test_projection_profile_and_contour_detector(self):
        page = np.full((360, 400, 3), 255, dtype=np.uint8)
        # Two synthetic text blocks on Line 1, one on Line 2
        cv2.rectangle(page, (30, 40), (110, 62), (15, 15, 15), -1)
        cv2.rectangle(page, (140, 40), (230, 62), (15, 15, 15), -1)
        cv2.rectangle(page, (30, 100), (160, 122), (15, 15, 15), -1)

        lines = detect_lines(page)
        words = detect_words_by_line(page)
        self.assertEqual(len(lines), 2)
        self.assertEqual(len(words), 3)

        rects = double_contour_detect(page, iteration=2)
        self.assertGreaterEqual(len(rects), 3)
        vis = annotate_detected_regions(page, rects)
        self.assertEqual(vis.shape, page.shape)

    def test_preprocess_crop_and_ctc_greedy_collapse_law(self):
        # Standard crop normalizes to (1, 32, 192, 1) with values in [0, 1]
        img = np.random.default_rng(7).integers(0, 255, size=(24, 80, 3), dtype=np.uint8)
        out = preprocess_crop(img)
        self.assertEqual(out.shape, (1, 32, 192, 1))
        self.assertGreaterEqual(float(out.min()), 0.0)
        self.assertLessEqual(float(out.max()), 1.0)

        # Wide crop expands dynamically to a multiple of 4 when dynamic_width=True
        wide_img = np.full((16, 200, 3), 200, dtype=np.uint8)
        wide_out = preprocess_crop(wide_img, dynamic_width=True)
        self.assertEqual(wide_out.shape[1], 32)
        self.assertGreater(wide_out.shape[2], 192)
        self.assertEqual(wide_out.shape[2] % 4, 0)

        # CTC collapse invariant: [k, k, k] -> 'क', whereas [k, blank, k] -> 'कक'
        num_classes = len(unicodes) + 1
        idx_ka = char_to_idx["क"]
        probs_repeat = np.zeros((1, 6, num_classes), dtype=np.float32)
        for t in [0, 1, 2]:
            probs_repeat[0, t, idx_ka] = 1.0
        for t in [3, 4, 5]:
            probs_repeat[0, t, CTC_BLANK_INDEX] = 1.0
        self.assertEqual(ctc_greedy_decode(probs_repeat, apply_postprocess=False), "क")

        probs_sep = np.zeros((1, 6, num_classes), dtype=np.float32)
        for t, c in enumerate([idx_ka, idx_ka, CTC_BLANK_INDEX, idx_ka, idx_ka, CTC_BLANK_INDEX]):
            probs_sep[0, t, c] = 1.0
        self.assertEqual(ctc_greedy_decode(probs_sep, apply_postprocess=False), "कक")

    def test_crnn_forward_shapes_loss_decrease_and_weight_persistence(self):
        rng = np.random.default_rng(42)
        x_batch = rng.random((2, 32, 192, 1), dtype=np.float32)
        targets = ["कख", "गघ"]

        model = CRNN()
        preds = model.predict(x_batch)
        self.assertEqual(preds.shape, (2, 48, len(unicodes) + 1))
        # Framewise outputs must be valid probability distributions summing to 1
        np.testing.assert_allclose(preds.sum(axis=-1), 1.0, atol=1e-5)

        losses = model.train(
            x_batch, targets, epochs=6, batch_size=2, learning_rate=2e-3
        )
        self.assertEqual(len(losses), 6)
        self.assertLess(losses[-1], losses[0])

        preds_trained = model.predict(x_batch)
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = model.save_model(os.path.join(tmpdir, "weights.pt"))
            reloaded = CRNN()
            reloaded.load_weights(ckpt)
            preds_reloaded = reloaded.predict(x_batch)
            np.testing.assert_allclose(preds_trained, preds_reloaded, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
