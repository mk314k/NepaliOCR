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
    double_contour_detect,
    encode_text,
    preeti_to_unicode,
    preprocess_crop,
)


class TestNepaliOCR(unittest.TestCase):
    def test_point_and_rect_geometry(self):
        p1 = Point(10, 20)
        p2 = Point(30, 50)
        self.assertEqual(p1 + p2, Point(40, 70))
        self.assertEqual(p1.min_xy(p2), Point(10, 20))
        self.assertEqual(p1.max_xy(p2), Point(30, 50))

        r1 = Rect(10, 20, width=40, height=30, type="text")
        r2 = Rect(35, 30, width=30, height=25, type="text")
        self.assertEqual(r1.width(), 40)
        self.assertEqual(r1.height(), 30)
        self.assertEqual(r1.area(), 1200)
        self.assertTrue(r1.is_overlapping(r2))
        self.assertGreater(r1.iou(r2), 0.0)

        self.assertTrue(r1.overlap_update(r2))
        self.assertEqual(r1["lowerLeft"], (10, 20))
        self.assertEqual(r1["upperRight"], (65, 55))

    def test_dynamic_array_and_rectset_spatial_merging(self):
        dyn = DynamicArray(Num, init_len=2)
        dyn.append(Num(100))
        i1 = dyn.append(Num(200))
        dyn.append(Num(300))
        self.assertEqual(len(dyn), 3)
        dyn.remove_index(i1)
        self.assertEqual(len(dyn), 2)
        i3 = dyn.append(Num(400))
        self.assertEqual(i3, i1)

        rset = RectSet((800, 600), block_size=100, block_mode=True)
        rset.add_rect(Rect(50, 50, width=60, height=25, type="text"))
        rset.add_rect(Rect(95, 52, width=70, height=25, type="text"))
        rset.add_rect(Rect(50, 200, width=300, height=150, type="Table"))
        rset.add_rect(Rect(60, 210, width=80, height=30, type="Table"))

        ordered = rset.reading_order()
        self.assertEqual(len(ordered), 3)
        self.assertEqual(ordered[0].type, "text")
        self.assertEqual(ordered[0]["lowerLeft"], (50, 50))
        self.assertEqual(ordered[0]["upperRight"], (165, 77))

    def test_preeti_to_unicode_and_vocab_encoding(self):
        self.assertEqual(preeti_to_unicode("g]kfn"), "नेपाल")
        self.assertEqual(preeti_to_unicode("lj1fg"), "विज्ञान")
        self.assertEqual(preeti_to_unicode("Kf"), "प")

        sample = "विज्ञान Science १२३"
        enc = encode_text(sample)
        dec = decode_indices(enc)
        self.assertEqual(dec, sample)

    def test_layout_detector_on_synthetic_and_sample_page(self):
        page = np.full((600, 500, 3), 255, dtype=np.uint8)
        cv2.rectangle(page, (40, 40), (140, 62), (10, 10, 10), -1)
        cv2.rectangle(page, (160, 40), (280, 62), (10, 10, 10), -1)
        rng = np.random.default_rng(0)
        page[120:240, 60:220] = rng.integers(
            0, 200, size=(120, 160, 3), dtype=np.uint8
        )
        cv2.rectangle(page, (40, 320), (440, 500), (0, 0, 0), 3)

        rects = double_contour_detect(page, padding=2, iteration=2)
        self.assertGreater(len(rects), 0)
        annotated = annotate_detected_regions(page, rects)
        self.assertEqual(annotated.shape, page.shape)

        if os.path.isfile("assets/gr9sctrain23_detected.jpg"):
            real_img = cv2.imread("assets/gr9sctrain23_detected.jpg")
            self.assertIsNotNone(real_img)
            real_rects = double_contour_detect(real_img, iteration=2)
            self.assertGreater(len(real_rects), 10)

    def test_crnn_ctc_training_and_checkpoint_roundtrip(self):
        words = ["Science", "Taxon"]
        crops = []
        for w in words:
            canvas = np.full((40, 160, 3), 255, dtype=np.uint8)
            cv2.putText(
                canvas, w, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (10, 10, 10), 2
            )
            crops.append(preprocess_crop(canvas)[0])
        x_batch = np.stack(crops, axis=0)

        model = CRNN()
        losses = model.train(
            x_batch, words, epochs=6, batch_size=2, learning_rate=2e-3
        )
        self.assertEqual(len(losses), 6)
        preds = model.predict(x_batch)
        self.assertEqual(preds.shape[0], 2)

        with tempfile.TemporaryDirectory() as tmpdir:
            w_path = os.path.join(tmpdir, "test_crnn.pt")
            saved = model.save_model(w_path)
            self.assertTrue(os.path.isfile(saved))

            reloaded = CRNN()
            reloaded.load_weights(saved)
            pred_reload = reloaded.predict(x_batch[:1])
            self.assertEqual(
                ctc_greedy_decode(pred_reload), ctc_greedy_decode(preds[:1])
            )


if __name__ == "__main__":
    unittest.main()
