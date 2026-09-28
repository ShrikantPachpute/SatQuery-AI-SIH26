"""Integration check using the project's existing sample pair read-only."""

import tempfile
import unittest
from pathlib import Path

from change_detection import detect_change


PROJECT_ROOT = Path(__file__).resolve().parent


class ChangeDetectionTests(unittest.TestCase):
    def test_oscd_sample_pair_generates_outputs_outside_project_data(self):
        earlier = PROJECT_ROOT / "data" / "oscd" / "earlier.png"
        later = PROJECT_ROOT / "data" / "oscd" / "later.png"

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            result = detect_change(
                earlier_path=earlier,
                later_path=later,
                mask_path=output / "mask.png",
                evidence_path=output / "evidence.png",
            )

            self.assertEqual(result["method"], "classical_cv_baseline")
            self.assertEqual(result["algorithm"], "RGB L2 difference + Otsu + morphology")
            self.assertGreater(result["total_pixel_count"], 0)
            self.assertGreaterEqual(result["changed_pixel_count"], 0)
            self.assertLessEqual(result["changed_pixel_count"], result["total_pixel_count"])
            self.assertGreaterEqual(result["changed_area_percent"], 0.0)
            self.assertLessEqual(result["changed_area_percent"], 100.0)
            self.assertTrue((output / "mask.png").is_file())
            self.assertTrue((output / "evidence.png").is_file())
            self.assertEqual(result["mask_path"], output / "mask.png")
            self.assertEqual(result["evidence_path"], output / "evidence.png")


if __name__ == "__main__":
    unittest.main(verbosity=2)
