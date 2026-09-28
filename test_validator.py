"""Small self-contained checks for the SatQuery image validator."""

from io import BytesIO
import unittest

from PIL import Image

from core.validator import (
    check_pair_compatibility,
    inspect_image,
    validate_image_file,
)


class UploadedImage:
    def __init__(self, name, payload):
        self.name = name
        self._payload = payload

    def getvalue(self):
        return self._payload


def png_bytes(size=(8, 6), color=(30, 80, 120)):
    buffer = BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


class ValidatorTests(unittest.TestCase):
    def test_inspects_uploaded_benchmark_image(self):
        file = UploadedImage("scene.png", png_bytes())
        result = validate_image_file(file, "scene")
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(result["format_category"], "benchmark")
        self.assertEqual(result["inspection"]["width"], 8)
        self.assertEqual(result["inspection"]["height"], 6)
        self.assertEqual(result["inspection"]["channels"], 3)
        self.assertEqual(result["inspection"]["mode"], "RGB")
        self.assertTrue(result["warnings"])
        self.assertIn("prescribed/public benchmark data", result["warnings"][0])

    def test_tiff_does_not_claim_crs_or_ground_sampling_distance(self):
        buffer = BytesIO()
        Image.new("RGB", (8, 6)).save(buffer, format="TIFF")
        result = validate_image_file(UploadedImage("scene.tif", buffer.getvalue()))
        self.assertTrue(result["valid"], result["errors"])
        self.assertTrue(any("CRS" in warning and "ground" in warning for warning in result["warnings"]))

    def test_rejects_missing_or_unsupported_input(self):
        self.assertFalse(validate_image_file(None)["valid"])
        bad = UploadedImage("scene.gif", b"not an image")
        result = validate_image_file(bad)
        self.assertFalse(result["valid"])
        self.assertTrue(result["errors"])

    def test_rejects_extension_content_mismatch(self):
        file = UploadedImage("scene.jpg", png_bytes())
        result = validate_image_file(file)
        self.assertFalse(result["valid"])
        self.assertTrue(any("do not match" in e for e in result["errors"]))

    def test_pair_dimensions_must_match(self):
        earlier = inspect_image(png_bytes())
        later = inspect_image(png_bytes((9, 6)))
        result = check_pair_compatibility(earlier, later)
        self.assertFalse(result["compatible"])
        self.assertFalse(result["dimensions_match"])

    def test_same_size_pair_is_compatible(self):
        earlier = inspect_image(png_bytes())
        later = inspect_image(png_bytes(color=(90, 90, 90)))
        result = check_pair_compatibility(earlier, later)
        self.assertTrue(result["compatible"])
        self.assertTrue(result["dimensions_match"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
