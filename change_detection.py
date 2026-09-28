"""
SatQuery AI — classical bi-temporal change-detection specialist.

This module is a transparent OpenCV/NumPy baseline, not an AI/ML model.
The public `detect_change()` entry point is the specialist contract so a
trained remote-sensing model can replace the internals later without
changing the rest of the application architecture.
"""

import tempfile
from pathlib import Path

import cv2
import numpy as np

# Paths are resolved from this file so the specialist does not depend on cwd.
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "oscd"
EARLIER_PATH = DATA_DIR / "earlier.png"
LATER_PATH = DATA_DIR / "later.png"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "change_detection"
MASK_PATH = OUTPUT_DIR / "detected_change_mask.png"
EVIDENCE_PATH = OUTPUT_DIR / "change_evidence.png"

# Light morphology: large enough to drop speckle, small enough to keep
# compact urban-change blobs typical of OSCD patches.
MORPH_KERNEL_SIZE = 3


def load_pair(earlier_path=EARLIER_PATH, later_path=LATER_PATH):
    """Stage 1 — load the earlier/later RGB pair as BGR uint8 arrays."""
    earlier = cv2.imread(str(earlier_path), cv2.IMREAD_COLOR)
    later = cv2.imread(str(later_path), cv2.IMREAD_COLOR)
    if earlier is None:
        raise FileNotFoundError(f"Could not read earlier image: {earlier_path}")
    if later is None:
        raise FileNotFoundError(f"Could not read later image: {later_path}")
    return earlier, later


def verify_same_dimensions(earlier, later):
    """Stage 2 — bi-temporal differencing requires co-registered, equal-size rasters."""
    if earlier.shape != later.shape:
        raise ValueError(
            "Earlier and later images must have the same dimensions. "
            f"Got {earlier.shape} and {later.shape}."
        )
    height, width = earlier.shape[:2]
    return height, width


def to_float_channels(image):
    """Stage 3 — convert 8-bit BGR to float32 in [0, 1] for stable subtraction."""
    return image.astype(np.float32) / 255.0


def pixelwise_difference(earlier_f, later_f):
    """Stage 4 — per-pixel spectral distance across the three color channels."""
    delta = later_f - earlier_f
    return np.sqrt(np.sum(delta * delta, axis=2))


def normalize_difference(diff):
    """Stage 5 — min-max stretch the distance map to 8-bit for thresholding."""
    minimum = float(diff.min())
    maximum = float(diff.max())
    if maximum <= minimum:
        return np.zeros(diff.shape, dtype=np.uint8)
    scaled = (diff - minimum) / (maximum - minimum)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


def otsu_change_mask(diff_u8):
    """
    Stage 6–7 — Otsu threshold + binary mask.

    Otsu chooses the cut that maximises between-class variance of the
    difference histogram (unchanged vs changed pixels). That is a standard
    unsupervised baseline, not a learned model.
    """
    threshold, binary = cv2.threshold(
        diff_u8,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    return float(threshold), binary


def clean_mask(binary_mask, kernel_size=MORPH_KERNEL_SIZE):
    """
    Stage 8 — opening removes isolated speckle; closing fills small gaps
    inside coherent change regions.
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    return closed


def change_statistics(mask):
    """Stage 9 — pixel counts and share of pixels classified as changed."""
    changed_pixels = int(np.count_nonzero(mask))
    total_pixels = int(mask.size)
    percent = (100.0 * changed_pixels / total_pixels) if total_pixels else 0.0
    return {
        "changed_pixel_count": changed_pixels,
        "total_pixel_count": total_pixels,
        # Retained for result-key compatibility; this is not geographic area.
        "changed_area_percent": percent,
    }


def build_evidence(later_bgr, mask):
    """
    Stage 10 — overlay detected change on the later image.

    Changed pixels are tinted red so the evidence is readable without
    hiding the underlying satellite scene.
    """
    highlight = later_bgr.copy()
    highlight[mask > 0] = (0, 0, 255)
    return cv2.addWeighted(later_bgr, 0.62, highlight, 0.38, 0.0)


def save_outputs(mask, evidence, mask_path=MASK_PATH, evidence_path=EVIDENCE_PATH):
    """Stage 11 — write the binary mask and visual evidence PNGs."""
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(mask_path), mask):
        raise IOError(f"Failed to write change mask: {mask_path}")
    if not cv2.imwrite(str(evidence_path), evidence):
        raise IOError(f"Failed to write change evidence: {evidence_path}")
    return mask_path, evidence_path


def detect_change(
    earlier_path=EARLIER_PATH,
    later_path=LATER_PATH,
    mask_path=MASK_PATH,
    evidence_path=EVIDENCE_PATH,
):
    """
    Run the baseline specialist and return a structured result.

    Later model swaps should keep this function name and result keys.
    """
    earlier, later = load_pair(earlier_path, later_path)
    height, width = verify_same_dimensions(earlier, later)

    earlier_f = to_float_channels(earlier)
    later_f = to_float_channels(later)
    diff = pixelwise_difference(earlier_f, later_f)
    diff_u8 = normalize_difference(diff)
    threshold, raw_mask = otsu_change_mask(diff_u8)
    mask = clean_mask(raw_mask)
    stats = change_statistics(mask)
    evidence = build_evidence(later, mask)
    saved_mask, saved_evidence = save_outputs(mask, evidence, mask_path, evidence_path)

    return {
        "method": "classical_cv_baseline",
        "algorithm": "RGB L2 difference + Otsu + morphology",
        "image_size": (width, height),
        "otsu_threshold": threshold,
        "changed_pixel_count": stats["changed_pixel_count"],
        "total_pixel_count": stats["total_pixel_count"],
        "changed_area_percent": stats["changed_area_percent"],
        "mask_path": saved_mask,
        "evidence_path": saved_evidence,
    }


def print_statistics(result):
    """Stage 12 — console summary for MVP inspection."""
    width, height = result["image_size"]
    print("SatQuery change-detection specialist (classical CV baseline)")
    print(f"  Method:              {result['algorithm']}")
    print(f"  Image size:          {width} x {height}")
    print(f"  Otsu threshold:      {result['otsu_threshold']:.1f} (on 0-255 difference map)")
    print(f"  Changed pixels:      {result['changed_pixel_count']}")
    print(f"  Total pixels:        {result['total_pixel_count']}")
    print(f"  Changed-pixel %:     {result['changed_area_percent']:.2f} (not geographic area)")
    print(f"  Mask saved to:       {result['mask_path']}")
    print(f"  Evidence saved to:   {result['evidence_path']}")
    print("  Note: this is not an AI/ML model; it is an unsupervised OpenCV baseline.")


def main():
    result = detect_change()
    print_statistics(result)
    return result


if __name__ == "__main__":
    main()
