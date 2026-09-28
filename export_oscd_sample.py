from pathlib import Path

import numpy as np
from datasets import load_dataset
from PIL import Image

OUTPUT_DIR = Path("data/oscd")
EARLIER_PATH = OUTPUT_DIR / "earlier.png"
LATER_PATH = OUTPUT_DIR / "later.png"
MASK_PATH = OUTPUT_DIR / "ground_truth.png"


def _to_pil(image):
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, dict) and image.get("path"):
        return Image.open(image["path"])
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = np.transpose(array, (1, 2, 0))
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[:, :, 0]
    return Image.fromarray(array)


def _to_uint8(array):
    array = np.asarray(array)
    if array.dtype == np.uint8:
        return array
    values = array.astype(np.float32)
    max_value = float(np.nanmax(values)) if values.size else 0.0
    if max_value <= 1.0:
        values = values * 255.0
    elif max_value > 255.0:
        low, high = np.percentile(values, (2, 98))
        if high > low:
            values = (values - low) / (high - low)
        else:
            values = values / max(max_value, 1.0)
        values = np.clip(values, 0.0, 1.0) * 255.0
    return np.clip(values, 0, 255).astype(np.uint8)


def to_rgb_png_image(image):
    pil = _to_pil(image)
    array = np.asarray(pil)
    array = _to_uint8(array)
    if array.ndim == 2:
        array = np.stack([array, array, array], axis=-1)
    elif array.ndim == 3 and array.shape[-1] == 4:
        array = array[:, :, :3]
    elif array.ndim == 3 and array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    return Image.fromarray(array, mode="RGB")


def to_grayscale_mask(image):
    pil = _to_pil(image)
    array = np.asarray(pil)
    if array.ndim == 3:
        array = array[:, :, 0]
    array = _to_uint8(array)
    unique_values = np.unique(array)
    if set(unique_values.tolist()).issubset({0, 1}):
        array = (array * 255).astype(np.uint8)
    return Image.fromarray(array, mode="L")


def main():
    ds = load_dataset("blanchon/OSCD_RGB")
    sample = ds["test"][0]

    earlier = to_rgb_png_image(sample["image1"])
    later = to_rgb_png_image(sample["image2"])
    mask = to_grayscale_mask(sample["mask"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    earlier.save(EARLIER_PATH)
    later.save(LATER_PATH)
    mask.save(MASK_PATH)

    print(f"earlier.png  size={earlier.size}  mode={earlier.mode}  path={EARLIER_PATH.resolve()}")
    print(f"later.png    size={later.size}  mode={later.mode}  path={LATER_PATH.resolve()}")
    print(f"ground_truth.png  size={mask.size}  mode={mask.mode}  path={MASK_PATH.resolve()}")


if __name__ == "__main__":
    main()
