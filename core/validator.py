"""Input validation and lightweight inspection for SatQuery image workflows.

PNG/JPEG files are accepted for prescribed/public benchmark imagery. TIFF is
accepted as a geospatial input, but this module does not infer or fabricate
CRS/GSD metadata; callers can add metadata-aware checks when such data exists.
"""

from io import BytesIO
from pathlib import Path

from PIL import Image, UnidentifiedImageError

SUPPORTED_TIFF_FORMATS = {".tif", ".tiff"}
SUPPORTED_BENCHMARK_FORMATS = {".png", ".jpg", ".jpeg"}
SUPPORTED_FORMATS = SUPPORTED_TIFF_FORMATS | SUPPORTED_BENCHMARK_FORMATS

_CHANNELS_BY_MODE = {
    "1": 1,
    "L": 1,
    "LA": 2,
    "P": 1,
    "RGB": 3,
    "RGBA": 4,
    "CMYK": 4,
    "I": 1,
    "F": 1,
    "I;16": 1,
}


def _filename(image_file):
    return str(getattr(image_file, "name", ""))


def validate_image_file(image_file, name="image"):
    """Validate presence, extension, decodability, and basic raster metadata."""
    errors = []
    warnings = []
    if image_file is None:
        return {
            "valid": False, "name": name, "filename": None, "format": None,
            "format_category": None, "inspection": None,
            "errors": [f"{name}: image was not provided."], "warnings": [],
        }

    filename = _filename(image_file) or Path(str(image_file)).name
    suffix = Path(filename).suffix.lower()
    category = (
        "tiff" if suffix in SUPPORTED_TIFF_FORMATS
        else "benchmark" if suffix in SUPPORTED_BENCHMARK_FORMATS
        else None
    )
    if category is None:
        errors.append(
            f"{name}: unsupported format '{suffix or 'unknown'}'. "
            "Use GeoTIFF/TIFF or PNG/JPG/JPEG for prescribed benchmark data."
        )
    elif category == "benchmark":
        warnings.append(
            f"{name}: {suffix.upper()} is accepted for prescribed/public "
            "benchmark data; geospatial metadata is not available from this format."
        )
    elif category == "tiff":
        warnings.append(
            f"{name}: TIFF can store geospatial metadata, but CRS and ground "
            "sampling distance are not verified by this prototype."
        )

    inspection = None
    if category is not None:
        try:
            inspection = inspect_image(image_file)
            actual_format = (inspection["format"] or "").upper()
            expected_formats = {
                ".tif": {"TIFF"}, ".tiff": {"TIFF"},
                ".jpg": {"JPEG"}, ".jpeg": {"JPEG"},
                ".png": {"PNG"},
            }
            if actual_format not in expected_formats.get(suffix, set()):
                errors.append(
                    f"{name}: file contents do not match the '{suffix}' extension."
                )
            if inspection["width"] < 1 or inspection["height"] < 1:
                errors.append(f"{name}: image dimensions must be greater than zero.")
            if inspection["channels"] is None:
                warnings.append(
                    f"{name}: channel count is unknown for image mode "
                    f"'{inspection['mode']}'."
                )
        except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
            errors.append(f"{name}: image could not be decoded ({exc}).")

    return {
        "valid": not errors,
        "name": name,
        "filename": filename,
        "format": suffix.upper() if suffix else None,
        "format_category": category,
        "inspection": inspection,
        "errors": errors,
        "warnings": warnings,
    }


def _read_image_source(image_source):
    """Accept filesystem paths, bytes, or Streamlit UploadedFile-like objects."""
    if isinstance(image_source, (str, Path)):
        return Image.open(image_source)
    if isinstance(image_source, (bytes, bytearray, memoryview)):
        return Image.open(BytesIO(bytes(image_source)))
    if hasattr(image_source, "getvalue"):
        return Image.open(BytesIO(image_source.getvalue()))
    if hasattr(image_source, "read"):
        position = image_source.tell() if hasattr(image_source, "tell") else None
        try:
            return Image.open(image_source)
        finally:
            if position is not None and hasattr(image_source, "seek"):
                image_source.seek(position)
    raise TypeError("Expected a file path, bytes, or uploaded image file.")


def inspect_image(image_source):
    """Return actual image dimensions, mode, channel count, and format."""
    with _read_image_source(image_source) as image:
        mode = image.mode
        return {
            "filename": Path(_filename(image_source)).name or None,
            "width": int(image.width),
            "height": int(image.height),
            "channels": _CHANNELS_BY_MODE.get(mode),
            "mode": mode,
            "format": image.format,
        }


def check_pair_compatibility(earlier_info, later_info):
    """Check the requirements of the current OpenCV change-analysis path.

    The detector decodes both inputs to three-channel color, so source mode and
    channel differences are reported as warnings. Equal spatial dimensions are
    required because the current pixelwise comparison is not a registration or
    resampling algorithm.
    """
    errors = []
    warnings = []
    dimensions_match = (
        earlier_info["width"] == later_info["width"]
        and earlier_info["height"] == later_info["height"]
    )
    if not dimensions_match:
        errors.append(
            "Earlier and later image dimensions do not match "
            f"({earlier_info['width']}×{earlier_info['height']} vs "
            f"{later_info['width']}×{later_info['height']})."
        )
    channels_match = earlier_info["channels"] == later_info["channels"]
    mode_match = earlier_info["mode"] == later_info["mode"]
    if not channels_match or not mode_match:
        warnings.append(
            "The source image modes/channels differ. OpenCV converts both to "
            "three-channel color for this baseline; verify that this is suitable."
        )
    return {
        "compatible": not errors,
        "errors": errors,
        "warnings": warnings,
        "dimensions_match": dimensions_match,
        "channels_match": channels_match,
        "mode_match": mode_match,
    }
