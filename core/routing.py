"""Keyword-based routing for the currently exposed SatQuery workflows."""

import re

WORKFLOW_CHANGE = "Bi-temporal Change Analysis"
WORKFLOW_VQA = "Single-Image VQA / Description"
WORKFLOW_GROUNDING = "Grounding"
WORKFLOW_SAR = "Optical + SAR Analysis"
WORKFLOW_UNSUPPORTED = "Unsupported / Needs Clarification"

TOOL_BY_WORKFLOW = {
    WORKFLOW_CHANGE: "Change Detection Specialist",
    WORKFLOW_VQA: "VQA / Scene Description Specialist",
    WORKFLOW_GROUNDING: "Visual Grounding Specialist",
    WORKFLOW_SAR: "Optical-SAR Fusion Specialist",
    WORKFLOW_UNSUPPORTED: "Clarification",
}

CHANGE_TERMS = [
    "change", "changed", "difference", "before", "after", "between dates",
    "increased", "decreased",
]
GROUNDING_TERMS = [
    "highlight", "where", "locate", "location", "region", "mark",
    "identify where",
]
SAR_TERMS = ["SAR", "optical and SAR", "radar", "multimodal"]
VQA_TERMS = ["describe", "what is", "what do you see", "land cover", "objects", "scene"]


def _term_in_query(query_text, term):
    lowered = query_text.lower()
    needle = term.lower()
    if " " in needle:
        return needle in lowered
    return re.search(rf"\b{re.escape(needle)}\b", lowered) is not None


def _matched_terms(query_text, terms):
    return [term for term in terms if _term_in_query(query_text, term)]


def _image_presence(earlier_image, later_image):
    has_earlier = earlier_image is not None
    has_later = later_image is not None
    if has_earlier and has_later:
        summary = "Both earlier and later images provided"
    elif has_earlier:
        summary = "Only the earlier image is provided"
    elif has_later:
        summary = "Only the later image is provided"
    else:
        summary = "No images provided"
    return has_earlier, has_later, summary


def _result(workflow, reason, input_status):
    return {
        "workflow": workflow,
        "reason": reason,
        "input_status": input_status,
        "tool_name": TOOL_BY_WORKFLOW[workflow],
    }


def route_query(query, earlier_image, later_image):
    """Route a natural-language query using the app's existing keyword rules."""
    query_text = (query or "").strip()
    has_earlier, has_later, presence = _image_presence(earlier_image, later_image)
    both_images = has_earlier and has_later
    has_any_image = has_earlier or has_later

    if not query_text:
        return _result(
            WORKFLOW_UNSUPPORTED,
            "No query was provided. Enter a natural-language question so SatQuery can select a workflow.",
            presence,
        )

    sar_hits = _matched_terms(query_text, SAR_TERMS)
    change_hits = _matched_terms(query_text, CHANGE_TERMS)
    if _term_in_query(query_text, "between") and (
        _term_in_query(query_text, "date") or _term_in_query(query_text, "dates")
    ) and "between dates" not in change_hits:
        change_hits.append("between dates")
    grounding_hits = _matched_terms(query_text, GROUNDING_TERMS)
    vqa_hits = _matched_terms(query_text, VQA_TERMS)

    # Preserve existing precedence: multimodal, change, grounding, description.
    if sar_hits:
        intended, hits = WORKFLOW_SAR, sar_hits
    elif change_hits:
        intended, hits = WORKFLOW_CHANGE, change_hits
    elif grounding_hits:
        intended, hits = WORKFLOW_GROUNDING, grounding_hits
    elif vqa_hits:
        intended, hits = WORKFLOW_VQA, vqa_hits
    else:
        return _result(
            WORKFLOW_UNSUPPORTED,
            "The query did not match a supported SatQuery workflow. "
            "Try asking about change between dates, scene description, "
            "where something is located, or optical + SAR analysis.",
            presence,
        )

    hit_text = ", ".join(f'"{item}"' for item in hits)

    if intended == WORKFLOW_SAR:
        if not both_images:
            return _result(
                WORKFLOW_UNSUPPORTED,
                f"Optical + SAR analysis was indicated by {hit_text}, but this workflow requires two images. "
                "Upload both scenes. Actual optical vs SAR modality validation will be added later.",
                f"{presence}. Optical + SAR requires two images; file-type/modality checks are not implemented yet.",
            )
        return _result(
            WORKFLOW_SAR,
            f"Query terms {hit_text} indicate an optical + SAR / multimodal request, and two images are present. "
            "Actual optical vs SAR modality validation will be added later; this MVP only checks that two files were uploaded. "
            "No image analysis has been performed.",
            f"{presence}. Ready for routing. Optical vs SAR modality validation will be added later.",
        )

    if intended == WORKFLOW_CHANGE:
        if not both_images:
            missing = "both earlier and later images" if not has_any_image else (
                "the later image" if has_earlier else "the earlier image"
            )
            return _result(
                WORKFLOW_UNSUPPORTED,
                f"Bi-temporal change analysis was indicated by {hit_text}, but {missing} must be uploaded.",
                f"{presence}. Bi-temporal Change Analysis requires both images.",
            )
        return _result(
            WORKFLOW_CHANGE,
            f"Query terms {hit_text} indicate a before/after comparison, and both images are available. "
            "Change detection has not been run yet.",
            f"{presence}. Inputs are sufficient for change analysis; processing is not implemented yet.",
        )

    if not has_any_image:
        return _result(
            WORKFLOW_UNSUPPORTED,
            f"{intended} was indicated by {hit_text}, but at least one image is required.",
            f"{presence}. Upload at least one image for this workflow.",
        )

    if intended == WORKFLOW_GROUNDING:
        return _result(
            WORKFLOW_GROUNDING,
            f"Query terms {hit_text} indicate a locating / highlighting request. "
            "Grounding has not been run yet.",
            f"{presence}. Inputs are sufficient for grounding; processing is not implemented yet.",
        )

    return _result(
        WORKFLOW_VQA,
        f"Query terms {hit_text} indicate a scene description / VQA request. "
        "No description has been generated yet.",
        f"{presence}. Inputs are sufficient for single-image VQA; processing is not implemented yet.",
    )
