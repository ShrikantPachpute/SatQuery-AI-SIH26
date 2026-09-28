import streamlit as st
import tempfile
import time
from pathlib import Path

from change_detection import detect_change
from core.routing import WORKFLOW_CHANGE, route_query
from core.validator import check_pair_compatibility, validate_image_file

def run_change_analysis(earlier_file, later_file):
    """Run the existing change-detection specialist on uploaded images."""

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        earlier_path = temp_dir / "earlier.png"
        later_path = temp_dir / "later.png"
        mask_path = temp_dir / "detected_change_mask.png"
        evidence_path = temp_dir / "change_evidence.png"

        earlier_path.write_bytes(earlier_file.getvalue())
        later_path.write_bytes(later_file.getvalue())

        start_time = time.perf_counter()

        result = detect_change(
            earlier_path=earlier_path,
            later_path=later_path,
            mask_path=mask_path,
            evidence_path=evidence_path,
        )

        runtime = time.perf_counter() - start_time

        result["runtime_seconds"] = runtime
        result["mask_bytes"] = mask_path.read_bytes()
        result["evidence_bytes"] = evidence_path.read_bytes()

        return result


st.set_page_config(
    page_title="SatQuery AI",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.6rem;
            padding-bottom: 2.4rem;
            max-width: 1180px;
        }
        h1 {
            letter-spacing: -0.03em;
            margin-bottom: 0.15rem !important;
        }
        .satquery-subtitle {
            color: #334155;
            font-size: 1.15rem;
            font-weight: 600;
            margin-bottom: 0.45rem;
        }
        .satquery-desc {
            color: #475569;
            font-size: 0.98rem;
            margin-bottom: 1.2rem;
        }
        .satquery-card {
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 1rem 1.1rem;
            background: #ffffff;
        }
        .satquery-status {
            font-size: 0.92rem;
            line-height: 1.7;
            color: #334155;
        }
        .stButton > button {
            font-weight: 600;
            border-radius: 8px;
            height: 2.6rem;
        }
        [data-testid="stSidebar"] {
            background: #f8fafc;
            border-right: 1px solid #e2e8f0;
        }
        div[data-testid="stFileUploader"] section {
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

if "route_result" not in st.session_state:
    st.session_state.route_result = None
if "routed_query" not in st.session_state:
    st.session_state.routed_query = ""
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "validation_results" not in st.session_state:
    st.session_state.validation_results = None

IMAGE_TYPES = ["png", "jpg", "jpeg", "tiff", "tif"]

st.title("SatQuery AI")
st.markdown(
    '<div class="satquery-subtitle">Interactive Remote-Sensing Intelligence through Natural Language</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="satquery-desc">Ask questions about satellite imagery and let SatQuery determine the appropriate analysis workflow.</div>',
    unsafe_allow_html=True,
)
st.info(
    "Current change detection is a classical pixelwise OpenCV baseline "
    "(RGB difference + Otsu threshold + morphology), not a trained AI model. "
    "Changed-pixel percentage is not geographic area; CRS and ground-sampling "
    "distance are not inferred when metadata is unavailable."
)

upload_left, upload_right = st.columns(2)
with upload_left:
    earlier_image = st.file_uploader(
        "Earlier Image",
        type=IMAGE_TYPES,
        key="earlier_image",
        help="Upload the earlier satellite scene (PNG, JPG, JPEG, or TIFF).",
    )
with upload_right:
    later_image = st.file_uploader(
        "Later Image",
        type=IMAGE_TYPES,
        key="later_image",
        help="Upload the later satellite scene (PNG, JPG, JPEG, or TIFF).",
    )

if earlier_image is not None or later_image is not None:
    preview_left, preview_right = st.columns(2)
    with preview_left:
        if earlier_image is not None:
            st.image(earlier_image, caption="Earlier Image", use_container_width=True)
        else:
            st.caption("Earlier image not uploaded yet.")
    with preview_right:
        if later_image is not None:
            st.image(later_image, caption="Later Image", use_container_width=True)
        else:
            st.caption("Later image not uploaded yet.")

st.markdown("#### Query")
query = st.text_input(
    "Natural-language query",
    value="What changed between these two dates?",
    help="Describe the analysis you want SatQuery to perform.",
)

analyze = st.button("Analyze", type="primary")

if analyze:

    st.session_state.routed_query = query
    # Clear results from a previous run before handling this submission.
    st.session_state.analysis_result = None
    st.session_state.execution_trace = None
    st.session_state.validation_results = None

    route_result = route_query(
        query,
        earlier_image,
        later_image
    )

    st.session_state.route_result = route_result

    if route_result["workflow"] == WORKFLOW_CHANGE:

        if earlier_image is None or later_image is None:

            st.error(
                "Bi-temporal Change Analysis requires both "
                "an earlier image and a later image."
            )

        else:

            earlier_validation = validate_image_file(
                earlier_image, name="Earlier image"
            )
            later_validation = validate_image_file(
                later_image, name="Later image"
            )
            pair_result = None
            if earlier_validation["valid"] and later_validation["valid"]:
                pair_result = check_pair_compatibility(
                    earlier_validation["inspection"],
                    later_validation["inspection"],
                )

            st.session_state.validation_results = {
                "earlier": earlier_validation,
                "later": later_validation,
                "pair": pair_result,
            }
            validation_errors = (
                earlier_validation["errors"]
                + later_validation["errors"]
                + (pair_result["errors"] if pair_result else [])
            )

            if validation_errors:
                st.error("Input validation failed. Change detection was not run.")
            else:
                try:

                    with st.spinner("Running change detection..."):

                        analysis_result = run_change_analysis(
                            earlier_image,
                            later_image
                        )

                    st.session_state.analysis_result = analysis_result

                    st.session_state.execution_trace = {
                        "status": "success",
                        "task": "Bi-temporal Change Analysis",
                        "query": query,
                        "specialist": "Change Detection Specialist",
                        "method": analysis_result["method"],
                        "algorithm": analysis_result["algorithm"],
                        "image_size": analysis_result["image_size"],
                        "changed_area_percent": analysis_result["changed_area_percent"],
                        "changed_pixel_count": analysis_result["changed_pixel_count"],
                        "total_pixel_count": analysis_result["total_pixel_count"],
                        "otsu_threshold": analysis_result["otsu_threshold"],
                        "runtime_seconds": analysis_result["runtime_seconds"],
                    }

                    st.success(
                        f"Pixels classified as changed: approximately "
                        f"{analysis_result['changed_area_percent']:.2f}% "
                        f"of image pixels (not geographic area)."
                    )

                except Exception as error:

                    st.error(
                        f"Change detection failed: {error}"
                    )

if st.session_state.route_result is not None:
    result = st.session_state.route_result
    st.markdown("---")
    st.markdown("## SatQuery Agent")

    st.markdown("#### Query")
    st.write(
        st.session_state.routed_query.strip() or "(empty query)"
    )

    st.markdown("#### Detected Workflow")
    st.write(result["workflow"])

    st.markdown("#### Selected Specialist")
    st.write(result["tool_name"])

    st.markdown("#### Input Validation")

    validation_results = st.session_state.get("validation_results")
    if validation_results:
        for label, key in (("Earlier image", "earlier"), ("Later image", "later")):
            item = validation_results[key]
            if item["valid"]:
                info = item["inspection"]
                st.write(
                    f"✓ {label}: {info['width']} × {info['height']}, "
                    f"{info['mode']} ({info['format'] or 'unknown format'})"
                )
            for message in item["errors"]:
                st.error(message)
            for message in item["warnings"]:
                st.warning(message)
        pair = validation_results["pair"]
        if pair:
            for message in pair["errors"]:
                st.error(message)
            for message in pair["warnings"]:
                st.warning(message)
            if pair["compatible"]:
                st.write("✓ Pair dimensions are compatible for the current pixelwise detector.")
        st.caption(
            "CRS and ground-sampling distance are not inferred from PNG/JPEG "
            "or from files that do not provide those metadata values."
        )

    elif st.session_state.get("analysis_result") is not None:

        st.write(
            "✓ Both earlier and later images provided. "
            "Inputs are sufficient and change analysis completed successfully."
        )

    else:

        st.write(result["input_status"])

    st.markdown("#### Routing Explanation")

    if st.session_state.get("analysis_result") is not None:

        st.write(
            "Query routed to the Bi-temporal Change Analysis "
            "specialist and analysis completed successfully."
        )

    else:

        st.write(result["reason"])
    
    if st.session_state.get("analysis_result") is not None:

        analysis_result = st.session_state.analysis_result

        st.markdown("---")

        st.markdown("### Change Analysis Evidence")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.image(
                earlier_image,
                caption="Earlier Image",
                use_container_width=True,
            )

        with col2:
            st.image(
                later_image,
                caption="Later Image",
                use_container_width=True,
            )

        with col3:
            st.image(
                analysis_result["evidence_bytes"],
                caption="Detected Change Evidence",
                use_container_width=True,
            )

        st.markdown("### Analysis Statistics")

        stat_col1, stat_col2, stat_col3 = st.columns(3)

        with stat_col1:
            st.metric(
                "Changed Pixels (%)",
                f"{analysis_result['changed_area_percent']:.2f}%"
            )

        with stat_col2:
            st.metric(
                "Changed Pixels",
                f"{analysis_result['changed_pixel_count']:,}"
            )

        with stat_col3:
            st.metric(
                "Runtime",
                f"{analysis_result['runtime_seconds']:.3f}s"
            )

        st.markdown("### Detection Details")

        st.write(
            f"**Method:** {analysis_result['method']}"
        )

        st.write(
            f"**Algorithm:** {analysis_result['algorithm']}"
        )

        st.write(
            f"**Image Size:** {analysis_result['image_size']}"
        )

        st.write(
            f"**Otsu Threshold:** {analysis_result['otsu_threshold']} "
        )
        
        st.markdown("### Execution Trace")
        if st.session_state.get("execution_trace") is not None:

            st.json(st.session_state.execution_trace)
    else:
        if result["workflow"] == "Single-Image VQA / Description":
            st.warning(
                "Single-image VQA and scene description are not implemented. "
                "No answer or caption was generated."
            )
        elif result["workflow"] == "Optical + SAR Analysis":
            st.warning(
                "Optical + SAR analysis is not implemented. "
                "No multimodal analysis was performed."
            )
        elif result["workflow"] == "Grounding":
            st.warning(
                "Visual grounding is not implemented. "
                "No objects were located or highlighted."
            )
        else:
            st.caption("No specialist analysis was performed.")

with st.sidebar:
    st.markdown("### System Status")
    st.markdown(
        """
        <div class="satquery-status">
        <b>Interface:</b> Ready<br>
        <b>Image Input:</b> Ready<br>
        <b>Query Router:</b> Ready<br>
        <b>Change Detection:</b> Ready<br>
        <b>Single-Image VQA / Description:</b> Not implemented<br>
        <b>Optical + SAR Analysis:</b> Not implemented
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("SatQuery routes natural-language queries to specialist analysis workflows.")
