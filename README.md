# SatQuery AI baseline

This repository contains a Streamlit interface and a classical, pixelwise
bi-temporal change-detection baseline. It is not a trained remote-sensing
model. The other named workflows are routing placeholders and do not run
specialist analysis yet.

## Current demo scope

The implemented analysis is limited to the existing earlier/later image
change-detection workflow. Single-image visual question answering (VQA),
scene description/captioning, visual grounding, and optical + SAR analysis
are not implemented. The app may route a query to one of those categories,
but it does not generate an answer, caption, localization, or multimodal
result. The current change detector does not process SAR imagery as a
separate modality and has not been verified against the hidden SIH
Cartosat-2S/RISAT evaluation.

## Requirements

- 64-bit Python 3.13
- The packages pinned in `requirements.txt`

Python 3.13 is the version used by the existing project virtual environment.

## Setup and run (Windows PowerShell)

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local URL printed by Streamlit. Upload the earlier and later images,
ask a change-related question, and select **Analyze**.

## Setup and run (macOS/Linux)

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py
```

## Checks

Run the automated checks from the repository root:

```sh
python -m unittest discover -s . -p 'test_*.py' -v
```

The detector integration test reads the included `data/oscd/earlier.png` and
`data/oscd/later.png` files. It writes its generated mask and evidence only to
a temporary directory; it does not overwrite the sample files under `data/`.
The standalone `python change_detection.py` command writes generated files to
`outputs/change_detection/`, outside the source dataset.

## Current detector limits

The detector decodes both inputs as three-channel color images, requires equal
dimensions, computes RGB Euclidean differences, applies Otsu thresholding, and
cleans the mask with morphological opening and closing. It does not register
images, calibrate sensor values, read or validate CRS or ground-sampling
distance (GSD), or estimate geographic area. The displayed percentage is the
share of pixels classified as changed, not a geographic area measurement.

PNG/JPEG are accepted only for prescribed/public benchmark data and trigger a
warning. TIFF is accepted, but CRS/GSD values are not verified by this
prototype. Do not infer those values for images that do not provide them.

## Legacy copies

The root `app.py` imports `core/validator.py` and `change_detection.py`; root
tests exercise those modules. The separate `zzzzzz/` directory and the root
`app_backup.py` and `app_before_phase1.py` are not imported by the app or the
root test suite. Their purpose is unclear, so they are retained untouched.
