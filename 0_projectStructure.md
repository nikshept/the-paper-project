## Project Summary — Pipeline + UI, Fully Detailed

**One Streamlit app, 3 screens, wrapping the 4-stage pipeline underneath.**

---

### Screen 1 — Upload *(runs Stage 1 + Stage 2, automatic)*

**UI:** `st.file_uploader` — drag-and-drop `.docx`/`.pdf`, no terminal. On submit, backend runs Stages 1–2 with a loading spinner, then stores results in `st.session_state`.

**Stage 1 — Extract**
- **Input:** Word document → **Input type:** `.docx` file
- **Tools:**
  - `docx2pdf` / LibreOffice (`soffice --headless`) — converts docx to PDF
  - `pdf2image` — converts PDF pages to raster images
  - `pytesseract` — OCR; extracts text strings + bounding box coordinates (x, y, w, h)
  - `opencv-python` (`cv2.HoughCircles`) — detects circular bubbles; returns center + radius
  - `python-docx` — reads table structure/cell text directly, bypassing OCR for tabular content
- **Output:** flat list of detected elements with pixel coordinates → **Output type:** list of dicts / JSON

**Stage 2 — Structure**
- **Input:** Stage 1's element list → **Input type:** list of dicts / JSON
- **Tools:**
  - Custom Python function (y-threshold grouping) — clusters bubbles into rows
  - `scipy.spatial.KDTree` (or manual nearest-neighbor) — matches each bubble row to its nearest question text
  - Custom Python function (averaging) — computes spacing/radius/margins from the grouped coordinates
- **Output 1:** content structure (question text + type, no coordinates) → JSON
- **Output 2:** layout config (measured spacing values) → JSON

---

### Screen 2 — Adjust *(Stage 3, the only human step)*

**UI:** loads `content_json` + `layout_config` from session state. Sliders/number inputs bound to layout fields (spacing, bubble radius, margins). Shared `render()` function re-runs on every change, showing a live PDF preview inline (rendered page → image via `pdf2image`). Save button writes the finalized config.

- **Input:** Stage 2's content JSON (for preview) + layout config (as starting values) → two JSON files
- **Tools:**
  - `streamlit` — the UI itself
  - `reportlab` — the shared `render()` function; draws text, bubbles, tables, corner markers using current config values
  - `qrcode` + `reportlab.lib.utils.ImageReader` — placeholder QR shown in preview
- **Output:** finalized layout config (human-approved values) → JSON file

---

### Screen 3 — Generate *(Stage 4, automatic)*

**UI:** button to generate either a single test PDF or a batch (with a participant list upload). Produces downloadable PDF(s) directly in-browser.

- **Input:** content JSON (unchanged) + finalized layout config + participant list → two JSON files + list/CSV
- **Tools:**
  - Same `render()` function from Screen 2 (imported, unchanged) — called once or looped per participant
  - `qrcode` — generates a unique QR per participant ID, replacing the placeholder
- **Output:** one or many finalized PDFs → `.pdf` file(s), named by participant ID

---

**Key design point carried through:** `render()` is written once and shared — Screen 2 calls it repeatedly for live preview, Screen 3 calls it for final output. Nothing is duplicated, and every input at every stage traces back to a real output from the stage before it.


Debug notes

Here's what I found, organized by type. I searched rather than relying on memory, so these are real, current links.

## Blog tutorials (closest match — same core technique)

- **[PyImageSearch — Bubble sheet scanner & test grader](https://pyimagesearch.com/2016/10/03/bubble-sheet-multiple-choice-scanner-and-test-grader-using-omr-python-and-opencv/)** — this is *the* canonical tutorial for this exact approach. Directly relevant: it explicitly discusses adding a minimum-pixel-count threshold to catch unmarked bubbles, and flagging when multiple bubbles clear the threshold — the same `no_mark_detected`/`multiple_marks_detected` logic this project uses, arrived at independently.
- **[PyImageSearch — Adaptive Thresholding with cv2.adaptiveThreshold](https://pyimagesearch.com/2021/05/12/adaptive-thresholding-with-opencv-cv2-adaptivethreshold/)** — direct explanation of the exact function `_binarize()` calls.
- **[LearnOpenCV — ArUco Augmented Reality in OpenCV](https://learnopencv.com/augmented-reality-using-aruco-markers-in-opencv-c-python/)** — the canonical resource for ArUco-marker homography, matching `dewarp.py`'s approach.

## Academic paper (the strongest direct match)

- **[PLOS ONE — "A new method of mark detection for software-based optical mark recognition"](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0206420)** — a real peer-reviewed comparison of pixel-counting (what this project does) against other detection methods, with actual sensitivity/specificity numbers on thousands of real filled forms, including under print/scan artifacts. Directly useful for your threshold-tuning: it quantifies exactly the tradeoff you've been hand-tuning by feel.

## Workflow examples (open-source implementations, similar architecture)

- **[bthicks/OMR-Grader](https://github.com/bthicks/OMR-Grader)** — full bubble-sheet grading pipeline.
- **[ShadyAbuKalam/Bubble-Sheet-Corrector](https://github.com/ShadyAbuKalam/Bubble-Sheet-Corrector)** — explicitly notes needing special-case handling for "most filled contour" logic, same class of edge case you've been hitting.
- **[sakethbachu/OMR-scanner](https://github.com/sakethbachu/OMR-scanner)** — built for an actual live competition (grading real submitted sheets at scale), closer in spirit to your actual use case than a toy demo.

## What I didn't find

Formal course material specifically on this technique — general computer vision syllabi exist (Gonzalez & Woods' *Digital Image Processing* textbook comes up as the standard reference across several), but nothing narrowly focused on OMR/bubble-sheet scoring as a course unit. The PyImageSearch tutorial above functions as the closest thing to structured teaching material in this space. I also didn't find a strong, focused forum thread (Reddit/StackOverflow) specifically about tuning ink-threshold false positives — closest was the same PyImageSearch post's own comment discussion.
