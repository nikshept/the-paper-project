"""
Main app -- two top-level tabs: Generate and Detect. See core/overlay.py
for the stamping logic and core/identify.py for the detection logic;
this file is purely layout/UI.

Run with:
    streamlit run app.py
"""
import base64
import csv
import io
import json
import os
import zipfile
from dataclasses import asdict

import cv2
import numpy as np
import pymupdf
import streamlit as st
from components.box_marker import box_marker

from core.overlay import compose_page, preview_page_png
from core.overlay_config import OverlayConfig
from core.identify import identify_photos, get_qr_position_hint
from core.extract import extract
from core.structure import structure
from core.template_builder import guess_questions, resolve_guess_text, QuestionGuess
from core.score import build_blank_references, score_response, build_export_xlsx, build_debug_report_xlsx

st.set_page_config(page_title="Paper Survey Tool", layout="wide")


def crop_bbox_from_pdf(pdf_bytes: bytes, page_num: int, bbox_pts: list,
                        dpi: int = 150, margin_pt: float = 8, highlight_bboxes: list = None) -> bytes:
    """Renders one page of a PDF and crops it to a bbox given in PDF
    points (top-left origin) -- same coordinate convention extract.py's
    bboxes already use, and PyMuPDF's own pixmap space, so no y-flip or
    other conversion is needed, just a scale by dpi/72. Used to show the
    researcher an actual image of what a guessed question looks like on
    the real page, not just its extracted text.

    highlight_bboxes, if given, draws a red rectangle for each (also in
    PDF points) -- used to show exactly which regions were detected as
    response options, so the researcher can visually verify the count
    and positions rather than trusting a bare number."""
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    page = doc[page_num]
    pix = page.get_pixmap(dpi=dpi)
    scale = dpi / 72.0
    x0, y0, x1, y1 = bbox_pts
    crop_x0, crop_y0 = max(0, (x0 - margin_pt) * scale), max(0, (y0 - margin_pt) * scale)
    crop_x1, crop_y1 = min(pix.width, (x1 + margin_pt) * scale), min(pix.height, (y1 + margin_pt) * scale)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    crop = cv2.cvtColor(arr[int(crop_y0):int(crop_y1), int(crop_x0):int(crop_x1)], cv2.COLOR_RGB2BGR)
    doc.close()

    if highlight_bboxes:
        crop = crop.copy()
        for hb in highlight_bboxes:
            hx0, hy0, hx1, hy1 = hb
            rx0, ry0 = int(hx0 * scale - crop_x0), int(hy0 * scale - crop_y0)
            rx1, ry1 = int(hx1 * scale - crop_x0), int(hy1 * scale - crop_y0)
            cv2.rectangle(crop, (rx0, ry0), (rx1, ry1), (0, 0, 255), 2)

    success, enc = cv2.imencode(".png", crop)
    return enc.tobytes()



def crop_bbox_from_image(img_bgr, bbox_pts: list, margin_pt: float = 4, highlight_bboxes: list = None) -> bytes:
    """Same idea as crop_bbox_from_pdf, but for an already-in-memory
    dewarped photo (numpy array) instead of re-rendering a PDF page --
    used to show a respondent's actual marked answer, not the blank
    template. Uses dewarp.py's own SCALE so points->pixels matches
    exactly what score.py's ink-diff already assumes.

    highlight_bboxes, if given (in PDF points, same as option_elements'
    own "bbox" values), draws the EXACT region _ink_score reads from --
    for debugging exactly what the scorer sees on a real photo, not an
    approximation of it."""
    from core.dewarp import SCALE
    h, w = img_bgr.shape[:2]
    x0, y0, x1, y1 = bbox_pts
    crop_x0, crop_y0 = max(0, int((x0 - margin_pt) * SCALE)), max(0, int((y0 - margin_pt) * SCALE))
    crop_x1, crop_y1 = min(w, int((x1 + margin_pt) * SCALE)), min(h, int((y1 + margin_pt) * SCALE))
    crop = img_bgr[crop_y0:crop_y1, crop_x0:crop_x1].copy()
    if highlight_bboxes:
        for hb in highlight_bboxes:
            hx0, hy0, hx1, hy1 = hb
            rx0, ry0 = int(hx0 * SCALE - crop_x0), int(hy0 * SCALE - crop_y0)
            rx1, ry1 = int(hx1 * SCALE - crop_x0), int(hy1 * SCALE - crop_y0)
            cv2.rectangle(crop, (rx0, ry0), (rx1, ry1), (0, 0, 255), 1)
    success, enc = cv2.imencode(".png", crop)
    return enc.tobytes()


def render_full_page_with_boxes(pdf_bytes: bytes, page_num: int, target_width_px: int, highlight_bboxes: list,
                                 pending_bbox: list = None):
    """Renders a full page (no crop) with already-added option boxes
    drawn in green, for the manual click-and-drag add flow -- lets the
    researcher see what's already been marked while dragging the next
    box. Coordinates of any NEW box the researcher drags are read
    directly off this same image, so pixel->point conversion is a
    single consistent scale factor, no separate tracking of "displayed
    size" needed.

    DPI is computed from target_width_px (the actual page width, in
    points, is read directly here) rather than taking a fixed DPI --
    a fixed DPI produces a native pixel width that depends on the
    source page's point-size, which can exceed the column width it's
    displayed in and get visually cropped (confirmed: an A4 page at a
    DPI picked without this in mind rendered ~992px wide, too close to
    a typical 2/3-width column to reliably fit). Computing DPI from a
    target display width instead guarantees a consistent, fitting size
    regardless of page size, and -- critically -- means the image's
    NATIVE pixel size always matches its DISPLAYED size (no CSS
    scaling), so drag coordinates never need a separate correction
    factor for "shown smaller than actual."

    pending_bbox, if given, is the just-released (but not yet added)
    drag result, drawn in yellow.

    Returns (numpy array in RGB order, the scale factor used -- pass
    this to the caller's own pixel->point math, don't recompute it
    separately)."""
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    page = doc[page_num]
    dpi = int(72 * target_width_px / page.rect.width)
    pix = page.get_pixmap(dpi=dpi)
    scale = dpi / 72.0
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    img_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR).copy()
    doc.close()
    for bbox in highlight_bboxes:
        x0, y0, x1, y1 = bbox
        cv2.rectangle(img_bgr, (int(x0 * scale), int(y0 * scale)), (int(x1 * scale), int(y1 * scale)),
                     (0, 200, 0), 2)
    if pending_bbox:
        x0, y0, x1, y1 = pending_bbox
        cv2.rectangle(img_bgr, (int(x0 * scale), int(y0 * scale)), (int(x1 * scale), int(y1 * scale)),
                     (0, 220, 255), 2)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), scale


def render_zoomable_preview(png_bytes: bytes, height: int = 640):
    """A real, self-contained zoom/pan viewer, built as HTML/JS rather
    than st.image, since Streamlit's own fullscreen feature didn't
    actually enlarge anything and couldn't be fixed without a live
    browser session to inspect (now fixed properly with one, see below).

    Sizing: the image is CONTAIN-fit within the container by default
    (max-width/max-height with width/height:auto, centered) -- an
    earlier version used width:100% instead, which filled the
    container's width but let a portrait page overflow its height,
    so the bottom of the survey was cut off and needed scrolling to
    see at all.

    Zoom/pan persistence: every slider change triggers a full Streamlit
    rerun, which regenerates this HTML from scratch and destroys the
    iframe's JS state (scale/pan reset to nothing every time). A plain
    st.iframe has no return channel back to Python session_state, so
    the fix is client-side: save scale/pan to localStorage on every
    change, and read it back as the starting point on load. Confirmed
    live in-browser that this iframe's localStorage IS shared with the
    parent page's origin (a `srcdoc` iframe reports location.origin as
    the string "null", but still shares the parent's storage partition
    unless explicitly sandboxed) -- checked this empirically rather
    than assuming it from documentation wording.

    Mouse wheel zooms (toward center), click-and-drag pans, and
    +/-/Reset buttons cover the same actions for anyone who prefers
    not to scroll."""
    b64 = base64.b64encode(png_bytes).decode()
    html = f"""
    <div style="display:flex; flex-direction:column; align-items:center; font-family:sans-serif;">
      <div id="zoom-container" style="
            width:100%; height:{height - 50}px; overflow:hidden; position:relative;
            border:1px solid #444; border-radius:6px; cursor:grab; background:#0e1117;">
        <img id="zoom-img" src="data:image/png;base64,{b64}"
             style="position:absolute; top:50%; left:50%; max-width:100%; max-height:100%;
                    width:auto; height:auto; transform-origin: center center;
                    user-select:none; -webkit-user-drag:none;">
      </div>
      <div style="margin-top:8px; display:flex; gap:8px;">
        <button onclick="window.zoomIn()" style="padding:4px 14px; cursor:pointer;">+</button>
        <button onclick="window.zoomOut()" style="padding:4px 14px; cursor:pointer;">−</button>
        <button onclick="window.resetZoom()" style="padding:4px 14px; cursor:pointer;">Reset</button>
        <span style="align-self:center; color:#888; font-size:0.85rem;">scroll to zoom, drag to pan</span>
      </div>
    </div>
    <script>
      (function() {{
        const STORAGE_KEY = 'paper_project_preview_zoom';
        let saved = {{}};
        try {{ saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{{}}'); }} catch (e) {{ saved = {{}}; }}
        let scale = saved.scale || 1, panX = saved.panX || 0, panY = saved.panY || 0;
        let isDragging = false, startX, startY;
        const img = document.getElementById('zoom-img');
        const container = document.getElementById('zoom-container');

        function saveState() {{
          try {{ localStorage.setItem(STORAGE_KEY, JSON.stringify({{ scale, panX, panY }})); }} catch (e) {{}}
        }}

        function applyTransform() {{
          img.style.transform = `translate(${{panX}}px, ${{panY}}px) translate(-50%, -50%) scale(${{scale}})`;
          saveState();
        }}
        applyTransform();  // restores the last saved zoom/pan immediately, or the default centered fit

        container.addEventListener('wheel', function(e) {{
          e.preventDefault();
          const delta = e.deltaY < 0 ? 1.15 : 1/1.15;
          scale = Math.min(Math.max(scale * delta, 1), 6);
          applyTransform();
        }}, {{ passive: false }});

        container.addEventListener('mousedown', function(e) {{
          isDragging = true;
          startX = e.clientX - panX;
          startY = e.clientY - panY;
          container.style.cursor = 'grabbing';
        }});
        window.addEventListener('mousemove', function(e) {{
          if (!isDragging) return;
          panX = e.clientX - startX;
          panY = e.clientY - startY;
          applyTransform();
        }});
        window.addEventListener('mouseup', function() {{
          isDragging = false;
          container.style.cursor = 'grab';
        }});

        window.zoomIn = function() {{ scale = Math.min(scale * 1.3, 6); applyTransform(); }};
        window.zoomOut = function() {{ scale = Math.max(scale / 1.3, 1); applyTransform(); }};
        window.resetZoom = function() {{ scale = 1; panX = 0; panY = 0; applyTransform(); }};
      }})();
    </script>
    """
    st.iframe(html, height=height)

# ---- compact spacing + larger tab labels ----
# Streamlit's default vertical spacing is generous enough that the full
# control panel doesn't fit in one screen; this tightens it. Exact
# "fits with no scrolling at all" depends on the actual browser window
# size/zoom on your end -- this can't be guaranteed for every screen,
# only made meaningfully tighter than the default.
st.markdown("""
<style>
    /* padding-top must clear Streamlit's own fixed header (measured at
       60px live in-browser) -- 1.2rem (19.2px) left the nav buttons
       poking up BEHIND the header, invisibly clipping their top half.
       Confirmed this fix live before shipping it, not guessed. */
    .block-container { padding-top: 4.5rem; padding-bottom: 1rem; }
    [data-testid="stVerticalBlock"] { gap: 0.5rem; }

    /* Big Generate/Detect nav buttons ONLY -- scoped via their keys so
       other buttons (Generate sample, page arrows, etc.) are untouched.
       Sized down from an earlier version that was too large and ate
       too much vertical space -- still bigger than Streamlit's default
       so the current screen is unmistakable, just not oversized. */
    .st-key-nav_generate button, .st-key-nav_template button, .st-key-nav_upload button, .st-key-nav_review button {
        min-height: 2.8rem !important;
        height: auto !important;
        overflow: visible !important;
        white-space: normal !important;
        width: 100%;
    }
    .st-key-nav_generate button p, .st-key-nav_template button p, .st-key-nav_upload button p, .st-key-nav_review button p {
        font-size: 1.3rem !important;
        font-weight: 800 !important;
        line-height: 1.3 !important;
        margin: 0 !important;
    }

    /* Places the codes.csv caption INSIDE the uploader's own box, to the
       right of the Upload button, instead of below the whole box as a
       separate line -- there's dead space there by default. Uses :has()
       rather than Streamlit's internal emotion-cache class names (which
       aren't part of any stable API and could change between versions);
       :has() only depends on data-testid and the documented st-key-*
       convention, both stable. Confirmed these exact pixel offsets live
       against the running app before shipping -- they're tied to this
       uploader's specific box/button size, so if marker/QR control sizes
       change substantially later, these may need re-checking the same way. */
    [data-testid="stVerticalBlock"]:has(.st-key-codes_csv) {
        position: relative;
    }
    [data-testid="stVerticalBlock"]:has(.st-key-codes_csv) div:has(> [data-testid="stCaptionContainer"]) {
        position: absolute;
        top: -94px;
        left: 119px;
        width: 195px;
        margin: 0;
    }
</style>
""", unsafe_allow_html=True)

if "active_view" not in st.session_state:
    st.session_state.active_view = "Generate Survey"

nav_gen, nav_tpl, nav_up, nav_rev = st.columns(4, gap="small")
with nav_gen:
    if st.button("Generate Survey", key="nav_generate", width='stretch',
                 type="primary" if st.session_state.active_view == "Generate Survey" else "secondary"):
        st.session_state.active_view = "Generate Survey"
        st.rerun()
with nav_tpl:
    if st.button("Get Template", key="nav_template", width='stretch',
                 type="primary" if st.session_state.active_view == "Get Template" else "secondary"):
        st.session_state.active_view = "Get Template"
        st.rerun()
with nav_up:
    if st.button("Upload Responses", key="nav_upload", width='stretch',
                 type="primary" if st.session_state.active_view == "Upload Responses" else "secondary"):
        st.session_state.active_view = "Upload Responses"
        st.rerun()
with nav_rev:
    if st.button("Review Responses", key="nav_review", width='stretch',
                 type="primary" if st.session_state.active_view == "Review Responses" else "secondary"):
        st.session_state.active_view = "Review Responses"
        st.rerun()

st.divider()

tab_generate = st.session_state.active_view == "Generate Survey"
tab_template = st.session_state.active_view == "Get Template"
tab_upload = st.session_state.active_view == "Upload Responses"
tab_review_responses = st.session_state.active_view == "Review Responses"

# ============================================================= IDENTIFY ===
if tab_upload:
    st.caption("Upload the blank sample PDF you generated for printing, and a PDF of the "
               "photographed/scanned responses -- one survey page per photo, in order.")

    det_col1, det_col2 = st.columns(2)
    with det_col1:
        blank_upload = st.file_uploader("Blank sample PDF", type=["pdf"], key="detect_blank")
    with det_col2:
        photos_upload = st.file_uploader("Photos PDF", type=["pdf"], key="detect_photos")

    run_clicked = st.button("Run detection", disabled=(blank_upload is None or photos_upload is None))

    if run_clicked:
        progress = st.progress(0.0, text="Starting...")
        def _on_progress(i, n, result):
            status = "identified" if result.identified else ("dewarped, QR unreadable" if result.dewarped else "no markers found")
            progress.progress((i + 1) / n, text=f"Photo {i + 1} of {n}: {status}")
        results = identify_photos(blank_upload.getvalue(), photos_upload.getvalue(), on_progress=_on_progress)
        progress.empty()
        st.session_state.detect_results = results
        st.session_state.blank_pdf_bytes = blank_upload.getvalue()  # kept for the Review tab's
                                                                       # template-guessing step
        # recomputed here (not returned from identify_photos) purely for the
        # diagnostic crop display below -- lets a failed QR be visually
        # inspected directly in the app, without needing the raw photo
        blank_path_for_hints = None
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(blank_upload.getvalue())
                blank_path_for_hints = tmp.name
            st.session_state.detect_qr_hints = get_qr_position_hint(blank_path_for_hints)
        finally:
            if blank_path_for_hints:
                os.remove(blank_path_for_hints)

    if "detect_results" in st.session_state:
        results = st.session_state.detect_results
        n = len(results)
        n_dewarped = sum(1 for r in results if r.dewarped)
        n_identified = sum(1 for r in results if r.identified)
        failed_results = [r for r in results if not r.identified]

        st.divider()
        st.metric(f"Dewarped successfully", f"{n_dewarped} of {n}")
        st.metric(f"Identified successfully", f"{n_identified} of {n}")

        st.divider()
        if failed_results:
            st.subheader(f"Needs attention ({len(failed_results)})")
            st.caption("Only photos that didn't fully identify are shown here -- make a note "
                      "of these, then re-photograph/rescan them and re-run as a new PDF. "
                      "Successfully identified photos don't need review.")
            for r in failed_results:
                with st.container(border=True):
                    res_str = (f"{r.native_resolution[0]}\u00d7{r.native_resolution[1]}px"
                              if r.native_resolution else "unknown resolution")
                    if r.dewarped:
                        st.warning(f"Photo {r.photo_index}: geometry recovered, but QR unreadable. "
                                  f"{r.error}")
                    else:
                        st.error(f"Photo {r.photo_index}: {r.error}")
                    st.caption(f"Source image: {res_str}")

                    display_image = r.dewarped_image if r.dewarped_image is not None else r.original_image
                    if display_image is not None:
                        _, enc = cv2.imencode(".png", display_image)
                        render_zoomable_preview(enc.tobytes(), height=380)
                    else:
                        st.caption("(photo could not be read at all)")

                    if r.dewarped:
                        hints = st.session_state.get("detect_qr_hints", {})
                        hint = hints.get(r.matched_page_guess) if r.matched_page_guess is not None else None
                        if hint is not None and r.dewarped_image is not None:
                            cx_frac, cy_frac, size_frac = hint
                            h, w = r.dewarped_image.shape[:2]
                            cx, cy = int(w * cx_frac), int(h * cy_frac)
                            half = int(w * size_frac / 2 * 1.6)  # extra margin, purely for
                                                                    # human viewing context
                            y0, y1 = max(0, cy - half), min(h, cy + half)
                            x0, x1 = max(0, cx - half), min(w, cx + half)
                            crop = r.dewarped_image[y0:y1, x0:x1]
                            if crop.size > 0:
                                st.caption("What the code is actually looking at (QR search region):")
                                st.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), width=220)
        else:
            st.success("All photos identified -- nothing needs review.")

        if n_identified > 0:
            st.divider()
            st.subheader("Proceed with the successful ones")
            st.caption(f"{n_identified} photo(s) identified this run: "
                      + ", ".join(f"{r.tracking_code} (p{r.page_num})"
                                  for r in results if r.identified))
            if st.button(f"Proceed with identified pages ({n_identified})", type="primary"):
                st.session_state.identified_for_review = [r for r in results if r.identified]
                st.session_state.active_view = "Review Responses"
                st.rerun()


# ========================================================= GET TEMPLATE ===
if tab_template:
    blank_upload_tpl = st.file_uploader("Blank form", type=["pdf"], key="template_blank_upload")

    if blank_upload_tpl is None:
        st.info("Upload a blank form to guess its question/answer template.")
    else:
        st.session_state.template_blank_bytes = blank_upload_tpl.getvalue()

        # ---- build the guessed template once, cache it in session_state ----
        if "template_guesses" not in st.session_state:
            with st.spinner("Reading the blank reference to guess the question/answer template..."):
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp.write(st.session_state.template_blank_bytes)
                    tmp_path = tmp.name
                try:
                    extract_result = extract(tmp_path)
                finally:
                    os.remove(tmp_path)
                structured = structure(extract_result)
                st.session_state.template_guesses = guess_questions(structured)
                st.session_state.template_structured = structured  # kept so the +/- line
                                                                       # controls below can
                                                                       # recompute on demand

        st.subheader("First let's review if I identified the template correctly")
        st.caption("Each guess below pairs a question's text with the response options "
                  "found near it. Confirm the ones that are right, fix the ones that "
                  "aren't, and discard anything that isn't really a question -- this "
                  "becomes the template used to score every photo in this batch.")

        guesses = st.session_state.template_guesses
        total = len(guesses)
        active_guesses = [g for g in guesses if not g.discarded]
        n_confirmed = sum(1 for g in active_guesses if g.confirmed)

        crop_margin = st.slider("Preview image padding (points)", 0, 60, 8,
                                help="Purely visual -- adds blank space around the preview "
                                     "image. To change what CONTENT is included as the "
                                     "question's text, use the +/- line controls on each "
                                     "question below instead.")

        if "review_index" not in st.session_state:
            st.session_state.review_index = 0
        idx = st.session_state.review_index

        st.caption(f"Reviewing {min(idx + 1, total)} of {total} \u2014 "
                  f"{n_confirmed} confirmed, {total - len(active_guesses)} discarded so far.")

        if idx < total:
            g = guesses[idx]
            with st.container(border=True):
                crop_bytes = crop_bbox_from_pdf(st.session_state.template_blank_bytes, g.page,
                                                g.preview_bbox, margin_pt=crop_margin,
                                                highlight_bboxes=[el["bbox"] for el in g.option_elements])
                st.image(crop_bytes)
                st.caption("Red boxes = detected response options. Check the count and "
                          "positions actually match the real options before confirming.")

                if g.is_manual:
                    st.caption("Manually marked question -- line adjustment controls don't "
                              "apply (there's no auto-detected row to adjust from).")
                else:
                    lcol1, lcol2, lcol3, lcol4 = st.columns(4)
                    with lcol1:
                        if st.button("+ Include line above", key=f"more_line_{idx}"):
                            rows = st.session_state.template_structured["pages"][g.page]["rows"]
                            g.text_line_count = min(g.text_line_count + 1, g.row_index)
                            new_text, new_bbox, new_options = resolve_guess_text(
                                rows, g.row_index, g.text_line_count, g.text_line_count_below)
                            g.question_text_guess, g.preview_bbox = new_text, new_bbox
                            if not g.options_manually_set:
                                g.option_elements = new_options
                            g.rev += 1
                            st.rerun()
                    with lcol2:
                        if st.button("\u2212 Exclude last line above", key=f"less_line_{idx}",
                                     disabled=g.text_line_count == 0):
                            rows = st.session_state.template_structured["pages"][g.page]["rows"]
                            g.text_line_count = max(g.text_line_count - 1, 0)
                            new_text, new_bbox, new_options = resolve_guess_text(
                                rows, g.row_index, g.text_line_count, g.text_line_count_below)
                            g.question_text_guess, g.preview_bbox = new_text, new_bbox
                            if not g.options_manually_set:
                                g.option_elements = new_options
                            g.rev += 1
                            st.rerun()
                    with lcol3:
                        if st.button("+ Include line below", key=f"more_line_below_{idx}"):
                            rows = st.session_state.template_structured["pages"][g.page]["rows"]
                            g.text_line_count_below = min(g.text_line_count_below + 1,
                                                           len(rows) - g.row_index - 1)
                            new_text, new_bbox, new_options = resolve_guess_text(
                                rows, g.row_index, g.text_line_count, g.text_line_count_below)
                            g.question_text_guess, g.preview_bbox = new_text, new_bbox
                            if not g.options_manually_set:
                                g.option_elements = new_options
                            g.rev += 1
                            st.rerun()
                    with lcol4:
                        if st.button("\u2212 Exclude last line below", key=f"less_line_below_{idx}",
                                     disabled=g.text_line_count_below == 0):
                            rows = st.session_state.template_structured["pages"][g.page]["rows"]
                            g.text_line_count_below = max(g.text_line_count_below - 1, 0)
                            new_text, new_bbox, new_options = resolve_guess_text(
                                rows, g.row_index, g.text_line_count, g.text_line_count_below)
                            g.question_text_guess, g.preview_bbox = new_text, new_bbox
                            if not g.options_manually_set:
                                g.option_elements = new_options
                            g.rev += 1
                            st.rerun()

                new_text = st.text_input("Question text", value=g.question_text_guess, key=f"qtext_{idx}_{g.rev}")
                new_label = st.text_input("Label (e.g. Q1, K3) -- optional",
                                          value=g.label or "", key=f"qlabel_{idx}_{g.rev}")
                type_options = ["mcq", "open_text"]
                new_type = st.selectbox("Question type", type_options,
                                        index=type_options.index(g.question_type),
                                        format_func=lambda t: "Multiple choice" if t == "mcq" else "Open text",
                                        key=f"qtype_{idx}_{g.rev}",
                                        help="Guessed automatically -- check it's right, especially "
                                             "if this only has one option marked (that usually means "
                                             "open text, but double-check).")
                st.caption(f"{len(g.option_elements)} option(s) detected on page {g.page + 1}")

                gcol1, gcol2 = st.columns(2)
                with gcol1:
                    if st.button("✅ Confirm", key=f"confirm_{idx}", width='stretch', type="primary"):
                        g.question_text_guess = new_text
                        g.label = new_label or None
                        g.question_type = new_type
                        g.confirmed = True
                        st.session_state.review_index += 1
                        st.rerun()
                with gcol2:
                    if st.button("Discard (not a question)", key=f"discard_{idx}", width='stretch'):
                        g.discarded = True
                        st.session_state.review_index += 1
                        st.rerun()

            if idx > 0:
                if st.button("\u2190 Back", key=f"back_{idx}"):
                    st.session_state.review_index -= 1
                    st.rerun()
        else:
            st.divider()
            if active_guesses and n_confirmed == len(active_guesses):
                st.success(f"All {n_confirmed} question(s) confirmed.")
                template_json = json.dumps(
                    [asdict(g) for g in active_guesses if g.confirmed], indent=2)
                st.download_button("Download template as JSON", template_json,
                                    file_name="survey_template.json", mime="application/json")
            else:
                st.info(f"Reviewed all {total} guess(es) -- {n_confirmed} confirmed, "
                       f"{total - len(active_guesses)} discarded.")
            if st.button("\u2190 Back", key="back_end"):
                st.session_state.review_index -= 1
                st.rerun()

            st.divider()
            st.subheader("Missed a question?")
            st.caption("Draw a box below to add.")

            structured = st.session_state.template_structured
            page_nums = [p["page"] for p in structured["pages"]]

            if "manual_page_idx" not in st.session_state:
                st.session_state.manual_page_idx = 0
            st.session_state.manual_page_idx = max(0, min(st.session_state.manual_page_idx, len(page_nums) - 1))
            manual_page = page_nums[st.session_state.manual_page_idx]

            if st.session_state.get("manual_last_page") != manual_page:
                st.session_state.manual_pending_box = None
                st.session_state.manual_last_coords_key = None
                st.session_state.manual_last_page = manual_page

            if "manual_option_boxes" not in st.session_state:
                st.session_state.manual_option_boxes = []
            if "manual_pending_box" not in st.session_state:
                st.session_state.manual_pending_box = None
            if "manual_last_coords_key" not in st.session_state:
                st.session_state.manual_last_coords_key = None

            img_col, ctrl_col = st.columns([2, 1])

            with img_col:
                TARGET_WIDTH_PX = 700
                page_img_array, render_scale = render_full_page_with_boxes(
                    st.session_state.template_blank_bytes, manual_page, TARGET_WIDTH_PX,
                    st.session_state.manual_option_boxes, pending_bbox=st.session_state.manual_pending_box)
                coords = box_marker(page_img_array, key=f"manual_click_{manual_page}")

                # Each completed drag is added as an option immediately on release --
                # no separate "Add" click needed. "Undo" (in ctrl_col) removes one
                # if a drag was mismarked.
                if coords and all(k in coords for k in ("x1", "y1", "x2", "y2")):
                    coords_key = (coords["x1"], coords["y1"], coords["x2"], coords["y2"])
                    if coords_key != st.session_state.manual_last_coords_key:
                        st.session_state.manual_last_coords_key = coords_key
                        x0px, x1px = sorted([coords["x1"], coords["x2"]])
                        y0px, y1px = sorted([coords["y1"], coords["y2"]])
                        if x1px - x0px > 3 and y1px - y0px > 3:  # ignore tiny/accidental clicks
                            new_box = [x0px / render_scale, y0px / render_scale,
                                      x1px / render_scale, y1px / render_scale]
                            st.session_state.manual_option_boxes.append(new_box)
                            st.session_state.manual_pending_box = None
                            st.rerun()

            with ctrl_col:
                pcol1, pcol2, pcol3 = st.columns([1, 2, 1])
                with pcol1:
                    if st.button("\u2190", key="manual_page_prev",
                                 disabled=st.session_state.manual_page_idx == 0):
                        st.session_state.manual_page_idx -= 1
                        st.rerun()
                with pcol2:
                    st.markdown(f"<div style='text-align:center'>Page {st.session_state.manual_page_idx + 1}</div>",
                               unsafe_allow_html=True)
                with pcol3:
                    if st.button("\u2192", key="manual_page_next",
                                 disabled=st.session_state.manual_page_idx >= len(page_nums) - 1):
                        st.session_state.manual_page_idx += 1
                        st.rerun()

                st.caption(f"{len(st.session_state.manual_option_boxes)} option(s) marked.")
                manual_type_options = ["mcq", "open_text"]
                manual_type = st.selectbox("Question type", manual_type_options,
                                           format_func=lambda t: "Multiple choice" if t == "mcq" else "Open text",
                                           key="manual_question_type")

                ucol, scol = st.columns(2)
                with ucol:
                    if st.button("Undo", key="remove_manual_box",
                                 disabled=not st.session_state.manual_option_boxes):
                        st.session_state.manual_option_boxes.pop()
                        st.rerun()
                with scol:
                    can_save = bool(st.session_state.manual_option_boxes)
                    if st.button("Submit Question", key="save_manual_question", type="primary",
                                 disabled=not can_save):
                        option_elements = [{"type": "shape", "bbox": b} for b in st.session_state.manual_option_boxes]
                        x0 = min(el["bbox"][0] for el in option_elements)
                        y0 = min(el["bbox"][1] for el in option_elements)
                        x1 = max(el["bbox"][2] for el in option_elements)
                        y1 = max(el["bbox"][3] for el in option_elements)
                        combined_bbox = [x0, y0, x1, y1]
                        # Question text isn't asked here -- the review card this drops into
                        # right after already has its own text field, and asking twice was
                        # pure redundancy (confirmed confusing: typing it once here, then
                        # being asked again immediately on the next screen).
                        new_guess = QuestionGuess(page=manual_page, row_index=-1,
                                                  question_text_guess="",
                                                  option_elements=option_elements, preview_bbox=combined_bbox,
                                                  text_line_count=0, options_manually_set=True, is_manual=True,
                                                  question_type=manual_type)
                        st.session_state.template_guesses.append(new_guess)
                        st.session_state.manual_option_boxes = []
                        st.session_state.manual_pending_box = None
                        st.session_state.manual_last_coords_key = None
                        st.session_state.review_index = len(st.session_state.template_guesses) - 1
                        st.rerun()

                st.divider()
                template_json = json.dumps([asdict(g) for g in active_guesses if g.confirmed], indent=2)
                st.download_button("Download template", template_json,
                                    file_name="survey_template.json", mime="application/json",
                                    key="download_template_from_missing", type="primary")



# ===================================================== REVIEW RESPONSES ===
if tab_review_responses:
    if not st.session_state.get("identified_for_review"):
        st.info("Upload responses first to start reviewing.")
    else:
        # ---- resolve the confirmed template: prefer this session's
        # Get Template work, but allow an uploaded template JSON too,
        # e.g. if the template was built in an earlier session ----
        confirmed_template = None
        if st.session_state.get("template_guesses"):
            confirmed = [g for g in st.session_state.template_guesses if g.confirmed and not g.discarded]
            if confirmed:
                confirmed_template = [asdict(g) for g in confirmed]
                st.caption(f"Using the {len(confirmed_template)}-question template confirmed in 'Get Template'.")

        template_json_upload = st.file_uploader("Or upload a template JSON", type=["json"],
                                                 key="review_template_upload")
        if template_json_upload is not None:
            confirmed_template = json.loads(template_json_upload.getvalue())

        # Reuse whatever blank-form bytes are already in session from EITHER
        # Get Template or Upload Responses -- both already have the same
        # underlying file by the time someone reaches this tab in the normal
        # flow, so there's no real reason to ask a third time. Only fall
        # back to a fresh upload if genuinely neither is available.
        blank_bytes_for_scoring = (st.session_state.get("template_blank_bytes")
                                   or st.session_state.get("blank_pdf_bytes"))
        if blank_bytes_for_scoring is None:
            blank_upload_rev = st.file_uploader("Blank form (needed to score answers against)",
                                                type=["pdf"], key="review_blank_upload")
            if blank_upload_rev is not None:
                blank_bytes_for_scoring = blank_upload_rev.getvalue()

        if not confirmed_template:
            st.info("Build and confirm a template in 'Get Template' first, or upload one above.")
        elif blank_bytes_for_scoring is None:
            st.info("Upload the blank form above so answers can be scored against it.")
        else:
            # ---- score every respondent once, cache in session ----
            if "scored_responses" not in st.session_state:
                with st.spinner("Scoring responses against the template..."):
                    blank_refs = build_blank_references(blank_bytes_for_scoring)
                    respondents = {}
                    for r in st.session_state.identified_for_review:
                        respondents.setdefault(r.tracking_code, {})[r.page_num] = r.dewarped_image
                    scored = {code: score_response(pages, blank_refs, confirmed_template)
                             for code, pages in respondents.items()}
                    st.session_state.scored_responses = scored
                    st.session_state.scored_respondents = respondents
                    st.session_state.scored_template = confirmed_template
                    st.session_state.corrections = {}
                    # flag_index has its OWN separate "not in session_state" check further
                    # down, unrelated to whether scoring was just freshly redone -- without
                    # resetting it here too, a leftover value from an earlier, larger review
                    # queue in the same browser tab can sit above this run's actual queue
                    # length, skipping straight to "all reviewed" without ever showing any
                    # of THIS run's flagged items. Confirmed as the actual cause of exactly
                    # that symptom, not a guess.
                    st.session_state.flag_index = 0
                    # flagged_queue must be a STABLE, fixed list computed ONCE here, not
                    # rebuilt every rerun while filtering out corrected items. Rebuilding
                    # it each render (the previous approach) made it shrink every time an
                    # answer got corrected, which silently shifted every later item's
                    # position -- flag_index=3 could point at a completely different
                    # question after a correction than it did a moment before, and "Back"
                    # would land on something never even seen yet rather than the item
                    # just answered. Confirmed as the actual mechanism, not a guess.
                    flagged_queue = []
                    for code in sorted(st.session_state.scored_responses.keys()):
                        for qi in range(len(confirmed_template)):
                            sc = st.session_state.scored_responses[code].get(qi)
                            if sc is None or sc.get("flagged"):
                                flagged_queue.append((code, qi))
                    st.session_state.flagged_queue = flagged_queue

            scored = st.session_state.scored_responses
            respondents = st.session_state.scored_respondents
            template = st.session_state.scored_template
            corrections = st.session_state.corrections
            flagged_queue = st.session_state.flagged_queue

            n_total_answers = len(scored) * len(template)
            n_remaining = sum(1 for item in flagged_queue if item not in corrections)
            st.metric("Respondents", len(scored))
            st.metric("Needs review", f"{n_remaining} of {n_total_answers}")

            st.divider()
            crop_margin_review = st.slider("Preview image padding (points)", 0, 60, 4,
                                           key="review_crop_margin",
                                           help="How much extra space shows around the cropped "
                                                "answer below. Kept tight by default since rows "
                                                "in a dense table can sit close together -- too "
                                                "much padding can pull in a neighboring row.")
            if "flag_index" not in st.session_state:
                st.session_state.flag_index = 0
            fidx = st.session_state.flag_index

            if flagged_queue and fidx < len(flagged_queue):
                code, qi = flagged_queue[fidx]
                g = template[qi]
                score = scored[code].get(qi)
                # Falls back to the old implicit heuristic if question_type is missing
                # entirely (a template saved before this field existed) -- otherwise an
                # old template's open-text questions would silently default to "mcq" and
                # reintroduce the exact bug this field was added to fix.
                is_open_text = g.get("question_type") or ("open_text" if len(g["option_elements"]) == 1 else "mcq")
                is_open_text = is_open_text == "open_text"

                st.caption(f"Reviewing {fidx + 1} of {len(flagged_queue)} flagged answer(s)")
                existing_correction = corrections.get((code, qi))
                if existing_correction is not None:
                    st.info(f"Already corrected as: {existing_correction}. Submitting again below will overwrite it.")
                with st.container(border=True):
                    st.write(f"**{code}** \u2014 {g.get('label') or g.get('question_text_guess', '')}")

                    page_img = respondents.get(code, {}).get(g["page"])
                    if page_img is not None:
                        crop_bytes = crop_bbox_from_image(page_img, g["preview_bbox"], margin_pt=crop_margin_review,
                                                          highlight_bboxes=[el["bbox"] for el in g["option_elements"]])
                        st.image(crop_bytes)
                    else:
                        st.warning(f"No photo for page {g['page'] + 1} from this respondent.")

                    if is_open_text:
                        text_val = st.text_area("Transcription", key=f"transcribe_{fidx}")
                        if st.button("Save and next", key=f"save_text_{fidx}", type="primary"):
                            corrections[(code, qi)] = text_val
                            st.session_state.flag_index += 1
                            st.rerun()
                    else:
                        n_options = len(g["option_elements"])
                        cols = st.columns(n_options + 2)
                        for oi in range(n_options):
                            if cols[oi].button(str(oi + 1), key=f"opt_{fidx}_{oi}", width='stretch'):
                                corrections[(code, qi)] = oi + 1
                                st.session_state.flag_index += 1
                                st.rerun()
                        if cols[n_options].button("NA", key=f"na_{fidx}", width='stretch'):
                            corrections[(code, qi)] = "NA"
                            st.session_state.flag_index += 1
                            st.rerun()
                        if cols[n_options + 1].button("MULT", key=f"mult_{fidx}", width='stretch'):
                            corrections[(code, qi)] = "MULT"
                            st.session_state.flag_index += 1
                            st.rerun()
                        if score is not None and not is_open_text:
                            diffs_str = ", ".join(f"{d:.0f}" for d in score.get("raw_diffs", []))
                            detected_str = (f"option {score['answer_index'] + 1}"
                                           if score.get("answer_index") is not None else "no option")
                            st.caption(f"Detected: {detected_str}  (reason: {score.get('reason') or 'confident'}, "
                                      f"confidence gap: {score.get('confidence_gap', 0):.0f})")
                            st.caption(f"Raw ink-diff scores per option, in order: [{diffs_str}]")

                if fidx > 0:
                    if st.button("\u2190 Back", key=f"flagback_{fidx}"):
                        st.session_state.flag_index -= 1
                        st.rerun()
            else:
                st.success(f"All flagged answers reviewed ({len(corrections)} corrected).")

            st.divider()
            dl_col1, dl_col2 = st.columns(2)
            with dl_col1:
                xlsx_bytes = build_export_xlsx(scored, template, corrections)
                st.download_button("Download Excel", xlsx_bytes, file_name="responses.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            with dl_col2:
                debug_xlsx_bytes = build_debug_report_xlsx(scored, template)
                st.download_button("Download debug report (Excel)", debug_xlsx_bytes,
                                   file_name="debug_report.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                   help="Every option box scored for every respondent -- exact "
                                        "pixel box, raw ink counts, and the config values active "
                                        "when generated. Available any time, doesn't require "
                                        "finishing review first.")


# ============================================================= GENERATE ===
if tab_generate:
    col_preview, col_controls = st.columns([1, 1], gap="medium")

    # ---- controls (right column) -- uploader lives HERE, inline with
    # the rest of the grid, not as a separate top-of-page element ----
    with col_controls:
        row1a, row1b = st.columns(2)
        with row1a:
            uploaded = st.file_uploader("Upload .pdf", type=["pdf"], label_visibility="visible")

        if uploaded is None:
            with row1b:
                st.caption("Content scale, marker position, and QR controls appear once a PDF is uploaded.")
            with col_preview:
                st.info("Upload a PDF to see the preview.")
            st.stop()

        pdf_bytes = uploaded.getvalue()
        src_doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        num_pages = len(src_doc)

        with row1b:
            content_scale_pct = st.number_input("Content scale (%)", min_value=0, max_value=100,
                                                  value=90, step=1)
            content_scale = content_scale_pct / 100.0

        row2a, row2b = st.columns(2)
        with row2a:
            left_x = st.slider("Marker left", 0, 100, 40)
        with row2b:
            right_x = st.slider("Marker right", 0, 100, 40)

        row3a, row3b = st.columns(2)
        with row3a:
            top_y = st.slider("Marker top", 0, 100, 40)
        with row3b:
            bottom_y = st.slider("Marker bottom", 0, 100, 45)

        row4a, row4b = st.columns(2)
        with row4a:
            marker_size = st.slider("Marker size", 20, 80, 30)
        with row4b:
            qr_size = st.slider("QR size", 20, 80, 35)

        page_w, page_h = src_doc[0].rect.width, src_doc[0].rect.height
        if "qr_position_choice" not in st.session_state:
            st.session_state.qr_position_choice = "bottom"
        row5a, row5b = st.columns(2)
        with row5a:
            if st.button("Top", key="qr_pos_top", width='stretch',
                         type="primary" if st.session_state.qr_position_choice == "top" else "secondary"):
                st.session_state.qr_position_choice = "top"
                st.rerun()
        with row5b:
            if st.button("Bottom", key="qr_pos_bottom", width='stretch',
                         type="primary" if st.session_state.qr_position_choice == "bottom" else "secondary"):
                st.session_state.qr_position_choice = "bottom"
                st.rerun()
        qr_position = st.session_state.qr_position_choice

        row6a, row6b = st.columns(2)
        with row6a:
            sample_code = st.text_input("Code", "667908")
        with row6b:
            codes_file = st.file_uploader("codes.csv", type=["csv"], key="codes_csv",
                                           label_visibility="collapsed")
            st.caption('Upload a list of unique tracking codes with a column named "tracking_code"')

        cfg = OverlayConfig(
            content_scale=content_scale,
            left_x=left_x, right_x=right_x, top_y=top_y, bottom_y=bottom_y,
            marker_size=marker_size,
            qr_position=qr_position, qr_size=qr_size,
        )

        row7a, row7b = st.columns(2)
        with row7a:
            generate_sample_clicked = st.button("Generate sample", width='stretch')
        with row7b:
            generate_batch_clicked = st.button("Generate batch", width='stretch',
                                                disabled=codes_file is None)

        # ---- actions -- results render HERE, directly under the buttons
        # that triggered them, instead of full-width below both columns ----
        if generate_sample_clicked:
            out_doc = pymupdf.open()
            for page_num in range(num_pages):
                compose_page(src_doc, page_num, cfg, sample_code, out_doc=out_doc)
            pdf_out_bytes = out_doc.tobytes()
            out_doc.close()
            st.download_button("Download sample PDF", pdf_out_bytes,
                                file_name=f"{sample_code}.pdf", mime="application/pdf")

        if generate_batch_clicked and codes_file is not None:
            text = codes_file.getvalue().decode("utf-8")
            reader = csv.DictReader(io.StringIO(text))
            if "tracking_code" not in (reader.fieldnames or []):
                st.error(f"codes.csv must have a 'tracking_code' column, found: {reader.fieldnames}")
            else:
                codes = [row["tracking_code"].strip() for row in reader if row["tracking_code"].strip()]
                seen = {}
                for c in codes:
                    seen[c] = seen.get(c, 0) + 1
                dupes = [c for c, n in seen.items() if n > 1]
                if not codes:
                    st.error("No codes found in codes.csv.")
                elif dupes:
                    st.error(f"Duplicate tracking codes found, aborting: {dupes}")
                else:
                    zip_buf = io.BytesIO()
                    progress = st.progress(0.0, text="Generating...")
                    with zipfile.ZipFile(zip_buf, "w") as zf:
                        for i, code in enumerate(codes):
                            out_doc = pymupdf.open()
                            for page_num in range(num_pages):
                                compose_page(src_doc, page_num, cfg, code, out_doc=out_doc)
                            pdf_out_bytes = out_doc.tobytes()
                            out_doc.close()
                            zf.writestr(f"{code}.pdf", pdf_out_bytes)
                            progress.progress((i + 1) / len(codes), text=f"Generated {code}.pdf")
                    progress.empty()
                    st.success(f"Generated {len(codes)} PDFs.")
                    st.download_button("Download all as .zip", zip_buf.getvalue(),
                                        file_name="generated_surveys.zip", mime="application/zip")

    if "preview_page" not in st.session_state:
        st.session_state.preview_page = 0
    st.session_state.preview_page = min(st.session_state.preview_page, num_pages - 1)

    # ---- preview + page navigator (left column) ----
    with col_preview:
        png_bytes = preview_page_png(src_doc, st.session_state.preview_page, cfg, sample_code)
        render_zoomable_preview(png_bytes)

        nav_prev, nav_label, nav_next = st.columns([1, 2, 1])
        with nav_prev:
            if st.button("←", width='stretch', disabled=st.session_state.preview_page == 0):
                st.session_state.preview_page -= 1
                st.rerun()
        with nav_label:
            st.markdown(f"<div style='text-align:center'>Page {st.session_state.preview_page + 1} of {num_pages}</div>",
                        unsafe_allow_html=True)
        with nav_next:
            if st.button("→", width='stretch',
                         disabled=st.session_state.preview_page >= num_pages - 1):
                st.session_state.preview_page += 1
                st.rerun()
