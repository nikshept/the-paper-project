"""
Stage 1 -- Extract.

Turns an uploaded .docx or .pdf into a flat list of positioned elements
(text words, table cells, embedded images, and detected drawn shapes),
all expressed in PDF-point coordinates. Stage 2 (Structure) consumes
this flat list and knows nothing about docx vs. pdf -- that distinction
stops existing after this module runs.

DESIGN PRINCIPLE -- read this before adding a new detector:
    This module extracts STRUCTURE, not MEANING. It does not decide
    that a particular word, shape, or image "is a response option" --
    that is a classification judgment that depends on CONTEXT (is it
    one of several similar, evenly-spaced things near a question?), and
    context is exactly what Stage 2's row-clustering step exists to
    evaluate. An earlier version of this module tried to pre-classify
    specific characters (circled digits, ballot-box glyphs, underscore
    runs) as "bubbles" directly in Stage 1. That was a mistake: it only
    covered the exact marker styles seen in testing and would silently
    do nothing on a document that used different characters, drawn
    shapes, or images instead. Stage 1 now stays deliberately dumb: any
    character is just a "word", any drawn circle/box is just a "shape",
    any picture is just an "image" -- all with positions, none with an
    assigned meaning. Whether five things in a row are a Likert scale,
    a set of checkboxes, or unrelated decoration is for Stage 2 to
    figure out from their arrangement, not for Stage 1 to guess from
    their identity.

REQUIREMENTS ON THE INPUT FILE (read this before debugging a "0 words
found" result):
    - A .docx must be a normal, text-based Word document -- i.e. text
      you could select and copy in Word.
    - A .pdf MUST be "machine-readable": a real, selectable text layer
      (e.g. exported from Word, or "Print to PDF"). A PHOTOGRAPHED OR
      SCANNED PDF WITH NO TEXT LAYER IS NOT SUPPORTED -- there is no OCR
      step in this module. A page with no extractable text is skipped
      and flagged in `warnings`, not guessed at.

KNOWN LIMITATION -- images embedded in a .docx:
    PyMuPDF's .docx renderer was tested and found to silently DROP
    embedded images entirely (confirmed: an image visible in Word did
    not appear in PyMuPDF's rendered page at all, and get_image_info()
    returned empty). So for .docx, image detection falls back to
    python-docx's `inline_shapes`, which reliably detects that an image
    EXISTS and its size, but -- because .docx has no fixed page layout
    at the XML level -- cannot give an exact page position, only a
    rough document-order index. This is recorded honestly (page=-1,
    like table cells) rather than guessed at. PDF image detection, by
    contrast, gives an exact bbox (PDFs are already fully laid out).

Dependencies: pymupdf, pdfplumber, python-docx, opencv-python-headless,
numpy. All pure pip wheels -- no LibreOffice, no Poppler, no Tesseract,
no other system-level install, on any platform.

Units: every bbox in the returned element list is in PDF points,
origin top-left of the page. Pixel-space detections (shapes) are
converted back to points using the actual page-vs-pixmap ratio (not an
assumed DPI constant), so it's exact regardless of render DPI chosen.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import cv2
import numpy as np
import pdfplumber
import pymupdf

try:
    import docx  # python-docx
except ImportError:  # pragma: no cover
    docx = None

RENDER_DPI = 200  # rasterization density for shape detection only


# ============================================================== TYPES =====

@dataclass
class Element:
    """One detected thing on a page, in a common shape regardless of
    source. Deliberately meaning-free: `type` describes WHAT KIND of
    thing this is structurally (a word, a table cell, a drawn shape, an
    image), never what it's FOR (Stage 2's job)."""
    type: str  # "word" | "table_cell" | "shape" | "image"
    page: int  # 0-indexed; -1 means "not positioned" (docx table cells, docx images)
    bbox: list  # [x0, y0, x1, y1] in PDF points, top-left origin, or None if unpositioned
    text: str = ""
    # table-only: which row/col this cell occupies within its table, and
    # a table index so cells from different tables aren't confused
    table_index: int = None
    row: int = None
    col: int = None
    # shape-only: detected geometry kind ("circle" | "rect") and radius
    # (circles only); see meta for pixel dimensions on rects
    radius: float = None
    meta: dict = field(default_factory=dict)


@dataclass
class ExtractResult:
    elements: list  # list[Element], the unified flat list Stage 2 consumes
    source_type: str  # "docx" | "pdf"
    page_sizes: list  # [(width_pt, height_pt), ...] per page
    warnings: list = field(default_factory=list)


# ============================================================ ENTRYPOINT ==

def extract(file_path: str) -> ExtractResult:
    """Dispatches on extension. Returns the same ExtractResult shape
    either way -- Stage 2 never needs to know which branch ran."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".docx":
        return _extract_docx(file_path)
    elif ext == ".pdf":
        return _extract_pdf(file_path)
    else:
        raise ValueError(f"Unsupported file type '{ext}' -- expected .docx or .pdf")


# ======================================================= SHARED (PyMuPDF) =

def _render_page_gray(pymupdf_page) -> tuple:
    """Rasterizes one page to a grayscale numpy array, and returns the
    exact pixel->point scale factors alongside it (computed from the
    page's own dimensions vs. the pixmap's, not assumed from DPI, so
    this is exact even if get_pixmap's actual output size doesn't match
    the requested DPI precisely)."""
    pix = pymupdf_page.get_pixmap(dpi=RENDER_DPI)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY) if pix.n >= 3 else arr[:, :, 0]
    scale_x = pymupdf_page.rect.width / pix.width
    scale_y = pymupdf_page.rect.height / pix.height
    return gray, scale_x, scale_y


def _detect_shapes(pymupdf_page, page_idx: int, words: list,
                    min_size_px=12, max_size_px=44) -> list:
    """Generic drawn-geometry detection: circles (Hough transform) AND
    small squares/rectangles (contour + polygon approximation), covering
    the two common drawn checkbox/bubble styles without assuming which
    one a given survey uses. Both kinds are returned as type="shape"
    with meta["kind"] set -- Stage 1 does not decide these ARE response
    controls (vs. decorative borders, table cells, a logo outline); it
    only reports "a circle/box of roughly this size was drawn here".

    min/max size bounds are wide enough for typical Likert/checkbox
    sizes without also picking up large elements like table cell
    borders or small ones like punctuation; a survey with unusually
    large or tiny markers may need these tuned, hence they're
    parameters, not buried constants.

    `words` filters out shapes that are actually GLYPHS -- a letter
    'O' looks like a circle to HoughCircles, some fonts' 'D' or digits
    can approximate a rectangle to contour detection. A real drawn
    shape is visually blank inside; one whose area is mostly covered by
    a text bbox is almost certainly a character. Verified this matters
    for circles in testing; applied the same filter to rects since the
    failure mode is identical in kind."""
    gray, scale_x, scale_y = _render_page_gray(pymupdf_page)
    blurred = cv2.medianBlur(gray, 5)
    shapes = []

    # --- circles ---
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=min_size_px,
        param1=50, param2=30, minRadius=min_size_px // 2, maxRadius=max_size_px // 2,
    )
    if circles is not None:
        for x, y, r in circles[0]:
            x, y, r = float(x), float(y), float(r)  # numpy float32 -> plain float, JSON-safe downstream
            bbox = [(x - r) * scale_x, (y - r) * scale_y, (x + r) * scale_x, (y + r) * scale_y]
            shapes.append(Element(type="shape", page=page_idx, bbox=bbox,
                                   radius=r * scale_x, meta={"kind": "circle"}))

    # --- squares / rectangles (e.g. drawn checkbox outlines) ---
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
        if len(approx) != 4:
            continue
        x, y, w, h = cv2.boundingRect(approx)
        if not (min_size_px <= w <= max_size_px and min_size_px <= h <= max_size_px):
            continue
        aspect = w / h if h else 0
        if not (0.7 <= aspect <= 1.4):  # roughly square, not an arbitrary rectangle (e.g. a table cell)
            continue
        bbox = [x * scale_x, y * scale_y, (x + w) * scale_x, (y + h) * scale_y]
        shapes.append(Element(type="shape", page=page_idx, bbox=bbox,
                               meta={"kind": "rect", "width_px": w, "height_px": h}))

    return _filter_glyph_false_positives(shapes, words)


def _filter_glyph_false_positives(shapes: list, words: list, overlap_thresh=0.5) -> list:
    """Drops any detected shape whose area is mostly covered by a
    text-layer word box -- see _detect_shapes docstring."""
    kept = []
    for s in shapes:
        sx0, sy0, sx1, sy1 = s.bbox
        s_area = (sx1 - sx0) * (sy1 - sy0)
        is_glyph = False
        for w in words:
            if w.bbox is None:
                continue
            wx0, wy0, wx1, wy1 = w.bbox
            ix0, iy0 = max(sx0, wx0), max(sy0, wy0)
            ix1, iy1 = min(sx1, wx1), min(sy1, wy1)
            if ix1 > ix0 and iy1 > iy0:
                overlap = (ix1 - ix0) * (iy1 - iy0)
                if s_area > 0 and overlap / s_area >= overlap_thresh:
                    is_glyph = True
                    break
        if not is_glyph:
            kept.append(s)
    return kept


def _detect_images_pdf(pymupdf_page, page_idx: int) -> list:
    """PDF images have an exact, reliable bbox via PyMuPDF -- PDFs are
    already fully laid out, no reflow ambiguity."""
    found = []
    for info in pymupdf_page.get_image_info(xrefs=True):
        x0, y0, x1, y1 = info["bbox"]
        found.append(Element(type="image", page=page_idx, bbox=[x0, y0, x1, y1],
                              meta={"xref": info.get("xref"),
                                    "width_px": info.get("width"), "height_px": info.get("height")}))
    return found


# =========================================================== DOCX PATH ====

def _extract_docx(docx_path: str) -> ExtractResult:
    if docx is None:
        raise RuntimeError("python-docx is not installed (`pip install python-docx`)")

    warnings = []
    all_elements = []
    page_sizes = []

    # PyMuPDF opens the .docx directly -- no conversion step of any kind.
    pmdoc = pymupdf.open(docx_path)
    for page_idx, page in enumerate(pmdoc):
        page_sizes.append((page.rect.width, page.rect.height))
        page_words = []
        for x0, y0, x1, y1, text, *_ in page.get_text("words"):
            el = Element(type="word", page=page_idx, text=text, bbox=[x0, y0, x1, y1])
            all_elements.append(el)
            page_words.append(el)
        all_elements.extend(_detect_shapes(page, page_idx, page_words))
    pmdoc.close()

    if not all_elements:
        warnings.append(
            "No text or shapes were found via PyMuPDF's rendering of this "
            ".docx. This is unusual for a normal Word document -- worth "
            "double-checking the file actually opens correctly in Word."
        )

    # Ground-truth table structure straight from the docx XML via
    # python-docx -- exact row/col/text, more reliable than inferring
    # table structure from a rendered page (checked: PyMuPDF's own table
    # detection missed a real docx table with no visible gridlines).
    all_elements.extend(_extract_docx_tables(docx_path))

    # Images: see the module docstring's "KNOWN LIMITATION" section --
    # PyMuPDF drops embedded images when rendering .docx, so this uses
    # python-docx instead. No page position is available this way, only
    # existence + size + document order; flagged honestly via page=-1
    # and a document_order index, not silently omitted.
    all_elements.extend(_extract_docx_images(docx_path))

    return ExtractResult(elements=all_elements, source_type="docx",
                          page_sizes=page_sizes, warnings=warnings)


def _extract_docx_tables(docx_path: str) -> list:
    """Walks every table in document order and returns each cell as an
    Element with row/col/table_index but no bbox. Generic: makes no
    assumption about how many tables exist, what their columns mean, or
    how many rows they have -- that interpretation is Stage 2's job."""
    d = docx.Document(docx_path)
    elements = []
    for t_idx, table in enumerate(d.tables):
        for r_idx, row in enumerate(table.rows):
            for c_idx, cell in enumerate(row.cells):
                text = cell.text.strip()
                if not text:
                    continue
                elements.append(Element(
                    type="table_cell", page=-1, bbox=None, text=text,
                    table_index=t_idx, row=r_idx, col=c_idx,
                ))
    return elements


def _extract_docx_images(docx_path: str) -> list:
    """See module docstring -- no page bbox available for docx images,
    only existence, size, and a document-order index."""
    d = docx.Document(docx_path)
    elements = []
    for i, shape in enumerate(d.inline_shapes):
        elements.append(Element(
            type="image", page=-1, bbox=None,
            meta={"document_order_index": i,
                  "width_emu": shape.width, "height_emu": shape.height,
                  "position_known": False},
        ))
    return elements


# ============================================================ PDF PATH ====

def _extract_pdf(pdf_path: str) -> ExtractResult:
    warnings = []
    all_elements = []
    page_sizes = []

    pmdoc = pymupdf.open(pdf_path)  # used for rasterizing (shapes) and image bboxes

    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, (page, pm_page) in enumerate(zip(pdf.pages, pmdoc)):
            page_sizes.append((page.width, page.height))

            # x_tolerance tightened from pdfplumber's default (3) to 1: some
            # PDF generators/fonts have tight enough inter-word spacing that
            # the default treats a real word boundary as within-word,
            # silently fusing two words into one token before any code
            # downstream ever sees them ("markhowoften..." instead of "mark
            # how often..." -- confirmed as a real, recurring symptom, not
            # a joining-logic bug on this project's side). A tighter value
            # only ever splits MORE readily; confirmed no regression on
            # existing fixtures at this setting.
            words = page.extract_words(use_text_flow=False, keep_blank_chars=False, x_tolerance=1)
            if not words:
                warnings.append(
                    f"Page {page_idx}: no extractable text layer found. This "
                    f"page appears to be a scan/photo rather than a "
                    f"machine-readable PDF, which this module does not "
                    f"support (no OCR step). Skipping this page -- results "
                    f"from it will be incomplete."
                )
                continue

            page_word_elements = []
            for w in words:
                el = Element(type="word", page=page_idx, text=w["text"],
                             bbox=[w["x0"], w["top"], w["x1"], w["bottom"]])
                all_elements.append(el)
                page_word_elements.append(el)

            all_elements.extend(_detect_shapes(pm_page, page_idx, page_word_elements))
            all_elements.extend(_detect_images_pdf(pm_page, page_idx))

    pmdoc.close()
    return ExtractResult(elements=all_elements, source_type="pdf",
                          page_sizes=page_sizes, warnings=warnings)


# =============================================================== DEBUG ====

def summarize(result: ExtractResult) -> str:
    """Quick human-readable sanity check -- counts by type/page, shape
    kinds, and any warnings raised. Not used by Stage 2; just useful
    while testing this module against a real file."""
    lines = [f"source_type={result.source_type}  pages={len(result.page_sizes)}"]
    by_page = {}
    shape_kinds = {}
    for el in result.elements:
        by_page.setdefault(el.page, {}).setdefault(el.type, 0)
        by_page[el.page][el.type] += 1
        if el.type == "shape":
            kind = el.meta.get("kind", "?")
            shape_kinds[kind] = shape_kinds.get(kind, 0) + 1
    for page in sorted(by_page, key=lambda p: (p is None, p)):
        counts = ", ".join(f"{t}={n}" for t, n in by_page[page].items())
        lines.append(f"  page {page}: {counts}")
    if shape_kinds:
        lines.append("shape breakdown: " + ", ".join(f"{k}={n}" for k, n in shape_kinds.items()))
    if result.warnings:
        lines.append("warnings:")
        lines.extend(f"  - {w}" for w in result.warnings)
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python extract.py <file.docx|file.pdf>")
        sys.exit(1)
    res = extract(sys.argv[1])
    print(summarize(res))
