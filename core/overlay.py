"""
Overlay -- takes an already-finished PDF (the researcher's real content,
untouched) and produces a new PDF with: the original content uniformly
scaled (independent of the marker margins -- see _content_target_rect),
plus corner markers and a per-page QR positioned using those margins.

Why a NEW page is composed rather than drawing onto the original: the
"scale the whole content" requirement can't be done by drawing more on
top of an existing page -- the existing content has to be re-embedded
as a scaled object. PyMuPDF's Page.show_pdf_page() does exactly this:
it places another PDF page's content, as a vector object, scaled to fit
a target rectangle -- confirmed in testing to scale ALL of a page's
content together as one unit (text position, size, everything moves
together), not per-element, which is exactly what was asked for.

QR data format: "{tracking_code}-P{page_num}" -- distinct per page.
"""
from __future__ import annotations

import io
import os

import pymupdf
import qrcode

from core.overlay_config import OverlayConfig, resolve_qr_position


# ============================================================== GEOMETRY ==

def _corner_rects(cfg: OverlayConfig, page_w: float, page_h: float) -> dict:
    """Four corner marker rects, each using its OWN margin (left_x,
    right_x, top_y, bottom_y independently) -- not a single shared
    inset. PyMuPDF coordinates: origin top-left, y increases downward."""
    M = cfg.marker_size
    return {
        0: pymupdf.Rect(cfg.left_x, cfg.top_y, cfg.left_x + M, cfg.top_y + M),
        1: pymupdf.Rect(page_w - cfg.right_x - M, cfg.top_y, page_w - cfg.right_x, cfg.top_y + M),
        2: pymupdf.Rect(page_w - cfg.right_x - M, page_h - cfg.bottom_y - M, page_w - cfg.right_x, page_h - cfg.bottom_y),
        3: pymupdf.Rect(cfg.left_x, page_h - cfg.bottom_y - M, cfg.left_x + M, page_h - cfg.bottom_y),
    }


def _content_target_rect(cfg: OverlayConfig, page_w: float, page_h: float) -> pymupdf.Rect:
    """Where the original content gets scaled into -- deliberately
    INDEPENDENT of the marker margins now. An earlier version scaled
    content into the margin-inset "safe area", which meant adjusting a
    margin also moved/resized the content, even though the person
    adjusting margins usually just wants to reposition the markers.
    Content size/position is controlled by content_scale alone: it
    scales the full page by that fraction, centered on the page's own
    center. Margins and content can now be set independently -- markers
    may end up overlapping content if the two aren't set sensibly
    together, which is now the person's call to make visually via the
    live preview, not something this function silently prevents."""
    target_w = page_w * cfg.content_scale
    target_h = page_h * cfg.content_scale
    x0 = (page_w - target_w) / 2
    y0 = (page_h - target_h) / 2
    return pymupdf.Rect(x0, y0, x0 + target_w, y0 + target_h)


# ================================================================ DRAWING =

def markers_ready(cfg: OverlayConfig) -> bool:
    """True only if all four ArUco marker image files actually exist on
    disk. Checked upfront (not via try/except on insert) so the
    fallback is all-or-nothing -- a survey should never end up with
    three real ArUco corners and one drawn bracket, which would look
    like a bug rather than an intentional choice."""
    if not cfg.marker_image_paths:
        return False
    return all(os.path.exists(cfg.marker_image_paths.get(i, "")) for i in range(4))


def _draw_corner_brackets(page: pymupdf.Page, cfg: OverlayConfig):
    """Simple L-shaped corner brackets, drawn as vector lines -- no
    image assets needed."""
    W, H = page.rect.width, page.rect.height
    arm = cfg.bracket_arm_length
    corners = [
        (cfg.left_x, cfg.top_y, 1, 1),                      # top-left
        (W - cfg.right_x, cfg.top_y, -1, 1),                # top-right
        (cfg.left_x, H - cfg.bottom_y, 1, -1),               # bottom-left
        (W - cfg.right_x, H - cfg.bottom_y, -1, -1),         # bottom-right
    ]
    for x, y, dx, dy in corners:
        page.draw_line((x, y), (x + dx * arm, y), width=1)
        page.draw_line((x, y), (x, y + dy * arm), width=1)


def _draw_corner_images(page: pymupdf.Page, cfg: OverlayConfig):
    """Real marker images (e.g. ArUco), for when detection needs them."""
    rects = _corner_rects(cfg, page.rect.width, page.rect.height)
    for idx, rect in rects.items():
        path = cfg.marker_image_paths.get(idx)
        if path:
            page.insert_image(rect, filename=path)


def _make_qr_png_bytes(data: str) -> bytes:
    qr = qrcode.QRCode(border=1, box_size=10)
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ============================================================== COMPOSING =

def compose_page(src_doc: pymupdf.Document, page_num: int, cfg: OverlayConfig,
                  tracking_code: str, out_doc: pymupdf.Document = None) -> pymupdf.Document:
    """Builds ONE output page: the source page's content scaled into the
    margin-defined safe area, plus corner markers and this page's QR.
    Returns the containing document (a new single-page one, unless
    `out_doc` is given -- in which case the page is appended to it, for
    building a multi-page document one page at a time)."""
    src_page = src_doc[page_num]
    page_w, page_h = src_page.rect.width, src_page.rect.height

    if out_doc is None:
        out_doc = pymupdf.open()
    new_page = out_doc.new_page(width=page_w, height=page_h)

    target_rect = _content_target_rect(cfg, page_w, page_h)
    new_page.show_pdf_page(target_rect, src_doc, page_num, keep_proportion=True)

    if markers_ready(cfg):
        _draw_corner_images(new_page, cfg)
    else:
        _draw_corner_brackets(new_page, cfg)

    qr_data = f"{tracking_code}-P{page_num}"
    qr_bytes = _make_qr_png_bytes(qr_data)
    qr_x, qr_y = resolve_qr_position(cfg, page_w, page_h)
    qr_rect = pymupdf.Rect(qr_x, qr_y, qr_x + cfg.qr_size, qr_y + cfg.qr_size)
    new_page.insert_image(qr_rect, stream=qr_bytes)

    if cfg.show_tracking_code_text:
        text_point = pymupdf.Point(
            qr_rect.x1 + cfg.qr_text_gap,
            qr_rect.y0 + qr_rect.height / 2 + cfg.qr_text_size / 3,
        )
        new_page.insert_text(text_point, str(tracking_code), fontsize=cfg.qr_text_size)

    return out_doc


def preview_page_png(src_doc: pymupdf.Document, page_num: int, cfg: OverlayConfig,
                      tracking_code: str, dpi: int = 150) -> bytes:
    """Renders just one composed page to PNG bytes -- for a fast,
    interactive live preview (e.g. Streamlit sliders) without writing a
    full multi-page PDF to disk on every adjustment."""
    tmp_doc = compose_page(src_doc, page_num, cfg, tracking_code)
    pix = tmp_doc[0].get_pixmap(dpi=dpi)
    png_bytes = pix.tobytes("png")
    tmp_doc.close()
    return png_bytes


def stamp_pdf(input_path: str, output_path: str, cfg: OverlayConfig, tracking_code: str):
    """Composes every page of `input_path` (scaled content + markers +
    per-page QR) into a fresh document, saved to `output_path`. The
    input file is opened read-only and never modified."""
    src_doc = pymupdf.open(input_path)
    out_doc = pymupdf.open()
    for page_num in range(len(src_doc)):
        compose_page(src_doc, page_num, cfg, tracking_code, out_doc=out_doc)
    out_doc.save(output_path)
    out_doc.close()
    src_doc.close()


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print("Usage: python overlay.py <input.pdf> <output.pdf> <tracking_code>")
        sys.exit(1)
    stamp_pdf(sys.argv[1], sys.argv[2], OverlayConfig(), sys.argv[3])
    print(f"Wrote {sys.argv[2]}")