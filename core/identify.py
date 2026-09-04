"""
Identify stage -- wraps dewarp.py (unmodified, from p1) to work against
our actual inputs: a blank reference PDF (the stamped sample the
researcher already generated) and a "photos PDF" (one photographed
survey page per PDF page, in order, per the upload note).

dewarp.py's public functions expect real file paths (it calls
cv2.imread / pymupdf.open on them directly), so this module bridges via
temp files rather than editing dewarp.py itself -- keeps p1's tested
code byte-for-byte as-is, per the "don't reinvent the wheel" direction.
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field

import cv2
import numpy as np
import pymupdf

from core.dewarp import (get_true_marker_positions, _try_dewarp, _decode_qr,
                          _CV2_QR, _get_wechat_detector, _HAVE_ZBAR, SCALE, WECHAT_MODEL_DIR)

try:
    from pyzbar.pyzbar import decode as zbar_decode
except (ImportError, OSError):
    zbar_decode = None  # matches dewarp.py's own handling: pyzbar's native
                          # libzbar DLL can fail to load on some Windows
                          # setups even when the Python package itself is
                          # installed -- that's an OSError, not ImportError,
                          # so both must be caught or the whole app fails
                          # to start (exactly what happened here).


@dataclass
class IdentifyResult:
    photo_index: int  # 0-based index of the page within the uploaded photos PDF
    dewarped: bool = False    # markers found, geometry recovered
    identified: bool = False  # dewarped AND the QR decoded successfully
    tracking_code: str = None
    page_num: int = None
    orientation_fix: str = None  # which flip/rotation, if any, was needed
    dewarped_image: "cv2.Mat" = None  # best available flattened image, even if QR failed
    error: str = None  # only set if dewarped is False -- no markers found in any orientation
    native_resolution: tuple = None  # (width_px, height_px) of the SOURCE embedded image, not
                                       # the re-render -- the true ceiling on detail available.
                                       # Always populated, success or failure, since a low value
                                       # here is a generic, self-diagnosable red flag: no amount
                                       # of detector tuning recovers detail the source never had.
    matched_page_guess: int = None  # which blank-reference page's marker layout matched, even
                                      # when QR decode failed -- needed to know which QR
                                      # position-hint applies, for building a diagnostic crop
    original_image: "cv2.Mat" = None  # the raw, undewarped photo as read -- always populated
                                        # when the file itself could be read, even if marker
                                        # detection found nothing at all, so there's ALWAYS
                                        # something to show the user for review, regardless of
                                        # how badly detection failed


def get_qr_position_hint(blank_pdf_path: str) -> dict:
    """Detects the QR's ACTUAL location in each page of the rendered
    blank reference, as fractions of page width/height -- rather than
    assuming a fixed position. p1's dewarp.py hardcodes a bottom-center
    crop assumption (qr_center_frac=(0.5, 0.965)), which was correct for
    p1's own survey but not for ours: our OverlayConfig lets a
    researcher choose "top" OR "bottom" QR placement, something p1
    never had. If "top" was used, that hardcoded crop always misses,
    silently falling back to full-page detection every time -- which
    still works sometimes, but throws away the robustness benefit
    cropping was meant to provide in the first place (a QR is much
    smaller, proportionally, in a full-page scan than in a tight crop,
    which measurably hurts detection on real-world blur/compression).

    Returns {page_num: (cx_frac, cy_frac, size_frac)} or {} if the
    blank reference's own QR couldn't be located (falls back to p1's
    fixed assumption in that case, no worse than before)."""
    doc = pymupdf.open(blank_pdf_path)
    hints = {}
    for pno, page in enumerate(doc):
        pix = page.get_pixmap(dpi=200)
        arr = cv2.imdecode(np.frombuffer(pix.tobytes("png"), dtype=np.uint8), cv2.IMREAD_COLOR)
        data, points, _ = _CV2_QR.detectAndDecode(arr)
        if points is None or len(points) == 0:
            continue
        pts = points.reshape(-1, 2)
        h, w = arr.shape[:2]
        cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
        size = max(pts[:, 0].max() - pts[:, 0].min(), pts[:, 1].max() - pts[:, 1].min())
        hints[pno] = (cx / w, cy / h, (size / w) * 1.8)  # 1.8x margin around the QR itself
    doc.close()
    return hints


def _decode_qr_with_hint(flat_img, hint: tuple):
    """Tries a crop around the LEARNED QR position first (same 3-detector
    cascade p1's _decode_qr uses: cv2 built-in, WeChat, pyzbar), then
    falls through to dewarp.py's _decode_qr as-is (its own bottom-center
    crop + full-page) -- this only ADDS attempts, never removes one of
    p1's existing fallback paths."""
    if hint is not None:
        cx_frac, cy_frac, size_frac = hint
        h, w = flat_img.shape[:2]
        cx, cy = int(w * cx_frac), int(h * cy_frac)
        half = int(w * size_frac / 2)
        y0, y1 = max(0, cy - half), min(h, cy + half)
        x0, x1 = max(0, cx - half), min(w, cx + half)
        crop = flat_img[y0:y1, x0:x1]

        if crop.size > 0:
            data, _, _ = _CV2_QR.detectAndDecode(crop)
            if data:
                return data
            wechat = _get_wechat_detector()
            if wechat is not None:
                results, _ = wechat.detectAndDecode(crop)
                if results:
                    return results[0]
            if _HAVE_ZBAR and zbar_decode is not None:
                results = zbar_decode(crop)
                if results:
                    return results[0].data.decode("utf-8")

            # CLAHE contrast enhancement as a last attempt on this crop --
            # verified this genuinely recovers cases raw decode misses
            # (tested on a generic low-contrast/blurred QR, not tuned to
            # any specific photo): faint printing, uneven lighting, or a
            # slight shadow across the QR specifically (even when the
            # rest of the photo is sharp) all reduce local contrast in
            # exactly the way CLAHE corrects for.
            gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray_crop)
            data, _, _ = _CV2_QR.detectAndDecode(enhanced)
            if data:
                return data

    return _decode_qr(flat_img)  # p1's existing bottom-center-crop + full-page fallback


def diagnose_qr_decoders() -> dict:
    """Explicit visibility into which QR decoders are ACTUALLY available at
    runtime -- dewarp.py's own _get_wechat_detector() silently swallows
    every exception (`except Exception: pass`), which is reasonable for
    not crashing the pipeline but means there's normally zero way to
    tell "WeChat is working" from "WeChat silently failed and we've been
    relying on the weaker plain-cv2 detector this whole time" -- exactly
    the distinction that matters here. This re-attempts the same two
    construction paths dewarp.py tries, but captures and returns the
    actual failure reason for each instead of discarding it."""
    result = {
        "cv2_builtin": True,  # cv2.QRCodeDetector() always available if cv2 imported at all
        "wechat_zeroarg": None,  # None = not tried/unknown, True/False = result
        "wechat_zeroarg_error": None,
        "wechat_model_dir": WECHAT_MODEL_DIR,  # exact path being checked, for transparency
        "wechat_files_present": all(
            os.path.exists(os.path.join(WECHAT_MODEL_DIR, f))
            for f in ("detect.prototxt", "detect.caffemodel", "sr.prototxt", "sr.caffemodel")
        ),
        "pyzbar": _HAVE_ZBAR and zbar_decode is not None,
        "wechat_active": _get_wechat_detector() is not None,  # the actual value dewarp.py will use
    }
    try:
        cv2.wechat_qrcode_WeChatQRCode()
        result["wechat_zeroarg"] = True
    except Exception as e:
        result["wechat_zeroarg"] = False
        result["wechat_zeroarg_error"] = f"{type(e).__name__}: {e}"

    return result


def _write_bytes_to_temp(data: bytes, suffix: str) -> str:
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        return tmp.name


def _get_native_image_resolution(page) -> tuple:
    """The SOURCE embedded image's real pixel dimensions -- independent
    of photo_render_dpi. Re-rendering a low-resolution source at a
    higher DPI only interpolates/upscales; it can't recover detail the
    source never had. Reports the largest embedded image on the page
    (in case of any incidental extras), so a genuinely low native
    resolution shows up as an objective, self-diagnosable number rather
    than needing to inspect the actual photo content."""
    infos = page.get_image_info(xrefs=True)
    if not infos:
        return None
    biggest = max(infos, key=lambda info: info.get("width", 0) * info.get("height", 0))
    return (biggest.get("width"), biggest.get("height"))


def _identify_one_photo(photo_path: str, truth: dict, page_w: float, page_h: float,
                         photo_index: int, native_resolution: tuple = None,
                         qr_hints: dict = None) -> IdentifyResult:
    """Re-implements dewarp_and_identify's orientation/page search loop
    (rather than calling it directly) so dewarp-succeeded-but-QR-failed
    can be reported as its own state -- dewarp_and_identify treats that
    as a single all-or-nothing failure, which loses a genuinely useful
    distinction: a photo with good geometry but an unreadable QR is a
    "flag for manual code entry" case, not a "photo unusable" case.
    Every actual detection/decode call is still the same p1 code
    (_try_dewarp, _decode_qr) -- only this orchestration is new."""
    original = cv2.imread(photo_path)
    if original is None:
        return IdentifyResult(photo_index=photo_index, error=f"Could not read image: {photo_path}",
                               native_resolution=native_resolution)

    candidates = {
        "as-is": original,
        "horizontal-flip": cv2.flip(original, 1),
        "vertical-flip": cv2.flip(original, 0),
        "180-rotation": cv2.flip(original, -1),
    }

    best_flat = None
    best_page_guess = None
    dewarped = False

    for label, candidate in candidates.items():
        for page_num_guess, truth_for_page in truth.items():
            flat = _try_dewarp(candidate, truth_for_page, page_w, page_h)
            if flat is None:
                continue
            dewarped = True
            if best_flat is None:
                best_flat = flat  # keep the first successful dewarp even if QR never decodes
                best_page_guess = page_num_guess

            hint = qr_hints.get(page_num_guess) if qr_hints else None
            qr_data = _decode_qr_with_hint(flat, hint)
            if qr_data:
                tracking_code = qr_data.rsplit("-P", 1)[0]  # matches our QR format:
                                                              # "{tracking_code}-P{page_num}"
                page_num = (int(qr_data.rsplit("-P", 1)[-1])
                            if "-P" in qr_data else page_num_guess)
                return IdentifyResult(
                    photo_index=photo_index, dewarped=True, identified=True,
                    tracking_code=tracking_code, page_num=page_num,
                    orientation_fix=label, dewarped_image=flat,
                    native_resolution=native_resolution, original_image=original,
                )

    if dewarped:
        return IdentifyResult(photo_index=photo_index, dewarped=True, identified=False,
                               dewarped_image=best_flat, native_resolution=native_resolution,
                               matched_page_guess=best_page_guess, original_image=original,
                               error="Markers found and geometry recovered, but the QR code "
                                     "could not be read on any orientation.")
    return IdentifyResult(photo_index=photo_index, dewarped=False, identified=False,
                           native_resolution=native_resolution, original_image=original,
                           error="No 4-marker set found in any orientation -- "
                                 "check the photo actually shows all four corners.")


def _compute_safe_render_dpi(page, native_resolution, fallback_dpi: int,
                              min_dpi: int = 100, max_dpi: int = 600) -> float:
    """DPI computed from the PHOTO PAGE's own point-size vs. its actual
    embedded image's pixel size -- not a fixed constant. A fixed DPI
    blindly multiplies whatever the page's point-dimensions happen to
    be; if those are wrong (e.g. a PDF built by some tool that set the
    page size using pixel counts as if they were points -- a common
    bug in ad-hoc image-to-PDF conversion), a fixed DPI can request an
    enormously oversized render and crash MuPDF outright ("Overly large
    image"). Computing DPI from native-resolution-vs-page-size instead
    is self-correcting: whatever caused the page to be oddly sized, this
    targets rendering at roughly the image's own real detail level, no
    more, no less -- clamped to a sane range either direction."""
    if not native_resolution or not native_resolution[0] or not native_resolution[1]:
        return int(fallback_dpi)  # couldn't determine native resolution at all -- fall back
    page_w_pt, page_h_pt = page.rect.width, page.rect.height
    if page_w_pt <= 0 or page_h_pt <= 0:
        return int(fallback_dpi)
    dpi_w = 72.0 * native_resolution[0] / page_w_pt
    dpi_h = 72.0 * native_resolution[1] / page_h_pt
    dpi = min(dpi_w, dpi_h)  # the more conservative of the two -- never request more
                              # detail than the source actually has in either dimension
    return int(round(max(min_dpi, min(max_dpi, dpi))))


def identify_photos(blank_pdf_bytes: bytes, photos_pdf_bytes: bytes,
                     photo_render_dpi: int = 400, on_progress=None) -> list:
    """Runs Identify on every page of `photos_pdf_bytes` against the
    marker positions found in `blank_pdf_bytes`. Returns one
    IdentifyResult per photo page, in order -- one bad photo never loses
    the rest of the batch.

    photo_render_dpi is now a FALLBACK/cap, not a fixed value blindly
    applied to every page -- see _compute_safe_render_dpi.

    on_progress(i, n, result), if given, is called after each photo --
    each photo genuinely can take a few seconds (ArUco detection across
    4 candidate orientations, on a real camera-resolution image, isn't
    instant), and a single opaque spinner for the whole batch gives no
    way to tell "still working, slowly" from "actually stuck". This
    lets the caller show real per-photo progress instead."""
    blank_path = _write_bytes_to_temp(blank_pdf_bytes, ".pdf")
    try:
        truth, page_w, page_h = get_true_marker_positions(blank_path)
        qr_hints = get_qr_position_hint(blank_path)  # learned from the SAME blank reference,
                                                        # not assumed -- see get_qr_position_hint
    finally:
        os.remove(blank_path)

    photos_doc = pymupdf.open(stream=photos_pdf_bytes, filetype="pdf")
    n = len(photos_doc)
    results = []

    for i, page in enumerate(photos_doc):
        native_res = _get_native_image_resolution(page)
        safe_dpi = _compute_safe_render_dpi(page, native_res, fallback_dpi=photo_render_dpi,
                                             max_dpi=photo_render_dpi)
        pix = page.get_pixmap(dpi=safe_dpi)
        img_bytes = pix.tobytes("png")
        photo_path = _write_bytes_to_temp(img_bytes, ".png")
        try:
            result = _identify_one_photo(photo_path, truth, page_w, page_h, i,
                                          native_resolution=native_res, qr_hints=qr_hints)
            results.append(result)
        finally:
            os.remove(photo_path)
        if on_progress is not None:
            on_progress(i, n, results[-1])

    photos_doc.close()
    return results


def summarize(results: list) -> str:
    n = len(results)
    n_dewarped = sum(1 for r in results if r.dewarped)
    n_identified = sum(1 for r in results if r.identified)
    lines = [f"{n_dewarped} of {n} photos dewarped successfully",
             f"{n_identified} of {n} photos identified successfully"]
    for r in results:
        res_str = f"{r.native_resolution[0]}x{r.native_resolution[1]}" if r.native_resolution else "?"
        if r.identified:
            lines.append(f"  photo {r.photo_index}: OK  code={r.tracking_code}  "
                         f"page={r.page_num}  orientation={r.orientation_fix}  native_res={res_str}")
        else:
            lines.append(f"  photo {r.photo_index}: FAILED  ({r.error})  native_res={res_str}")
    return "\n".join(lines)