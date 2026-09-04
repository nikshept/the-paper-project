"""
Score -- reads marked answers off an already-dewarped photo, using
whichever template the researcher confirmed in the "Get Template" tab.

The actual technique (binarized ink-diff against a blank reference) is
p1's own detect.py, unmodified in substance -- only the item/option
source changed: p1 read from a hardcoded ITEMS_PAGE0 list against a
hand-built boxes.json; this reads from a QuestionGuess's option_elements
(auto-guessed, researcher-confirmed), which are just bboxes in PDF-point
space, same coordinate system p1's boxes.json used. The scoring math
itself -- adaptive-threshold binarization first (so a photographed page
under real lighting compares fairly against a pixel-perfect blank
render, rather than reading as uniformly "darker" everywhere), then
diffing ink-pixel counts per option -- is exactly p1's, including the
calibrated thresholds.
"""
from __future__ import annotations

import io

import cv2
import numpy as np
import pymupdf

from core.dewarp import SCALE

# Same thresholds p1 calibrated against real photos (see p1's detect.py):
# genuinely blank items showed top scores of -34 to -10 (noise), genuinely
# marked items showed 30-180+. Starting point, not a large validated set.
MIN_INK_THRESHOLD = 15
MIN_CONFIDENCE_GAP = 15


def _binarize(gray):
    """Adaptive threshold: locally-normalized binarization, robust to
    the lighting/shadow variation a real photo has and a digital render
    doesn't. Identical to p1's detect.py."""
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, blockSize=41, C=10)


def build_blank_references(blank_pdf_bytes: bytes) -> dict:
    """Renders every page of the blank reference at the SAME pixel
    scale (SCALE, from dewarp.py) the dewarped photos already use, so a
    template bbox in PDF points maps to the same pixel location in
    both, no separate alignment step needed. Returns {page_num:
    binarized_image}."""
    doc = pymupdf.open(stream=blank_pdf_bytes, filetype="pdf")
    refs = {}
    for pno, page in enumerate(doc):
        pix = page.get_pixmap(matrix=pymupdf.Matrix(SCALE, SCALE))
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if pix.n >= 3 else img
        refs[pno] = _binarize(gray)
    doc.close()
    return refs


def binarize_dewarped(dewarped_bgr) -> "np.ndarray":
    gray = cv2.cvtColor(dewarped_bgr, cv2.COLOR_BGR2GRAY)
    return _binarize(gray)


def _bbox_to_px(bbox_pts: list) -> tuple:
    x0, y0, x1, y1 = bbox_pts
    return int(x0 * SCALE), int(y0 * SCALE), int(x1 * SCALE), int(y1 * SCALE)


def _ink_score(bin_img, bbox_px: tuple) -> float:
    """Samples ink from EXACTLY the detected box -- no expansion, no
    inset. Went through two other approaches first, both wrong in
    opposite directions, and it's worth recording why neither worked:

    1. Originally expanded by 4px outward. Caused phantom high scores
       on genuinely empty circles: bboxes near a table border/column
       divider (most often the leftmost option in a Likert row) picked
       up the printed divider line itself as false "ink" whenever a
       photo's dewarp was even slightly misaligned there.

    2. Tried insetting 15% toward the center instead, to avoid #1.
       Made things WORSE in practice on the real survey (confirmed:
       flagged count nearly doubled, 53->92, with raw scores showing
       uniformly elevated noise across every option on real photos).
       p1's actual bubble style prints a digit INSIDE each circle
       (see dewarp.py); insetting toward the center concentrates
       sampling on exactly that digit, and any dewarp misalignment in
       the digit's own printed edges becomes a much larger fraction of
       a smaller sampled region than it was of the full circle -- worse
       for every option uniformly, not just ones near a border.

    The exact bbox avoids both failure modes: never reaches past the
    circle into border content, and never over-concentrates on the
    digit at the circle's center. Confirmed only partially in
    synthetic testing (a controlled test didn't cleanly reproduce the
    digit-noise regression -- likely because it used larger, more
    spaced-out circles than the real survey's denser layout) -- this
    change is trusting the real observed regression on the actual
    survey over an inconclusive synthetic result, not a fully proven
    root cause. Worth re-checking against real flagged counts again."""
    x0, y0, x1, y1 = bbox_px
    h, w = bin_img.shape[:2]
    crop = bin_img[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
    if crop.size == 0:
        return 0.0
    return float(np.sum(crop > 0))


def score_question(dewarped_bin, blank_bin, option_elements: list) -> dict:
    """option_elements: the QuestionGuess's confirmed option bboxes (PDF
    points). Returns a dict: answer_index (0-based, or None), confidence_gap,
    flagged (bool), reason, raw_diffs -- same shape/logic as p1's
    score_item, just driven by generic bboxes instead of a hardcoded
    item's option list.

    Checks, in order (identical to p1):
    1. Nothing clears the ink threshold at all -> no_mark_detected
    2. More than one option clears it -> multiple_marks_detected
    3. Top pick isn't clearly ahead of the runner-up -> low_confidence
    4. Otherwise: confident single answer"""
    diffs = []
    for el in option_elements:
        bbox_px = _bbox_to_px(el["bbox"])
        d = _ink_score(dewarped_bin, bbox_px) - _ink_score(blank_bin, bbox_px)
        diffs.append(d)

    if not diffs:
        return {"answer_index": None, "confidence_gap": 0, "flagged": True,
                "reason": "no_options_in_template", "raw_diffs": diffs}

    ranked = sorted(diffs, reverse=True)
    top = ranked[0]
    gap = (ranked[0] - ranked[1]) if len(ranked) > 1 else top
    n_marked = sum(1 for d in diffs if d >= MIN_INK_THRESHOLD and d >= 0.6 * top)

    if top < MIN_INK_THRESHOLD:
        return {"answer_index": None, "confidence_gap": gap, "flagged": True,
                "reason": "no_mark_detected", "raw_diffs": diffs}

    if n_marked >= 2:
        return {"answer_index": int(np.argmax(diffs)), "confidence_gap": gap, "flagged": True,
                "reason": "multiple_marks_detected", "raw_diffs": diffs}

    answer_index = int(np.argmax(diffs))
    flagged = gap < MIN_CONFIDENCE_GAP
    return {"answer_index": answer_index, "confidence_gap": gap, "flagged": flagged,
            "reason": "low_confidence" if flagged else None, "raw_diffs": diffs}


def build_export_xlsx(scored_responses: dict, confirmed_template: list, corrections: dict) -> bytes:
    """scored_responses: {tracking_code: {q_index: score_dict_or_None}}.
    corrections: {(tracking_code, q_index): corrected_value} -- value is
    a 1-based option number, "NA", "MULT", or a transcribed string for
    an open-text question. Colors match p1's own review_app.py
    convention: yellow = flagged/unreviewed, green = corrected/reviewed
    in-app."""
    import openpyxl
    from openpyxl.styles import PatternFill, Font
    from openpyxl.comments import Comment

    FLAG_FILL = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")
    REVIEWED_FILL = PatternFill(start_color="D9EAD3", end_color="D9EAD3", fill_type="solid")

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Responses"

    def col_header(i, g):
        return g.get("label") or (g.get("question_text_guess") or f"Q{i + 1}")[:40]

    headers = ["tracking_code"] + [col_header(i, g) for i, g in enumerate(confirmed_template)]
    ws.append(headers)
    for cell in ws[1]:
        cell.font = Font(bold=True)

    for code in sorted(scored_responses.keys()):
        row_idx = ws.max_row + 1
        ws.cell(row=row_idx, column=1, value=code)
        for qi in range(len(confirmed_template)):
            score = scored_responses[code].get(qi)
            correction = corrections.get((code, qi))
            cell = ws.cell(row=row_idx, column=qi + 2)

            # Comment shows WHY, for debugging -- the reason + raw ink-diff
            # scores per option, same numbers the in-app review card shows.
            # Kept even on corrected cells (as "was: ...") so a corrected
            # answer's original detection is still auditable afterward,
            # not silently overwritten with no trace.
            comment_lines = []
            if score is not None:
                diffs_str = ", ".join(f"{d:.0f}" for d in score.get("raw_diffs", []))
                comment_lines.append(f"Detected reason: {score.get('reason') or 'confident'}")
                comment_lines.append(f"Confidence gap: {score.get('confidence_gap', 0):.0f}")
                comment_lines.append(f"Raw ink-diff scores: [{diffs_str}]")
            elif correction is None:
                comment_lines.append("No photo available for this page, or open-text "
                                     "(needs manual transcription).")

            if correction is not None:
                cell.value = correction
                cell.fill = REVIEWED_FILL
                if comment_lines:
                    comment_lines.insert(0, f"Corrected in review (was: auto-detected).")
            elif score is None:
                cell.value = ""
                cell.fill = FLAG_FILL
            else:
                if score["answer_index"] is None:
                    cell.value = "NA"
                elif score["reason"] == "multiple_marks_detected":
                    cell.value = "MULT"
                else:
                    cell.value = score["answer_index"] + 1  # 1-based, for researcher readability
                if score["flagged"]:
                    cell.fill = FLAG_FILL

            if comment_lines:
                cell.comment = Comment("\n".join(comment_lines), "Paper Survey Tool")

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def score_response(dewarped_images: dict, blank_refs: dict, confirmed_template: list) -> dict:
    """dewarped_images: {page_num: dewarped_bgr_image} for ONE respondent
    (may not have every page, if some of their photos didn't identify).
    confirmed_template: list of dicts (works whether they came straight
    from session -- via dataclasses.asdict(QuestionGuess) -- or from an
    uploaded template JSON; both are plain dicts with the same keys, so
    there's no separate code path needed for either source).
    Returns {index: score_dict_or_None} -- None for a page that wasn't
    available for this respondent, or for a lone-blank (open-text)
    question, which isn't ink-diff scorable and needs manual
    transcription instead."""
    results = {}
    for i, g in enumerate(confirmed_template):
        if g["page"] not in dewarped_images:
            results[i] = None
            continue
        option_elements = g["option_elements"]
        # Falls back to the old implicit heuristic if question_type is missing entirely
        # (a template saved before this field existed) -- see app.py's identical comment.
        q_type = g.get("question_type") or ("open_text" if len(option_elements) == 1 else "mcq")
        is_open_text = q_type == "open_text"
        if is_open_text:
            results[i] = None  # needs manual transcription, not ink-diff
            continue
        dewarped_bin = binarize_dewarped(dewarped_images[g["page"]])
        blank_bin = blank_refs.get(g["page"])
        if blank_bin is None:
            results[i] = None
            continue
        results[i] = score_question(dewarped_bin, blank_bin, option_elements)
    return results
