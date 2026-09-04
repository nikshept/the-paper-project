"""
Template Builder -- auto-guesses "this row of N shapes is one question's
options" from Stage 2's (structure.py) purely positional row data, so the
researcher has a starting point to confirm/edit rather than building a
template from scratch. This is explicitly a GUESS, not a classification
claim: it never asserts semantic meaning (no "this is a Likert item"),
only proposes a grouping based on structural signal (a row with 2+
shape/blank elements is very likely a set of response options; the row
immediately above is very likely that question's text) -- the same
"structure, not meaning" principle extract.py/structure.py were built
on, just used to make a proposal instead of stopping at description.

The confirmed-or-edited result of this becomes the template Score uses
for every photo in the batch -- built once per survey, not per photo.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class QuestionGuess:
    page: int
    row_index: int  # index into that page's rows, per structure.py's output
    question_text_guess: str
    option_elements: list  # the row's shape/blank element dicts (from structure.py)
    preview_bbox: list = None  # combined bbox (text row(s) + option row), for cropping a preview
    text_line_count: int = 0  # how many preceding rows are CURRENTLY included as question
                                # text -- adjustable by the researcher (+/- controls), not
                                # auto-walked. See resolve_guess_text.
    text_line_count_below: int = 0  # same idea, for rows AFTER the option row -- a
                                      # clarifying note or continuation printed below it.
                                      # Always starts at 0; never auto-guessed, only added
                                      # by the researcher when actually needed.
    options_manually_set: bool = False  # True once the researcher has corrected the option
                                          # count/positions -- prevents the +/- text-line
                                          # controls from silently overwriting that correction
                                          # (they'd otherwise re-run auto-detection on every
                                          # click, discarding a manual fix)
    rev: int = 0  # bumped on every programmatic change (line count, regenerate) -- baked into
                   # widget keys in app.py so Streamlit treats the widget as genuinely new and
                   # actually re-reads `value=`. Streamlit only honors value= on a key's FIRST
                   # render ever; after that the widget's own session-state copy wins even if
                   # the underlying variable changes -- a well-documented Streamlit gotcha,
                   # exactly consistent with "+/- doesn't seem to do anything."
    confirmed: bool = False  # researcher hasn't reviewed this yet
    discarded: bool = False  # researcher marked this as not a real question
    label: str = None  # researcher-assigned identifier (e.g. "Q1", "K3") -- None until set
    is_manual: bool = False  # True for questions added via click-and-drag (no real row_index
                               # to anchor to -- the +/- line controls don't apply to these)
    question_type: str = "mcq"  # "mcq" or "open_text" -- EXPLICIT and researcher-confirmable,
                                  # not inferred implicitly at each usage site. That implicit
                                  # approach (checking len(option_elements)==1 in multiple
                                  # separate places) is exactly what caused a real, serious bug:
                                  # every manually-added open-text question was silently scored
                                  # as multiple-choice instead, writing "1" for every response
                                  # with no flag to catch it, because the inference lived in two
                                  # different files and didn't agree. A stored, single-source,
                                  # user-visible field removes that whole class of mismatch.


def _row_bbox(row: list) -> list:
    """Combined bounding box of every element in a row, for cropping a
    preview image around it."""
    x0 = min(el["bbox"][0] for el in row if el.get("bbox"))
    y0 = min(el["bbox"][1] for el in row if el.get("bbox"))
    x1 = max(el["bbox"][2] for el in row if el.get("bbox"))
    y1 = max(el["bbox"][3] for el in row if el.get("bbox"))
    return [x0, y0, x1, y1]


def _is_option_like(el: dict) -> bool:
    """An element counts as a candidate response-option marker ONLY if
    it's an actually-detected shape or blank field -- a real drawn
    circle/box/checkbox, or an underscore-run fill-in field. Earlier
    versions also counted short "words" (1-2 characters) as candidates,
    to catch text-only markers like circled digits or checkbox glyphs
    Stage 1 doesn't classify by identity. That caused a structural
    problem no amount of exclusion-rule patching fully solved: a short
    word is indistinguishable from a real option marker by length
    alone, and there's always another short word waiting (a row
    number, "I", "if", "a") -- confirmed repeatedly in testing, most
    recently a plain row-number column ("5") and the word "I" both
    getting swept in as if they were response options.

    Restricting to real shapes/blanks trades recall for precision on
    purpose: a survey using purely-typed markers with no drawn shape at
    all (bare "1 2 3 4 5" with nothing drawn around them) won't be
    auto-detected anymore -- those go through the manual "add a
    question" flow instead. The goal is making guesses trustworthy
    enough to be worth reviewing, not maximizing how much gets
    auto-guessed."""
    return el["type"] in ("shape", "blank")


def _bbox_center(bbox):
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def _dedupe_overlapping_options(option_like: list) -> list:
    """A drawn bubble with a digit printed inside it (p1's own Likert
    style, and a very common survey convention generally) shows up as
    TWO separate detected elements at nearly the same position -- a
    "shape" and a "word" (the digit). Counted separately, 5 real bubbles
    become 10+ "options". This collapses any option-like element whose
    center falls inside another option-like element's bbox into ONE,
    preferring to keep the shape (its position is the more precise
    marker of the two) over the overlapping word."""
    shapes = [el for el in option_like if el["type"] == "shape"]
    others = [el for el in option_like if el["type"] != "shape"]

    kept_others = []
    for el in others:
        cx, cy = _bbox_center(el["bbox"])
        overlaps_a_shape = any(
            s["bbox"][0] <= cx <= s["bbox"][2] and s["bbox"][1] <= cy <= s["bbox"][3]
            for s in shapes
        )
        if not overlaps_a_shape:
            kept_others.append(el)

    return shapes + kept_others


def _dedupe_row_for_ratio(row: list) -> list:
    """Same overlap logic as _dedupe_overlapping_options, applied to the
    WHOLE row -- needed so the option ratio's denominator doesn't still
    count a digit-word as separate "diluting" content after it's
    already been absorbed into its overlapping shape on the numerator
    side. Without this, a genuinely dense option row could incorrectly
    read as too diluted to qualify (confirmed in testing: deduping only
    the option side dropped a real 5-bubble row's ratio from passing to
    failing, since the denominator still counted the absorbed digits)."""
    shapes = [el for el in row if el["type"] == "shape"]
    kept = []
    for el in row:
        if el["type"] == "shape":
            kept.append(el)
            continue
        cx, cy = _bbox_center(el["bbox"]) if el.get("bbox") else (None, None)
        overlaps_a_shape = cx is not None and any(
            s["bbox"][0] <= cx <= s["bbox"][2] and s["bbox"][1] <= cy <= s["bbox"][3]
            for s in shapes
        )
        if not overlaps_a_shape:
            kept.append(el)
    return kept


def _is_lone_blank_row(row: list) -> bool:
    """A single word consisting entirely of a repeated non-alphanumeric
    character (an underscore run, a dash run, a dot run) is a common,
    generic signal for a fill-in-the-blank field. Stage 1 (extract.py)
    deliberately doesn't classify this -- it's meaningful only in
    context (one lone token filling a whole row), which is exactly the
    kind of judgment that belongs in this guessing layer, not in
    Stage 1's mechanical extraction."""
    if len(row) != 1 or row[0]["type"] != "word":
        return False
    text = row[0].get("text", "")
    return len(text) >= 3 and len(set(text)) == 1 and not text[0].isalnum()


def resolve_guess_text(rows: list, row_index: int, lines_above: int, lines_below: int = 0) -> tuple:
    """Given EXPLICIT numbers of preceding AND following rows to include
    as question text, returns (text_guess, preview_bbox, option_elements).
    No stopping heuristics here -- an explicit count means the count has
    already been decided (by a safe default, or by the researcher via
    +/- controls in the review UI), so this just executes that
    decision. Confirmed in testing that any AUTOMATIC multi-row lookback
    heuristic, however it's tuned, can misjudge how much preceding
    content genuinely belongs to a question (a section heading, a scale
    legend, or another question's own text all look structurally similar
    to "plain text above an option row"). Rather than keep chasing that
    with more special-case rules, this makes it directly, cheaply
    adjustable instead of an opaque black box -- the researcher can see
    the crop grow/shrink line by line and stop exactly where it's right.
    lines_below covers the less common but real case of a clarifying
    note or continuation printed AFTER the option row.

    Still pulls text sharing the option row's OWN row (a wrapped final
    line sitting beside the bubbles) automatically -- that's a
    positional fact from the page layout, not a judgment call, so there's
    nothing for the researcher to decide there."""
    row = rows[row_index]
    option_like = _dedupe_overlapping_options([el for el in row if _is_option_like(el)])
    is_lone_blank = (len(row) == 1 and row[0]["type"] == "blank") or _is_lone_blank_row(row)

    start = max(0, row_index - lines_above)
    text_rows_above = rows[start:row_index]
    end = min(len(rows), row_index + 1 + lines_below)
    text_rows_below = rows[row_index + 1:end]

    option_like_set = {id(el) for el in option_like}
    deduped_row = _dedupe_row_for_ratio(row)
    same_row_text = [] if is_lone_blank else [
        el for el in deduped_row if el["type"] == "word" and id(el) not in option_like_set
    ]

    above_str = " ".join(el.get("text", "") for r in text_rows_above for el in r)
    same_row_str = " ".join(el.get("text", "") for el in same_row_text)
    below_str = " ".join(el.get("text", "") for r in text_rows_below for el in r)
    text_guess = " ".join(part for part in (above_str, same_row_str, below_str) if part).strip()

    text_row_elements = ([el for r in text_rows_above for el in r] + same_row_text
                          + [el for r in text_rows_below for el in r])
    combined_bbox = _row_bbox(text_row_elements + row)
    option_elements = option_like if option_like else row

    return text_guess, combined_bbox, option_elements


def guess_questions(structured: dict, min_options: int = 2) -> list:
    """Walks each page's rows (already top-to-bottom, left-to-right from
    structure.py) looking for rows with at least `min_options` real
    detected shapes/blanks -- a strong, low-false-positive signal of
    "these are response options". Earlier versions also counted short
    "words" toward this and needed a ratio-based dilution check to stop
    ordinary sentences (which often contain one or two short words by
    chance) from false-positiving. Restricting to real shapes removes
    that whole failure class -- a genuine drawn circle or checkbox
    doesn't show up "incidentally" in prose the way a short word does
    -- so the dilution check was removed too (confirmed necessary: it
    was incorrectly rejecting legitimate all-shape option rows that
    happened to share a line with a normal amount of question text).

    For each row that passes, the row immediately above (if it's a
    plain text row) is guessed as that question's text -- simplest
    defensible default, not a claim of certainty. Rows already claimed
    as a question's text aren't reused as another question's text.

    A lone "blank" element (a single fill-in-the-blank field) is always
    treated as a valid one-option "question" -- an open-text question
    genuinely only has one field, unlike a multiple-choice question
    which needs several options to be plausible.

    Trade-off, worth knowing: a survey using purely-typed option
    markers with no drawn shape at all (bare "1 2 3 4 5" with nothing
    drawn around them) won't be auto-detected. Those go through the
    manual "add a question" flow instead -- deliberately traded recall
    for precision, so a guess is worth trusting when it appears."""
    guesses = []

    for page in structured["pages"]:
        rows = page["rows"]
        claimed_as_text = set()

        for i, row in enumerate(rows):
            option_like = _dedupe_overlapping_options([el for el in row if _is_option_like(el)])
            is_lone_blank = (len(row) == 1 and row[0]["type"] == "blank") or _is_lone_blank_row(row)
            if not is_lone_blank and len(option_like) < min_options:
                continue  # not enough real detected shapes to be a question row. No ratio/
                          # dilution check needed anymore -- that existed to filter out
                          # ordinary sentences that happened to contain a couple of short
                          # WORDS, a risk that doesn't apply to real detected shapes (they
                          # don't show up "incidentally" the way short words do), and
                          # confirmed in testing that keeping it caused a real regression:
                          # a legitimate all-shapes option row sharing space with a normal
                          # amount of wrapped question text got rejected as "too diluted."

            # Safe DEFAULT: at most the single row immediately above,
            # only if it's plain text, not already used by another
            # question, and doesn't itself look like an option row.
            # Deliberately NOT an automatic multi-row walk (see
            # resolve_guess_text) -- the researcher adjusts from here
            # with +/- controls if a question genuinely wraps further.
            default_line_count = 0
            if i - 1 >= 0 and (i - 1) not in claimed_as_text:
                prev_row = rows[i - 1]
                if all(el["type"] == "word" for el in prev_row):
                    prev_option_like = _dedupe_overlapping_options(
                        [el for el in prev_row if _is_option_like(el)])
                    looks_like_option_row = len(prev_option_like) >= min_options
                    if not looks_like_option_row:
                        default_line_count = 1
                        claimed_as_text.add(i - 1)

            text_guess, combined_bbox, final_option_elements = resolve_guess_text(
                rows, i, default_line_count)

            guesses.append(QuestionGuess(
                page=page["page"], row_index=i,
                question_text_guess=text_guess,
                option_elements=final_option_elements,
                preview_bbox=combined_bbox,
                text_line_count=default_line_count,
                question_type="open_text" if is_lone_blank else "mcq",
            ))

    return guesses


def _dominant_cluster(option_elements: list) -> list:
    """Finds the longest run of elements with consistent spacing between
    them, robust to a single outlier (confirmed necessary in testing:
    without this, a stray miscounted word far from the real bubble
    cluster badly skewed the min/max span, producing wildly-spread
    regenerated boxes instead of ones matching where the real options
    actually are). Real option rows are evenly spaced; a spurious
    extra detection generally isn't anywhere near that spacing."""
    if len(option_elements) <= 2:
        return option_elements
    sorted_els = sorted(option_elements, key=lambda el: (el["bbox"][0] + el["bbox"][2]) / 2)
    centers = [(el["bbox"][0] + el["bbox"][2]) / 2 for el in sorted_els]
    gaps = [centers[i + 1] - centers[i] for i in range(len(centers) - 1)]
    sorted_gaps = sorted(gaps)
    median_gap = sorted_gaps[len(sorted_gaps) // 2]

    best_start, best_end = 0, 0
    start = 0
    for i, gap in enumerate(gaps):
        if gap > median_gap * 2.5 or gap < median_gap * 0.3:
            if i - start > best_end - best_start:
                best_start, best_end = start, i
            start = i + 1
    if len(gaps) - start > best_end - best_start:
        best_start, best_end = start, len(gaps)
    return sorted_els[best_start:best_end + 1]


def regenerate_evenly_spaced_options(option_elements: list, n: int) -> list:
    """If detection got the option COUNT wrong (missed one, or
    double-counted a stray glyph), there's no reliable way to know
    which specific detected bbox is spurious/missing without semantic
    understanding -- but every option-row convention seen in this
    project (drawn circles, glyph markers, checkboxes) lays its options
    out roughly evenly across a horizontal span. So rather than asking
    the researcher to hand-place individual bboxes, this takes the
    span the RELIABLE, evenly-spaced cluster of current detections
    covers (see _dominant_cluster -- ignores outliers so a stray
    miscounted word doesn't skew the span) and re-divides it into
    exactly `n` evenly-spaced boxes of the same average size. This is a
    real, structural correction -- not a cosmetic one -- since these
    bboxes are exactly what score.py's ink-diff reads positions from."""
    if not option_elements or n < 1:
        return option_elements

    cluster = _dominant_cluster(option_elements)
    x0s = [el["bbox"][0] for el in cluster]
    x1s = [el["bbox"][2] for el in cluster]
    y0s = [el["bbox"][1] for el in cluster]
    y1s = [el["bbox"][3] for el in cluster]
    span_x0, span_x1 = min(x0s), max(x1s)
    avg_y0 = sum(y0s) / len(y0s)
    avg_y1 = sum(y1s) / len(y1s)
    avg_w = sum(x1 - x0 for x0, x1 in zip(x0s, x1s)) / len(x0s)

    if n == 1:
        centers = [(span_x0 + span_x1) / 2]
    else:
        step = (span_x1 - span_x0) / (n - 1)
        centers = [span_x0 + i * step for i in range(n)]

    return [
        {"type": "shape", "bbox": [cx - avg_w / 2, avg_y0, cx + avg_w / 2, avg_y1],
         "meta": {"kind": "regenerated"}}
        for cx in centers
    ]


def summarize(guesses: list) -> str:
    lines = [f"{len(guesses)} question(s) guessed"]
    for g in guesses:
        lines.append(f"  page {g.page} row {g.row_index}: "
                     f"\"{g.question_text_guess}\" -- {len(g.option_elements)} option(s)")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from core.extract import extract
    from core.structure import structure

    if len(sys.argv) != 2:
        print("Usage: python template_builder.py <file.docx|file.pdf>")
        sys.exit(1)

    res = extract(sys.argv[1])
    struct = structure(res)
    guesses = guess_questions(struct)
    print(summarize(guesses))
