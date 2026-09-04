"""
Stage 2 -- Structure.

Takes Stage 1's flat, unordered element list and organizes it the way it
actually sits on the page: grouped into rows (top-to-bottom), ordered
left-to-right within each row. That's it -- this module does not decide
that a row "is a question" or "is an option set". It only answers "what
is next to what, and what comes before what" -- a positional
transcription of the layout into code/JSON, nothing more.

Why no classification: deciding what a row MEANS depends on the
specific survey's conventions, which vary (drawn circles vs. circled
digits vs. checkboxes vs. something else entirely -- already seen three
different conventions across two real test documents in this project).
Encoding a meaning-guess here would just reintroduce the same
overfitting problem Stage 1 had before it was made generic. Ordering
elements by position, by contrast, requires no assumption about what a
survey looks like -- every survey has SOME notion of top-to-bottom,
left-to-right reading order, so that's the only structure imposed here.

Table cells are NOT re-clustered by row here -- they already carry
exact row/col from the source docx XML (Stage 1), which is more
reliable than re-deriving it from position. This module only groups
elements that Stage 1 left un-positioned-relative-to-each-other: words,
shapes, images with a real page position.
"""
from __future__ import annotations

from dataclasses import asdict


# ============================================================ ROW GROUPING

def _row_tolerance(elements: list) -> float:
    """Derives a row-grouping tolerance from the elements actually on
    the page, rather than a hardcoded constant -- a survey with large
    print and one with small print need different tolerances, and this
    measures rather than assumes. Uses the median element height (a
    robust central estimate, insensitive to a handful of unusually
    tall/short outliers like a heading), halved -- two elements whose
    vertical centers are within half a typical line's height of each
    other are treated as being on the same line."""
    heights = [(el.bbox[3] - el.bbox[1]) for el in elements if el.bbox]
    if not heights:
        return 5.0  # fallback for a page with no positioned elements at all
    heights.sort()
    median_h = heights[len(heights) // 2]
    return max(median_h * 0.5, 1.0)


def _group_into_rows(elements: list) -> list:
    """Groups elements into rows by y-position, then sorts each row
    left-to-right by x-position. `elements` must all share a page and
    all have a real bbox (positioned elements only)."""
    positioned = [el for el in elements if el.bbox is not None]
    if not positioned:
        return []

    tolerance = _row_tolerance(positioned)

    def y_center(el):
        return (el.bbox[1] + el.bbox[3]) / 2

    ordered = sorted(positioned, key=y_center)

    rows = []
    current_row = [ordered[0]]
    current_y = y_center(ordered[0])
    for el in ordered[1:]:
        ey = y_center(el)
        if abs(ey - current_y) <= tolerance:
            current_row.append(el)
            # running average keeps the row's reference y stable as it grows,
            # rather than drifting toward whichever element was added last
            current_y = sum(y_center(e) for e in current_row) / len(current_row)
        else:
            rows.append(current_row)
            current_row = [el]
            current_y = ey
    rows.append(current_row)

    for row in rows:
        row.sort(key=lambda el: el.bbox[0])  # left-to-right within the row

    return rows


# ============================================================== STRUCTURE =

def structure(extract_result) -> dict:
    """Builds a purely positional, page-by-page transcription:

        {
          "pages": [
            {
              "page": 0,
              "rows": [
                [element_dict, element_dict, ...],   # row 0, left-to-right
                [element_dict, ...],                  # row 1
                ...
              ]
            },
            ...
          ],
          "unpositioned": [element_dict, ...],  # docx table cells / images
                                                  # with no page position --
                                                  # kept in their own original
                                                  # document order, untouched
        }

    No element is reclassified, relabeled, merged, or dropped -- every
    Element from Stage 1 appears exactly once, either inside a row (if
    it had a page position) or in `unpositioned` (if it didn't, e.g. a
    docx table cell or docx image). This is Stage 1's data, reordered --
    not new information."""
    by_page = {}
    unpositioned = []

    for el in extract_result.elements:
        if el.page is None or el.page < 0 or el.bbox is None:
            unpositioned.append(el)
        else:
            by_page.setdefault(el.page, []).append(el)

    pages_out = []
    for page_idx in sorted(by_page):
        rows = _group_into_rows(by_page[page_idx])
        pages_out.append({
            "page": page_idx,
            "rows": [[asdict(el) for el in row] for row in rows],
        })

    return {
        "pages": pages_out,
        "unpositioned": [asdict(el) for el in unpositioned],
    }


# =============================================================== DEBUG ====

def summarize(structured: dict) -> str:
    """Quick human-readable view -- row count per page, and how many
    elements landed in each row, so you can eyeball whether the
    row-grouping actually matches the real document's line breaks."""
    lines = []
    for page in structured["pages"]:
        row_sizes = [len(row) for row in page["rows"]]
        lines.append(f"page {page['page']}: {len(page['rows'])} rows, sizes={row_sizes}")
    if structured["unpositioned"]:
        lines.append(f"unpositioned: {len(structured['unpositioned'])} elements "
                      f"(table cells / docx images with no page bbox)")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    import json
    sys.path.insert(0, ".")
    from core.extract import extract

    if len(sys.argv) != 2:
        print("Usage: python structure.py <file.docx|file.pdf>")
        sys.exit(1)

    res = extract(sys.argv[1])
    struct = structure(res)
    print(summarize(struct))
    print()
    print(json.dumps(struct, indent=2)[:3000])  # preview -- full structure is often large
