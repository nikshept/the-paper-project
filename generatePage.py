"""Generate page -- upload a base PDF, configure marker/QR placement,
preview live, then generate a sample or a batch from a codes.csv.
Core stamping logic lives in core/overlay.py; this file is UI only."""
import csv
import io
import zipfile

import pymupdf
import streamlit as st

from overlay import compose_page, preview_page_png, OverlayConfig


def read_codes(codes_file):
    """Parses codes.csv, returns a validated list of codes, or None
    (with the error already shown) if the file's no good."""
    text = codes_file.getvalue().decode("utf-8")
    reader = csv.DictReader(io.StringIO(text))
    if "tracking_code" not in (reader.fieldnames or []):
        st.error(f"codes.csv must have a 'tracking_code' column, found: {reader.fieldnames}")
        return None
    codes = [row["tracking_code"].strip() for row in reader if row["tracking_code"].strip()]
    if not codes:
        st.error("No codes found in codes.csv.")
        return None
    dupes = {c for c in codes if codes.count(c) > 1}
    if dupes:
        st.error(f"Duplicate tracking codes found, aborting: {dupes}")
        return None
    return codes


def generate_batch(src_doc, num_pages, cfg, codes):
    """One stamped PDF per code, zipped in memory."""
    zip_buf = io.BytesIO()
    progress = st.progress(0.0)
    with zipfile.ZipFile(zip_buf, "w") as zf:
        for i, code in enumerate(codes):
            out_doc = pymupdf.open()
            for p in range(num_pages):
                compose_page(src_doc, p, cfg, code, out_doc=out_doc)
            zf.writestr(f"{code}.pdf", out_doc.tobytes())
            out_doc.close()
            progress.progress((i + 1) / len(codes))
    progress.empty()
    return zip_buf.getvalue()


st.header("Generate Survey")

uploaded = st.file_uploader("Upload .pdf", type=["pdf"])
if uploaded is None:
    st.info("Upload a PDF to get started.")
    st.stop()

src_doc = pymupdf.open(stream=uploaded.getvalue(), filetype="pdf")
num_pages = len(src_doc)

col_preview, col_controls = st.columns(2, gap="medium")

with col_controls:
    content_scale = st.number_input("Content scale (%)", 0, 100, 90) / 100.0

    c1, c2 = st.columns(2)
    left_x = c1.slider("Marker left", 0, 100, 40)
    right_x = c2.slider("Marker right", 0, 100, 40)
    top_y = c1.slider("Marker top", 0, 100, 40)
    bottom_y = c2.slider("Marker bottom", 0, 100, 45)
    marker_size = c1.slider("Marker size", 20, 80, 30)
    qr_size = c2.slider("QR size", 20, 80, 35)

    qr_position = st.radio("QR position", ["top", "bottom"], index=1, horizontal=True)
    sample_code = st.text_input("Sample code (for preview / single download)", "667908")
    codes_file = st.file_uploader("codes.csv (for batch)", type=["csv"])

    cfg = OverlayConfig(
        content_scale=content_scale,
        left_x=left_x, right_x=right_x, top_y=top_y, bottom_y=bottom_y,
        marker_size=marker_size, qr_position=qr_position, qr_size=qr_size,
    )

    gen_sample, gen_batch = st.columns(2)
    if gen_sample.button("Generate sample", width="stretch"):
        out_doc = pymupdf.open()
        for p in range(num_pages):
            compose_page(src_doc, p, cfg, sample_code, out_doc=out_doc)
        st.download_button("Download sample PDF", out_doc.tobytes(),
                            file_name=f"{sample_code}.pdf", mime="application/pdf")
        out_doc.close()

    if gen_batch.button("Generate batch", width="stretch", disabled=codes_file is None):
        codes = read_codes(codes_file)
        if codes is not None:
            zip_bytes = generate_batch(src_doc, num_pages, cfg, codes)
            st.success(f"Generated {len(codes)} PDFs.")
            st.download_button("Download all as .zip", zip_bytes,
                                file_name="generated_surveys.zip", mime="application/zip")

with col_preview:
    if "preview_page" not in st.session_state:
        st.session_state.preview_page = 0
    st.session_state.preview_page = min(st.session_state.preview_page, num_pages - 1)

    png_bytes = preview_page_png(src_doc, st.session_state.preview_page, cfg, sample_code)
    st.image(png_bytes)

    p1, p2, p3 = st.columns([1, 2, 1])
    if p1.button("\u2190", width="stretch", disabled=st.session_state.preview_page == 0):
        st.session_state.preview_page -= 1
        st.rerun()
    p2.markdown(f"<div style='text-align:center'>Page {st.session_state.preview_page + 1} of {num_pages}</div>",
                unsafe_allow_html=True)
    if p3.button("\u2192", width="stretch", disabled=st.session_state.preview_page >= num_pages - 1):
        st.session_state.preview_page += 1
        st.rerun()
