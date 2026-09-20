"""Generate page -- upload a base PDF, configure marker/QR placement,
preview live, then generate a sample or a batch from a codes.csv.
Core stamping logic lives in core/overlay.py; this file is UI only."""
import io
import zipfile

import pymupdf
import streamlit as st

from overlay import compose_page, preview_page_png, OverlayConfig

st.set_page_config(layout="wide")

def build_codes(date, batch_name, count):
    """Generates count tracking codes as DDMMYYYY-{batch_name}-{i},
    i zero-padded to match the width of the largest number (e.g.
    001, 002, ... for count=100)."""
    date_str = date.strftime("%Y%m%d")
    width = len(str(count))
    return [f"{date_str}-{batch_name}-{str(i).zfill(width)}" for i in range(1, count + 1)]

def generate_batch(src_doc, num_pages, cfg, codes):
    """One combined PDF, every code's pages appended in sequence."""
    out_doc = pymupdf.open()
    progress = st.progress(0.0)
    for i, code in enumerate(codes):
        for p in range(num_pages):
            compose_page(src_doc, p, cfg, code, out_doc=out_doc)
        progress.progress((i + 1) / len(codes))
    progress.empty()
    pdf_bytes = out_doc.tobytes()
    out_doc.close()
    return pdf_bytes


st.header("Generate Survey")

uploaded = st.file_uploader("Upload .pdf", type=["pdf"])
if uploaded is None:
    st.info("Upload a PDF to get started.")
    st.stop()

src_doc = pymupdf.open(stream=uploaded.getvalue(), filetype="pdf")
num_pages = len(src_doc)

col_preview, col_controls = st.columns(2, gap="medium")

with col_controls:
    d1, d2 = st.columns(2)
    survey_date = d1.date_input("Survey date")
    batch_name = d2.text_input("Batch name (e.g. school name)")
    sheet_count = st.number_input("Number of survey sheets", min_value=1, step=1)

    with st.expander("Advanced controls"): # advanced controls expand if user wants to change them
        content_scale = st.number_input("Content scale (%)", 0, 100, 90) / 100.0

        c1, c2 = st.columns(2)
        left_x = c1.slider("Marker left", 0, 100, 40)
        right_x = c2.slider("Marker right", 0, 100, 40)
        top_y = c1.slider("Marker top", 0, 100, 40)
        bottom_y = c2.slider("Marker bottom", 0, 100, 50)
        marker_size = c1.slider("Marker size", 20, 80, 30)
        qr_size = c2.slider("QR size", 20, 80, 40)

        qr_position = c1.radio("QR position", ["top", "bottom"], index=1, horizontal=True)
        sample_code = c2.text_input("Sample code (for sample download)", "20260915-JIRS-03")

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

    if gen_batch.button("Generate batch", width="stretch", disabled=not batch_name):
        codes = build_codes(survey_date, batch_name, sheet_count)
        pdf_bytes = generate_batch(src_doc, num_pages, cfg, codes)
        st.success(f"Generated {len(codes)} pages.")
        st.download_button("Download batch PDF", pdf_bytes,
                            file_name=f"{batch_name}.pdf", mime="application/pdf")

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
