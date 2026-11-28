"""Generate page -- upload a base PDF, configure marker/QR placement,
preview live, then generate a sample or a batch from a codes.csv.
Core stamping logic lives in core/overlay.py; this file is UI only."""
import pymupdf, glob, os
import streamlit as st

from overlay import compose_page, preview_page_png, OverlayConfig
from contacts import read_table

st.set_page_config(layout="wide")

# reads QR codes from the first column of the first sheet, skipping the header row
def read_codes(file):
    first_column = read_table(file).iloc[:, 0]
    return [str(code).strip() for code in first_column]

# one code per sheet: DDMMYYYY-BATCH-TREK-i, with full names (overlay.py adds the page number)
def build_codes(date, batch_name, trek_name, count):
    date_str = date.strftime("%Y%m%d")
    width = len(str(count))
    return [f"{date_str}-{batch_name}-{trek_name}-{str(i).zfill(width)}" for i in range(1, count + 1)]

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

def short_name(path):
    # shows the filename up to and including "Survey"
    name = os.path.basename(path)
    end = name.lower().find("survey")
    return name[:end + len("survey")] if end != -1 else os.path.splitext(name)[0]

SURVEY_DIR = "assets/surveys"  # folder in the repo holding the predefined PDFs

st.header("Generate Survey")

source = st.radio("Survey sheet", ["Upload my own", "Use a predefined sheet"], horizontal=True)

if source == "Upload my own":
    uploaded = st.file_uploader("Upload a survey sheet (.pdf)", type=["pdf"])
    if uploaded is None:
        st.stop()
    pdf_bytes = uploaded.getvalue()
else:
    sheets = sorted(glob.glob(f"{SURVEY_DIR}/*.pdf"))
    if not sheets:
        st.warning(f"No PDFs found in '{SURVEY_DIR}/'.")
        st.stop()
    choice = st.segmented_control("Select a survey sheet", sheets, format_func=short_name)
    if choice is None:
        st.stop()
    with open(choice, "rb") as f:
        pdf_bytes = f.read()

src_doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
num_pages = len(src_doc)

col_preview, col_controls = st.columns(2, gap="medium")

with col_controls:
    code_source = st.radio("QR codes", ["Generate from batch info", "Upload file"], horizontal=True)

    if code_source == "Generate from batch info":
        d1, d2 = st.columns(2)
        survey_date = d1.date_input("Batch date")
        trek_name = d2.text_input("Trek name")
        batch_name = d1.text_input("Organisation name")
        sheet_count = d2.number_input("Number of survey sheets", min_value=1, step=1)
        # placeholders show in the preview until names are entered
        codes = build_codes(survey_date, batch_name or "ORG", trek_name or "TRK", sheet_count)
        ready = bool(batch_name and trek_name)
        batch_file = f"{batch_name}.pdf"
    else:
        codes_file = st.file_uploader("File of QR codes (csv or excel, first column, header in row 1)", type=["csv", "xlsx"])
        codes = read_codes(codes_file) if codes_file else []
        duplicates = len(codes) != len(set(codes))
        if duplicates:
            st.error("Duplicate QR codes found in the file.")
        ready = bool(codes) and not duplicates
        batch_file = "survey_batch.pdf"

    sample_code = codes[0] if codes else "SAMPLE"
    

    with st.expander("Advanced controls"): # advanced controls expand if user wants to change them
        content_scale = st.number_input("Content scale (%)", 0, 120, 95) / 100.0

        c1, c2 = st.columns(2)
        left_x = c1.slider("Marker left", 0, 100, 30)
        right_x = c2.slider("Marker right", 0, 100, 30)
        top_y = c1.slider("Marker top", 0, 100, 30)
        bottom_y = c2.slider("Marker bottom", 0, 100, 30)
        marker_size = c1.slider("Marker size", 20, 80, 30)
        qr_size = c2.slider("QR size", 20, 80, 40)

        qr_position = c1.radio("QR position", ["top", "bottom"], index=1, horizontal=True)

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

    if gen_batch.button("Generate batch", width="stretch", disabled=not ready):
        pdf_bytes = generate_batch(src_doc, num_pages, cfg, codes)
        st.success(f"Generated {len(codes)} pages.")
        st.download_button("Download batch PDF", pdf_bytes,
                            file_name=batch_file, mime="application/pdf")

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
