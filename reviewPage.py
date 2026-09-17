import io
import json
import zipfile
import pymupdf

import cv2
import streamlit as st
from processPDF import process_pages, build_excel

st.set_page_config(layout="wide")

st.title("Review")

if "step" not in st.session_state:
    st.session_state.step = "upload"

# -------------------- Helpers --------------------

def get_selected_pages(mode, num_pages, pages_str=""):
    if mode == "Odd pages":
        return list(range(0, num_pages, 2))
    if mode == "Even pages":
        return list(range(1, num_pages, 2))
    selected = []
    if pages_str:
        for part in pages_str.split(","):
            part = part.strip()
            if "-" in part:
                start, end = part.split("-")
                selected += list(range(int(start) - 1, int(end)))
            else:
                selected.append(int(part) - 1)
    return selected

# -------------------- 1. upload --------------------

if st.session_state.step == "upload":
    st.subheader("Upload scanned responses and box template file")
    col1, col2 = st.columns(2, vertical_alignment="center")
    with col1:
        pdf_file = st.file_uploader("Response PDF", type="pdf")
    with col2:
        template_file = st.file_uploader("Template JSON", type="json")

    st.subheader("Select pages and page quality (DPI)")
    col3, col4 = st.columns(2)
    with col3:
        dpi = st.number_input("Page DPI", value=280, step=10)
    with col4:
        num_pages = 0
        if pdf_file:
            num_pages = pymupdf.open(stream=pdf_file.read(), filetype="pdf").page_count
            pdf_file.seek(0)
        
        mode = st.selectbox("Select pages to process", ["Odd pages", "Even pages", "Custom"])
        pages_str = st.text_input("Pages (e.g. 1,3,5-8)") if mode == "Custom" else ""
        selected_pages = get_selected_pages(mode, num_pages, pages_str)

    if st.button("Process", disabled=not (pdf_file and template_file)):
        with st.spinner("Processing..."):
            st.session_state.all_results = process_pages(
                template_file=template_file,
                pdf_file=pdf_file,
                selected_pages=selected_pages,
                page_dpi=dpi,
            )
        if st.session_state.all_results:
            st.session_state.review_idx = 0
            st.session_state.step = "review"
            st.rerun()


# -------------------- 2. review --------------------

elif st.session_state.step == "review":
    tags = list(st.session_state.all_results_images.keys())
    idx = st.session_state.review_idx
    tag = tags[idx]

    st.caption(f"Box {idx + 1} of {len(tags)}: {tag}")
    col5, col6 = st.columns([1,1.55],vertical_alignment="center")
    with col5:
        st.image(st.session_state.all_results_images[tag], channels="BGR")

    with col6:
        box = next(b for p in st.session_state.all_results for b in p["boxes"] if b.get("tag") == tag)
        options = [str(n) for n in range(1, box["bperRow"] + 1)] + ["NA", "MULT"]

        for label, ans in box["questionlabels"].items():
            current = str(ans["value"]) if ans["value"] is not None else options[-2]
            choice = st.pills(label, options, default=current, label_visibility="collapsed", key=f"{tag}_{label}")
            # any explicit choice counts as user-confirmed -- always green after this
            box["questionlabels"][label] = {"value": choice, "confidence_ratio": 1.0, "flagged": False, "source": "user"}

    col7, col8 = st.columns(2)
    if col7.button("Back", disabled=idx == 0):
        st.session_state.review_idx -= 1
        st.rerun()
    if col8.button("Next" if idx < len(tags) - 1 else "Finish"):
        if idx < len(tags) - 1:
            st.session_state.review_idx += 1
        else:
            st.session_state.step = "download"
        st.rerun()


# -------------------- 3. download --------------------

elif st.session_state.step == "download":
    all_results = st.session_state.all_results
    excel_buf = io.BytesIO()
    build_excel(all_results, excel_buf)
    

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        zf.writestr("results.json", json.dumps(all_results, indent=2))

        zf.writestr("responses.xlsx", excel_buf.getvalue())

        for tag, img in st.session_state.all_cropped_boxes.items():
            _, buf = cv2.imencode(".png", img)
            zf.writestr(f"cropped_boxes/{tag}.png", buf.tobytes())

    st.download_button("Download all results (zip)", zip_buf.getvalue(), "results.zip")
