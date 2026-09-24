import io
import json
import zipfile
import cv2
import streamlit as st
from processor import process_images, build_responses_excel, build_eval_excel, pdf_to_image_files

st.set_page_config(layout="wide")

# Turns selected pills (during review) into green colour, rather than default red
st.markdown("""
<style>
button[data-variant="pills"][data-selected="true"] {
    background-color: rgba(46, 204, 113, 0.25) !important;
    color: white !important;
    border-color: #2ecc71 !important;
}
</style>
""", unsafe_allow_html=True)

st.title("Review")

if "step" not in st.session_state:
    st.session_state.step = "upload"

# -------------------- 1. upload --------------------

if st.session_state.step == "upload":
    st.subheader("Upload scanned responses and box template file")
    col1, col2 = st.columns(2, vertical_alignment="center")
    with col1:
        image_files = st.file_uploader("Scanned response images", type=["jpg", "png"], accept_multiple_files=True)
        pdf_file = st.file_uploader("Or upload a PDF instead", type="pdf")
    with col2:
        template_file = st.file_uploader("Template JSON", type="json")

    if st.button("Process", disabled=not ((image_files or pdf_file) and template_file)):
        with st.spinner("Processing..."):
            files_to_process = pdf_to_image_files(pdf_file) if pdf_file else image_files
            st.session_state.all_results = process_images(
                image_files=files_to_process,
                template_file=template_file,
                debug=False,
            )
        if st.session_state.all_results:
            if st.session_state.all_results_images:
                st.session_state.review_idx = 0
                st.session_state.step = "review"
            elif st.session_state.text_results_images:
                st.session_state.text_idx = 0
                st.session_state.step = "review_text"
            else:
                st.session_state.step = "download"
            st.rerun()


# -------------------- 2. review response bubbles --------------------

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
            current = str(ans["model_value"]) if ans["model_value"] is not None else options[-2]
            choice = st.pills(label, options, default=current, label_visibility="collapsed", key=f"{tag}_{label}")
            ans["user_value"] = choice

    col7, col8 = st.columns(2)
    if col7.button("Back", disabled=idx == 0):
        st.session_state.review_idx -= 1
        st.rerun()
    if col8.button("Next" if idx < len(tags) - 1 else "Finish"):
        if idx < len(tags) - 1:
            st.session_state.review_idx += 1
        elif st.session_state.text_results_images:
            st.session_state.text_idx = 0
            st.session_state.step = "review_text"
        else:
            st.session_state.step = "download"
        st.rerun()

# -------------------- 3. Review Text --------------------

elif st.session_state.step == "review_text":
    text_tags = list(st.session_state.text_results_images.keys())
    tidx = st.session_state.text_idx
    ttag = text_tags[tidx]

    st.caption(f"Text box {tidx + 1} of {len(text_tags)}: {ttag}")
    
    st.image(st.session_state.text_results_images[ttag], channels="BGR")

    box = next(b for p in st.session_state.all_results for b in p["boxes"] if b.get("tag") == ttag)
    label = list(box["questionlabels"].keys())[0]  # Text_Box always has exactly one label
    ans = box["questionlabels"][label]

    current_text = ans.get("user_value", "") if isinstance(ans, dict) else ""
    entered_text = st.text_area("Transcription", value=current_text, key=f"text_{ttag}", height=200)

    if not isinstance(ans, dict):
        ans = {"model_value": None}
    ans["user_value"] = entered_text
    box["questionlabels"][label] = ans

    col11, col12 = st.columns(2)
    if col11.button("Back", disabled=tidx == 0):
        st.session_state.text_idx -= 1
        st.rerun()
    if col12.button("Next" if tidx < len(text_tags) - 1 else "Finish"):
        if tidx < len(text_tags) - 1:
            st.session_state.text_idx += 1
        else:
            st.session_state.step = "download"
        st.rerun()

# -------------------- 4. download --------------------

elif st.session_state.step == "download":
    all_results = st.session_state.all_results

    resp_buf = io.BytesIO()
    build_responses_excel(all_results, resp_buf)
    
    eval_buf = io.BytesIO()
    build_eval_excel(all_results, eval_buf)

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        zf.writestr("results.json", json.dumps(all_results, indent=2))

        zf.writestr("responses.xlsx", resp_buf.getvalue())
        zf.writestr("model_evaluation.xlsx", eval_buf.getvalue())

        for tag, img in st.session_state.all_cropped_boxes.items():
            _, buf = cv2.imencode(".png", img)
            zf.writestr(f"cropped_boxes/{tag}.png", buf.tobytes())

    st.download_button("Download all results (zip)", zip_buf.getvalue(), "results.zip")
