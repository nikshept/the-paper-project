import json
import pymupdf
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas

st.header("Draw Box Templates")

if "step" not in st.session_state:
    st.session_state.step = "upload"

### Upload the scanned image
if st.session_state.step == "upload":
    col1, col2 = st.columns(2, vertical_alignment="center")
    with col1:
        st.write("Step 1: Upload a scanned image of a response sheet.")
    with col2:
        fileUpload = st.file_uploader("Upload here:", type=["png", "jpg", "pdf"])

    if fileUpload is not None:
        if fileUpload.type == "application/pdf":
            doc = pymupdf.open(stream=fileUpload.read(), filetype="pdf")
            pix = doc[0].get_pixmap(dpi=280)
            st.session_state.template_img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        else:
            st.session_state.template_img = Image.open(fileUpload)
        st.session_state.step = "label"
        st.rerun()

### Drawing boxes on the image canvas and generating a template
elif st.session_state.step == "label":
    st.write("Step 2: Draw one box around each set of response bubbles and one box around the QR code.")
    img = st.session_state.template_img
    max_width = 700
    scale = min(1, max_width / img.width)
    canvas_w, canvas_h = int(img.width * scale), int(img.height * scale)

    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.1)",  # Fixed fill color with some opacity
        stroke_width=2,
        stroke_color="#0B8132",
        background_image=img,
        update_streamlit=True,
        height=canvas_h,
        width=canvas_w,
        drawing_mode="rect",
        return_image_data=True,
        key="canvas",
    )

    # Store the box boundaries in a template
    boxesTemplate = []
    for obj in canvas_result.json_data["objects"]:
        x1, y1 = obj["left"], obj["top"]
        x2, y2 = x1 + obj["width"], y1 + obj["height"]
        boxesTemplate.append({
            "x1": x1 / canvas_w, "y1": y1 / canvas_h,
            "x2": x2 / canvas_w, "y2": y2 / canvas_h,
        })

    # Allow the user to specify the box type and question labels for each box
    for i, box in enumerate(boxesTemplate):
        col1, col2, col3 = st.columns(3)
        with col1:
            box["box_type"] = st.selectbox(f"What is box {i+1}?", ["Response_Box", "QR_Box"], key=f"type_{i}")
        with col2:
            box["labels"] = st.text_input(f"Label(s) for box {i+1}", key=f"label_{i}")
        with col3:
            box["bperRow"] = st.number_input(f"Bubbles per row in Box {i+1}", min_value=1, step=1, key=f"bubbles_{i}")

        # Store the question labels separately
        label_list = box["labels"].split(",")  # "q1,q2,q3" -> ["q1", "q2", "q3"]

        box["questionlabels"] = {}
        for label in label_list:
            label = label.strip()       # remove stray spaces, e.g. " q2" -> "q2"
            if label:                   # skip empty entries (blank input, trailing comma)
                box["questionlabels"][label] = True

    # Store in session state and allow the user to download the template as a JSON file
    # st.json(boxesTemplate) # uncomment this line to display the template in the app for debugging purposes
    st.session_state.boxesTemplate = boxesTemplate

    st.write("Step 3: Download the template for the next stage.")
    st.download_button(
        "Download template",
        json.dumps(st.session_state.boxesTemplate, indent=2),
        file_name="template.json",
        disabled=not st.session_state.boxesTemplate,
        )

    # Uncomment for displaying the box details in the app for debugging purposes
    # st.write("Box details:")
    # for i, objects in enumerate(canvas_result.json_data["objects"]):
        # st.write(f"Box {i + 1}:  x1 is {objects['left']}, x2 is {objects['left'] + objects['width']}, y1 is {objects['top']}, y2 is {objects['top'] + objects['height']}")