import cv2, json, copy
import numpy as np
import openpyxl
from openpyxl.comments import Comment
from openpyxl.styles import PatternFill
from dewarp import dewarp_image_A4
import getResponses
import streamlit as st
import streamlit.logger
import os
streamlit.logger.set_log_level("ERROR")
os.makedirs("output", exist_ok=True)

################################ HELPERS #############################

### Crop images by box dimensions
def img_cropper (image, box):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]

    croppedImg = image[int(y1*h):int(y2*h), int(x1*w):int(x2*w)]

    return croppedImg

### Get QR Code function
def get_QR (croppedboxImage):
    qrID, _, _ = cv2.QRCodeDetector().detectAndDecode(croppedboxImage)
    if qrID == "":
        res, points = cv2.wechat_qrcode.WeChatQRCode().detectAndDecode(croppedboxImage)
        qrID = res[0] if res else ""

    return qrID


################################ MAIN PIPELINE #############################

### Validate if the template has QR boxes and response boxes
def validate_template(template, debug=False):
    if not any(box["box_type"] == "QR_Box" for box in template):
        msg = "No QR box found in template. Please re-upload a valid template."
        print(msg) if debug else st.error(msg)
        return False
    if not any(box["box_type"] == "Response_Box" for box in template):
        msg = "No Response box found in template. Please re-upload a valid template."
        print(msg) if debug else st.error(msg)
        return False
    return True

### Validate the images for successful dewarp, and reads QRs
def validate_images(image_files, template, debug=False):
    qr_box = next(box for box in template if box["box_type"] == "QR_Box")
    flagged_images = []
    dewarped_images = {}
    qr_ids = {}

    for img_file in image_files:
        file_bytes = np.frombuffer(img_file.read(), dtype=np.uint8)
        original = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_file.seek(0)  # reset -- dewarp_image_A4 reads this same stream again below

        if original.shape[1] < 1500:
            flagged_images.append(f"{img_file.name} (low quality)")
            continue

        dewarped = dewarp_image_A4(img_file, debug=debug)
        if dewarped is None:
            flagged_images.append(f"{img_file.name} (dewarp failed)")
            continue

        cropped_qr = img_cropper(dewarped, qr_box)
        qr_id = get_QR(cropped_qr)
        if qr_id == "":
            flagged_images.append(f"{img_file.name} (QR not detected)")
            continue

        dewarped_images[os.path.basename(img_file.name)] = dewarped
        qr_ids[os.path.basename(img_file.name)] = qr_id

    if flagged_images:
        msg = f"Please re-upload after improving the quality of: {flagged_images}"
        print(msg) if debug else st.error(msg)

    return flagged_images, dewarped_images, qr_ids

### Process the pages and get results (responses and scored images)
def extract_results(template, dewarped_images, qr_ids, debug=False):
    all_results = []
    st.session_state.all_results_images = {}
    st.session_state.all_cropped_boxes = {}

    # cycle through page images and get answers
    for key, img in dewarped_images.items():
        boxes = copy.deepcopy(template)
        page_qr_id = qr_ids[key]
        print(f"Investigating {key}, QR: {page_qr_id}")

        for i, box in enumerate(boxes):
            if box["box_type"] == "Response_Box":
                croppedResimg = img_cropper(img, box)
                tag = f"{key}_{page_qr_id}_Box{i}"
                box["tag"] = tag
                st.session_state.all_cropped_boxes[tag] = croppedResimg
                if debug:
                    cv2.imwrite(f"output/00_cropped_box_{tag}.png", croppedResimg)

                results, imgResults = getResponses.get_Responses(croppedResimg, box["bperRow"], tag, debug)

                for label, (value, gap_ratio, flag, source, flagReason) in zip(box["questionlabels"].keys(), results):
                    box["questionlabels"][label] = {"value": value, "confidence_ratio": gap_ratio, "flagged": flag, "source": source, "flagReason": flagReason}

                st.session_state.all_results_images[tag] = imgResults

                valsCount = len(results)
                flagCount = sum(1 for (_, _, flag, _, _) in results if flag)
                print(f"Box{i}: Received {valsCount} values and {flagCount} were flagged.")

        all_results.append({"page": key, "qr_ID": page_qr_id, "boxes": boxes})

    return all_results

### Build excel sheet based on results
def build_excel(all_results, output_path):
    RED_FILL = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
    wb = openpyxl.Workbook()
    ws = wb.active

    all_labels = []
    for page in all_results:
        for box in page["boxes"]:
            for label in box.get("questionlabels", {}):
                if label not in all_labels:
                    all_labels.append(label)

    ws.append(["page", "qr_ID"] + all_labels)

    for page in all_results:
        r = ws.max_row + 1
        ws.cell(row=r, column=1, value=page["page"])
        ws.cell(row=r, column=2, value=page["qr_ID"])
        for box in page["boxes"]:
            if box["box_type"] != "Response_Box":
                continue
            for label, ans in box.get("questionlabels", {}).items():
                if not isinstance(ans, dict):
                    continue
                col = 3 + all_labels.index(label)
                cell = ws.cell(row=r, column=col, value=ans.get("value"))
                cell.comment = Comment(f"confidence_ratio: {ans.get('confidence_ratio'):.0%}\nflagged: {ans.get('flagged')}\nsource: {ans.get('source')}", "cropper")
                if ans.get("flagged"):
                    cell.fill = RED_FILL

    wb.save(output_path)

### Top level function: First validates, and then extracts results
def process_images(image_files, template_file, debug=False):
    template = json.load(template_file)

    if not validate_template(template, debug=debug):
        return None

    flagged_images, dewarped_images, qr_ids = validate_images(image_files, template, debug=debug)
    if flagged_images:
        return None

    return extract_results(template, dewarped_images, qr_ids, debug=debug)

if __name__ == "__main__":
    image_files = [open(f"input/{name}", "rb") for name in ["image14.jpg"]]
    template_file = open("input/page2_template.json", "rb")

    all_results = process_images(image_files, template_file, debug=True)
    if all_results:
        build_excel(all_results, "output/responses.xlsx")
        json.dump(all_results, open("output/all_pages_results.json", "w"), indent=2)