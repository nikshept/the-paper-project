import pymupdf
import numpy as np
import cv2
import streamlit as st

st.title("ArUco Marker Test")

img = st.file_uploader("Upload a scanned photo", type=["jpg", "png"])

if img is not None:
    file_bytes = np.frombuffer(img.read(), dtype=np.uint8)  # raw file bytes -> byte array
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # byte array -> actual BGR pixel array, same as cv2.imread would give

    #if debug:
        #st.image(img, channels="BGR")

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)  # same dictionary used to GENERATE your markers earlier -- must match, or nothing will detect
    detector = cv2.aruco.ArucoDetector(aruco_dict)  # bundles the dictionary + default detection parameters

    corners, ids, rejected = detector.detectMarkers(imgGray)
    # corners: list of 4 (x,y) points per detected marker
    # ids: which marker ID each one is (0-3, matching your corner assignment)
    # rejected: candidate shapes that looked marker-like but didn't decode -- useful for debugging later, ignore for now

    st.write(f"Detected {len(corners)} marker(s), IDs: {ids.flatten().tolist() if ids is not None else 'none'}")

    annotated = img.copy()
    for c in corners:
        pts = c[0].astype(int)  # this marker's 4 detected corner points
        cv2.polylines(annotated, [pts], isClosed=True, color=(0, 255, 0), thickness=4)

    cv2.aruco.drawDetectedMarkers(annotated, rejected, borderColor=(0, 0, 255))  
    st.image(annotated, channels="BGR")

    ids = ids.flatten()
    id_to_center = {i: c[0].mean(axis=0) for i, c in zip(ids, corners)}

    target_w = 800
    target_h = int(target_w / (210/297))
    margin = 35  

    src = np.float32([id_to_center[0], id_to_center[1], id_to_center[2], id_to_center[3]])
    dst = np.float32([[margin, margin], [target_w-margin, margin],
                       [target_w-margin, target_h-margin], [margin, target_h-margin]])

    homography = cv2.getPerspectiveTransform(src, dst)
    flat = cv2.warpPerspective(img, homography, (target_w, target_h))
    st.image(flat, channels="BGR")