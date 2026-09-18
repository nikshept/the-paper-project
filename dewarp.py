import numpy as np
import cv2

### Detect aruco markers in a greyscale image
def detect_markers(gray_image):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(aruco_dict)
    return detector.detectMarkers(gray_image)

### Draws borders around identified markers and rejects
def draw_markers(image, corners, rejected):
    annotated = image.copy()
    for c in corners:
        pts = c[0].astype(int)
        cv2.polylines(annotated, [pts], isClosed=True, color=(0, 255, 0), thickness=4)
    cv2.aruco.drawDetectedMarkers(annotated, rejected, borderColor=(0, 0, 255))
    return annotated

### Warps image flat into A4 ratio using 4 marker centers as corners
def dewarp_image_A4(image, target_w=2000, target_h=None, margin=80, debug=False):
    # Pre process the image input
    file_bytes = np.frombuffer(image.read(), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # detect the markers
    corners, ids, rejected = detect_markers(gray)
    if debug:
        print(f"Detected {len(corners)} marker(s), IDs: {ids.flatten().tolist() if ids is not None else 'none'}")
        annotated = draw_markers(img, corners, rejected)
        cv2.imwrite("output/aruco_image.jpg", annotated)

    if ids is None or len(ids) != 4:
        return None

    # Figure out A4 dimensions and corner locations
    if target_h is None:
        target_h = int(target_w / (210 / 297))

    centers = np.array([c[0].mean(axis=0) for c in corners])  # one center point per marker

    sums = centers[:, 0] + centers[:, 1]   # x+y: smallest = top-left, largest = bottom-right
    diffs = centers[:, 0] - centers[:, 1]  # x-y: largest = top-right, smallest = bottom-left

    top_left = centers[np.argmin(sums)]
    bottom_right = centers[np.argmax(sums)]
    top_right = centers[np.argmax(diffs)]
    bottom_left = centers[np.argmin(diffs)]

    # Dewarp the image and fit it into the corners and dimensions
    src = np.float32([top_left, top_right, bottom_right, bottom_left])
    dst = np.float32([[margin, margin], [target_w - margin, margin],
                       [target_w - margin, target_h - margin], [margin, target_h - margin]])

    homography = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, homography, (target_w, target_h))

if __name__ == "__main__":
    with open("input/image14.jpg", "rb") as f:
        flat = dewarp_image_A4(f, debug=True)
    if flat is None:
        print("Insufficient markers detected.")
    else:
        cv2.imwrite("output/dewarped_image.jpg", flat)
        print("Saved to output folder.")