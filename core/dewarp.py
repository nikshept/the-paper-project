"""
Piece 1 of detect.py: dewarp a photo/scan to the flat template coordinate
space, self-correcting for orientation using the QR code as a ground-truth
check (rather than trying to hand-diagnose why a given photo came out wrong).

QR decoding tries three detectors in order of cost: cv2's built-in
detector (free, fast, handles most photos), then cv2's WeChatQRCode
detector (needs the 4 small model files in WECHAT_MODEL_DIR, but is far
more robust on small/blurry/low-contrast codes -- no extra system
dependency beyond opencv-contrib-python), then pyzbar as an optional
last resort if it happens to be installed and working.

Requires: opencv-contrib-python, pymupdf. pyzbar is optional.
"""
import os
import cv2
import numpy as np

try:
    from pyzbar.pyzbar import decode as zbar_decode
    _HAVE_ZBAR = True
except (ImportError, OSError):
    _HAVE_ZBAR = False  # pyzbar's native DLL can fail to load on some Windows setups;
                         # the WeChat detector below covers essentially everything anyway.

SCALE = 3  # pixels per PDF point in the dewarped output

ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

# Tuned for the actual use case (small printed markers, photographed by
# hand under normal room lighting/focus) rather than left at OpenCV's
# defaults, which assume more controlled conditions. Each change widens
# tolerance for a specific, well-understood real-world failure mode --
# none of this is tuned to any one photo:
_ARUCO_PARAMS = cv2.aruco.DetectorParameters()
# Wider adaptive-threshold window search: the default (3-23, step 10) is
# a narrow range tuned for fairly uniform lighting. A phone photo of a
# page often has real illumination gradients across it (window light,
# shadow from the phone/hand, uneven room lighting) -- searching a wider
# range of local threshold windows makes binarization succeed across
# more of the image before marker-candidate detection even runs.
_ARUCO_PARAMS.adaptiveThreshWinSizeMin = 3
_ARUCO_PARAMS.adaptiveThreshWinSizeMax = 53
_ARUCO_PARAMS.adaptiveThreshWinSizeStep = 4
# Wider accepted marker-size range (as a fraction of image perimeter):
# defaults (0.03-4.0) assume the marker takes up a fairly predictable
# fraction of the frame. A real photo's framing varies a lot -- someone
# photographing from farther back (more surrounding table in frame) or
# very close up both shift this fraction well outside the default
# window, causing correctly-drawn markers to be rejected before decode
# is even attempted.
_ARUCO_PARAMS.minMarkerPerimeterRate = 0.01
_ARUCO_PARAMS.maxMarkerPerimeterRate = 8.0
# More lenient polygon-fit tolerance: JPEG compression and slight motion
# blur round off what should be sharp marker corners; the default
# accuracy rate is tuned for clean synthetic/scanned input.
_ARUCO_PARAMS.polygonalApproxAccuracyRate = 0.05
# Subpixel corner refinement: default is no refinement at all. This
# directly improves homography accuracy (better corner localization ->
# better warp), which matters more for a handheld photo than a flatbed
# scan.
_ARUCO_PARAMS.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
# Higher bit-error tolerance during ID decoding: default (0.6) is tuned
# for clean input; real photo noise/compression can flip a bit or two
# in the marker's data cells without the mark itself being ambiguous to
# a human eye.
_ARUCO_PARAMS.errorCorrectionRate = 0.8

DETECTOR = cv2.aruco.ArucoDetector(ARUCO_DICT, _ARUCO_PARAMS)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WECHAT_MODEL_DIR = os.path.join(SCRIPT_DIR, "assets", "wechat_qr_models")


def get_true_marker_positions(blank_ref_pdf_path):
    """Detect the 4 ArUco markers directly in a rendering of the blank
    reference PDF, for each page. Returns {page_num: {marker_id: (px, py)}}
    in the SAME top-down pixel space cv2.warpPerspective needs -- no
    point-space round trip, so there's no chance of a top-down/bottom-up
    mismatch creeping back in."""
    import pymupdf as fitz
    doc = fitz.open(blank_ref_pdf_path)
    page_h = doc[0].rect.height
    truth = {}
    for pno, page in enumerate(doc):
        pix = page.get_pixmap(matrix=fitz.Matrix(SCALE, SCALE))
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if pix.n >= 3 else img
        corners, ids, _ = DETECTOR.detectMarkers(gray)
        if ids is None or len(ids) != 4:
            raise RuntimeError(f"Expected 4 markers in blank reference page {pno}, found {0 if ids is None else len(ids)}")
        page_markers = {}
        for i, mid in enumerate(ids.flatten()):
            cx, cy = corners[i][0].mean(axis=0)
            page_markers[int(mid)] = (float(cx), float(cy))  # top-down pixel space, as-is
        truth[pno] = page_markers
    return truth, doc[0].rect.width, page_h


def _try_dewarp(photo, truth_for_page, page_w, page_h):
    """One attempt: detect markers in `photo` as given (no flipping),
    compute homography, warp. Returns None if fewer than 4 markers found."""
    gray = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = DETECTOR.detectMarkers(gray)
    if ids is None or len(ids) < 4:
        return None

    img_pts, pdf_pts = [], []
    for i, mid in enumerate(ids.flatten()):
        if int(mid) not in truth_for_page:
            continue
        img_pts.append(corners[i][0].mean(axis=0))
        pdf_pts.append(truth_for_page[int(mid)])
    if len(img_pts) < 4:
        return None

    img_pts = np.array(img_pts, dtype=np.float32)
    pdf_pts = np.array(pdf_pts, dtype=np.float32)  # already in pixel space -- no rescale needed
    H, _ = cv2.findHomography(img_pts, pdf_pts, method=0)
    if H is None:
        return None
    out_w, out_h = int(page_w * SCALE), int(page_h * SCALE)
    return cv2.warpPerspective(photo, H, (out_w, out_h), borderValue=(255, 255, 255))


def _find_qr_region(flat_img, qr_center_frac=(0.5, 0.965), qr_size_frac=0.09):
    """Crop a generous region around where the QR should be (bottom-center
    of the canonical page), so pyzbar has a better chance than scanning the
    whole page. Fractions are relative to image width/height."""
    h, w = flat_img.shape[:2]
    cx, cy = int(w * qr_center_frac[0]), int(h * qr_center_frac[1])
    half = int(w * qr_size_frac)
    y0, y1 = max(0, cy - half), min(h, cy + half)
    x0, x1 = max(0, cx - half), min(w, cx + half)
    return flat_img[y0:y1, x0:x1]


_CV2_QR = cv2.QRCodeDetector()

_WECHAT_QR = None
_WECHAT_TRIED = False


def _get_wechat_detector():
    """Lazy-load the WeChat QR detector. cv2 5.0+ changed the API to a
    zero-argument constructor with models bundled internally; older cv2
    (4.x via opencv-contrib-python) needs the 4 external model files.
    Try the new API first, fall back to the old one, degrade gracefully
    to None if neither works rather than crashing the pipeline."""
    global _WECHAT_QR, _WECHAT_TRIED
    if _WECHAT_TRIED:
        return _WECHAT_QR
    _WECHAT_TRIED = True

    try:
        _WECHAT_QR = cv2.wechat_qrcode_WeChatQRCode()  # cv2 5.0+: no files needed
        return _WECHAT_QR
    except Exception:
        pass

    try:
        paths = [os.path.join(WECHAT_MODEL_DIR, f) for f in
                 ("detect.prototxt", "detect.caffemodel", "sr.prototxt", "sr.caffemodel")]
        if all(os.path.exists(p) for p in paths):
            _WECHAT_QR = cv2.wechat_qrcode_WeChatQRCode(*paths)  # cv2 4.x
    except Exception:
        _WECHAT_QR = None
    return _WECHAT_QR


def _decode_qr(flat_img):
    """Try cv2's built-in detector first (cheapest), then the WeChat
    detector (handles most of what the built-in one misses), then pyzbar
    if it happens to be available. Each tried cropped-first, then full-page."""
    crop = _find_qr_region(flat_img)

    data, _, _ = _CV2_QR.detectAndDecode(crop)
    if data:
        return data
    data, _, _ = _CV2_QR.detectAndDecode(flat_img)
    if data:
        return data

    wechat = _get_wechat_detector()
    if wechat is not None:
        results, _ = wechat.detectAndDecode(crop)
        if results:
            return results[0]
        results, _ = wechat.detectAndDecode(flat_img)
        if results:
            return results[0]

    if _HAVE_ZBAR:
        results = zbar_decode(crop)
        if results:
            return results[0].data.decode("utf-8")
        results = zbar_decode(flat_img)
        if results:
            return results[0].data.decode("utf-8")

    return None


def dewarp_and_identify(photo_path, truth, page_w, page_h, expected_page=None):
    """Main entry point. Tries the photo as-is, then horizontal flip,
    vertical flip, and 180-rotation, using successful QR decode as the
    signal that we found the right orientation. Returns
    (flat_image, qr_data, page_num) or raises if nothing worked."""
    original = cv2.imread(photo_path)
    if original is None:
        raise FileNotFoundError(photo_path)

    candidates = {
        "as-is": original,
        "horizontal-flip": cv2.flip(original, 1),
        "vertical-flip": cv2.flip(original, 0),
        "180-rotation": cv2.flip(original, -1),
    }

    for label, candidate in candidates.items():
        for page_num, truth_for_page in truth.items():
            flat = _try_dewarp(candidate, truth_for_page, page_w, page_h)
            if flat is None:
                continue
            qr_data = _decode_qr(flat)
            if qr_data:
                actual_page = int(qr_data.rsplit("-P", 1)[-1]) if "-P" in qr_data else page_num
                return flat, qr_data, actual_page, label

    raise RuntimeError(f"Could not dewarp {photo_path}: no orientation produced a decodable QR")


if __name__ == "__main__":
    import sys, os, glob

    if len(sys.argv) < 3:
        print("Usage: python dewarp.py <blank_reference.pdf> <photo1.jpg> [photo2.jpg ...]")
        print("       python dewarp.py <blank_reference.pdf> <folder_of_photos>")
        sys.exit(1)

    blank_pdf = sys.argv[1]
    raw_args = sys.argv[2:]

    photos = []
    for arg in raw_args:
        if os.path.isdir(arg):
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                photos.extend(sorted(glob.glob(os.path.join(arg, ext))))
        else:
            photos.append(arg)

    if not photos:
        print(f"No photos found in: {raw_args}")
        sys.exit(1)

    truth, page_w, page_h = get_true_marker_positions(blank_pdf)
    print(f"Loaded true marker positions from {blank_pdf} ({len(truth)} pages)")
    print(f"Found {len(photos)} photo(s) to process")

    out_dir = os.path.join("output", "dewarped_pics")
    os.makedirs(out_dir, exist_ok=True)

    for photo_path in photos:
        try:
            flat, qr_data, page_num, method = dewarp_and_identify(photo_path, truth, page_w, page_h)
            tracking_code = qr_data.split("-")[2] if qr_data.count("-") >= 2 else qr_data
            out_path = os.path.join(out_dir, f"{tracking_code}_P{page_num}.png")
            cv2.imwrite(out_path, flat)
            print(f"{photo_path}: OK  qr={qr_data!r}  page={page_num}  orientation_fix={method}  -> {out_path}")
        except Exception as e:
            print(f"{photo_path}: FAILED  ({e})")