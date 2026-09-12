import numpy as np
import cv2

def group_rows(shapes, get_y, y_gap=20):
	"""Groups shapes into rows by y-position. shapes must already be
	sorted top-to-bottom. get_y(shape) pulls the y-coordinate --
	different for a contour (boundingRect) vs a Hough circle (c[1])."""
	rows = [[shapes[0]]]
	for s in shapes[1:]:
		if get_y(s) - get_y(rows[-1][-1]) > y_gap:
			rows.append([])
		rows[-1].append(s)
	return rows


def sort_left_to_right(row, source):
	if source == "contour":
		return sorted(row, key=lambda s: cv2.boundingRect(s)[0])
	return sorted(row, key=lambda s: s[0])


def draw_shape(img, shape, source, color, thickness):
	if source == "contour":
		cv2.drawContours(img, [shape], -1, color, thickness)
	else:
		cv2.circle(img, (shape[0], shape[1]), shape[2], color, thickness)


def score_row(counts):
	"""Ranks a row's ink counts, returns (bubbled_idx, gap_ratio).
	gap_ratio is the top-vs-runner-up gap as a FRACTION of top, not a
	raw pixel difference -- raw counts scale with bubble area (radius^2),
	so a ratio stays meaningful across different radius/resolution."""
	ranked = sorted(counts, reverse=True)
	top = ranked[0]
	gap_ratio = (ranked[0] - ranked[1]) / top if top > 0 else 0
	return counts.index(top), gap_ratio


def get_Responses_Contour(image, bperRow, tag="", debug=False):
	MIN_GAP_RATIO = 0.3
	if debug:
		cv2.imwrite(f"output/0_contour_input_{tag}.jpg", image)

	imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	imgThresh = cv2.threshold(imgGray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

	if debug:
		cv2.imwrite(f"output/1_contour_thresh_{tag}.jpg", imgThresh)

	cnts, hierarchy = cv2.findContours(imgThresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	hierarchy = hierarchy[0]

	# Bubble size filter, proportional to this crop's width (605 = the
	# reference width these ratios were calibrated against).
	h_img, w_img = imgThresh.shape[:2]
	min_size, max_size = int(w_img * 20/605), int(w_img * 80/605)

	bubble_shaped_idx = set()
	for i, c in enumerate(cnts):
		x, y, w, h = cv2.boundingRect(c)
		ar = w / float(h)
		if min_size <= w <= max_size and min_size <= h <= max_size and 0.9 <= ar <= 1.1:
			bubble_shaped_idx.add(i)

	# Keep only the OUTER ring of each bubble -- reject the one whose
	# parent contour is also bubble-shaped (that's the inner ring).
	questionCnts = [cnts[j] for j in bubble_shaped_idx if hierarchy[j][3] not in bubble_shaped_idx]

	imgQconts = image.copy()
	cv2.drawContours(imgQconts, questionCnts, -1, (0, 255, 0), 1)
	if debug:
		cv2.imwrite(f"output/3_contour_questionContours_{tag}.jpg", imgQconts)

	if not questionCnts:
		print("No bubble-shaped contours found. Check thresholding/size filter.")
		return [], []

	questionCnts = sorted(questionCnts, key=lambda c: cv2.boundingRect(c)[1])
	rows = group_rows(questionCnts, lambda c: cv2.boundingRect(c)[1])

	imgResult = image.copy()
	results = []

	for row in rows:
		row = sort_left_to_right(row, "contour")

		if len(row) != bperRow:
			cv2.drawContours(imgResult, row, -1, (0, 0, 255), 1)
			results.append((None, 0, True))
			continue

		counts = []
		for c in row:
			mask = np.zeros(imgThresh.shape, dtype="uint8")
			cv2.drawContours(mask, [c], -1, 255, -1)
			counts.append(cv2.countNonZero(cv2.bitwise_and(imgThresh, imgThresh, mask=mask)))

		bubbled_idx, gap_ratio = score_row(counts)

		# Row count matched bperRow -- but STILL needs a real gap_ratio to
		# count as found. Either kind of failure (wrong count, or right
		# count but no clear answer) falls through to hough via merge_responses.
		if gap_ratio >= MIN_GAP_RATIO:
			cv2.drawContours(imgResult, [row[bubbled_idx]], -1, (0, 255, 0), 2)
			results.append((bubbled_idx + 1, gap_ratio, False))
		else:
			cv2.drawContours(imgResult, row, -1, (0, 0, 255), 1)
			results.append((None, gap_ratio, True))

	if debug:
		cv2.imwrite(f"output/5_contour_result_{tag}.jpg", imgResult)
	return results, rows


def get_Responses_Hough(image, bperRow, tag="", debug=False):
	MIN_GAP_RATIO = 0.1

	if debug:
		cv2.imwrite(f"output/0_hough_input_{tag}.jpg", image)

	imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	imgThresh = cv2.threshold(imgGray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
	blurred = cv2.medianBlur(imgGray, 5)

	h, w = blurred.shape[:2]
	min_radius, max_radius, min_dist = int(w*20/650), int(w*40/650), int(w*50/650)

	circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=min_dist,
	                            param1=50, param2=50, minRadius=min_radius, maxRadius=max_radius)

	# None (not an empty array) when nothing is found -- circles[0,:] on
	# None crashes without this check.
	if circles is None:
		print("No circles found. Check radius/threshold calibration.")
		return [], []

	questionCnts = np.round(circles[0, :]).astype("int")  # (x, y, r) per circle
	imgQconts = image.copy()
	for (x, y, r) in questionCnts:
		cv2.circle(imgQconts, (x, y), r, (0, 255, 0), 1)

	if debug:
		cv2.imwrite(f"output/2_hough_questionCircles_{tag}.jpg", imgQconts)

	questionCnts = sorted(questionCnts, key=lambda c: c[1])
	rows = group_rows(questionCnts, lambda c: c[1])

	imgResult = image.copy()
	results = []

	for row in rows:
		row = sorted(row, key=lambda c: c[0])

		if len(row) != bperRow:
			for c in row:
				cv2.circle(imgResult, (c[0], c[1]), c[2], (0, 0, 255), 1)
			results.append((None, 0, True))
			continue

		counts = []
		for c in row:
			mask = np.zeros(imgThresh.shape, dtype="uint8")
			cv2.circle(mask, (c[0], c[1]), c[2], 255, -1)
			counts.append(cv2.countNonZero(cv2.bitwise_and(imgThresh, imgThresh, mask=mask)))

		bubbled_idx, gap_ratio = score_row(counts)

		# Hough is only ever the fallback -- still requires a real
		# gap_ratio, unlike contour which trusts a count match alone.
		if gap_ratio >= MIN_GAP_RATIO:
			c = row[bubbled_idx]
			cv2.circle(imgResult, (c[0], c[1]), c[2], (0, 255, 0), 2)
			results.append((bubbled_idx + 1, gap_ratio, False))
		else:
			for c in row:
				cv2.circle(imgResult, (c[0], c[1]), c[2], (0, 0, 255), 1)
			results.append((None, gap_ratio, True))

	if debug:
		cv2.imwrite(f"output/4_hough_result_{tag}.jpg", imgResult)
	return results, rows


def get_Responses(image, bperRow, tag="", debug=False):
	contour_answers, contour_rows = get_Responses_Contour(image, bperRow, tag, debug)
	hough_answers, hough_rows = get_Responses_Hough(image, bperRow, tag, debug)

	imgMerged = image.copy()
	merged = []
	n_rows = max(len(contour_answers), len(hough_answers))

	for i in range(n_rows):
		c = contour_answers[i] if i < len(contour_answers) else None
		h = hough_answers[i] if i < len(hough_answers) else None

		if c is not None and not c[2]:
			value, gap_ratio, flag, source, row = *c, "contour", contour_rows[i]
			color = (0, 255, 0)
		elif h is not None and not h[2]:
			value, gap_ratio, flag, source, row = *h, "hough", hough_rows[i]
			color = (0, 255, 255)
		else:
			value, gap_ratio, flag = None, 0, True
			source, row = ("contour", contour_rows[i]) if c is not None else ("hough", hough_rows[i] if h is not None else (None, None))
			color = (0, 0, 255)

		merged.append((value, gap_ratio, flag, source))
		if row is None:
			continue

		if flag:
			for shape in row:
				draw_shape(imgMerged, shape, source, color, 1)
		else:
			row_sorted = sort_left_to_right(row, source)
			draw_shape(imgMerged, row_sorted[value - 1], source, color, 2)

	if debug:
		cv2.imwrite(f"output/6_merged_result_{tag}.jpg", imgMerged)
	return merged, imgMerged


if __name__ == "__main__":
	test_img = cv2.imread("input/image11.jpg")
	print("merged:", get_Responses(test_img, 5, "test", debug=True)[0])
