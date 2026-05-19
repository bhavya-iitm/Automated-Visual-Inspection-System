# import cv2
# import numpy as np

# # ===============================
# # LOAD ORIGINAL IMAGE
# # ===============================
# img = cv2.imread("Images/references/ref_front.jpg")
# if img is None:
#     print("❌ Error loading image")
#     exit()

# h, w = img.shape[:2]
# print(f"Original image size: {w} x {h}")

# # ===============================
# # DISPLAY SCALE (CHANGE IF NEEDED)
# # ===============================
# DISPLAY_SCALE = 0.4   # 0.3–0.5 works well for big images

# display = cv2.resize(
#     img,
#     None,
#     fx=DISPLAY_SCALE,
#     fy=DISPLAY_SCALE,
#     interpolation=cv2.INTER_AREA
# )

# display_clone = display.copy()
# points_display = []     # points clicked on display image
# points_original = []    # mapped to original image

# # ===============================
# # MOUSE CALLBACK
# # ===============================
# def mouse_callback(event, x, y, flags, param):
#     global display_clone

#     if event == cv2.EVENT_LBUTTONDOWN:
#         points_display.append((x, y))

#         # Map back to original image coordinates
#         ox = int(x / DISPLAY_SCALE)
#         oy = int(y / DISPLAY_SCALE)
#         points_original.append((ox, oy))

#         # Draw on display image
#         cv2.circle(display_clone, (x, y), 4, (0, 0, 255), -1)

#         if len(points_display) > 1:
#             cv2.line(
#                 display_clone,
#                 points_display[-2],
#                 points_display[-1],
#                 (0, 255, 0),
#                 2
#             )

#         cv2.imshow("Draw ROI (ENTER = finish)", display_clone)

# # ===============================
# # SHOW WINDOW
# # ===============================
# cv2.namedWindow("Draw ROI (ENTER = finish)", cv2.WINDOW_NORMAL)
# cv2.imshow("Draw ROI (ENTER = finish)", display_clone)
# cv2.setMouseCallback("Draw ROI (ENTER = finish)", mouse_callback)

# print("🟢 Draw ROI points with mouse. Press ENTER when done.")

# while True:
#     key = cv2.waitKey(1)
#     if key == 13:  # ENTER key
#         break

# cv2.destroyAllWindows()

# # ===============================
# # CREATE ROI MASK (ORIGINAL IMAGE SIZE)
# # ===============================
# if len(points_original) < 3:
#     print("❌ Not enough points to form ROI")
#     exit()

# roi_mask = np.zeros((h, w), dtype=np.uint8)
# pts = np.array(points_original, dtype=np.int32)

# cv2.fillPoly(roi_mask, [pts], 255)

# # ===============================
# # SAVE MASK
# # ===============================
# cv2.imwrite("Images/roi_masks/device_roi_mask_ref.jpg", roi_mask)
# print("✅ ROI mask saved as Images/roi_masks/device_roi_mask_ref.jpg")

# # ===============================
# # FINAL VISUAL CONFIRMATION
# # ===============================
# overlay = img.copy()
# cv2.polylines(overlay, [pts], True, (0, 255, 0), 3)

# # Resize overlay ONLY for viewing
# overlay_display = cv2.resize(
#     overlay,
#     None,
#     fx=DISPLAY_SCALE,
#     fy=DISPLAY_SCALE,
#     interpolation=cv2.INTER_AREA
# )

# cv2.imshow("Final ROI (visual check)", overlay_display)
# cv2.imshow(
#     "ROI Mask (scaled view)",
#     cv2.resize(roi_mask, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
# )

# cv2.waitKey(0)
# cv2.destroyAllWindows()





# ===============================
# Color Test Script
# ===============================

import cv2
import numpy as np

# ===============================
# PATHS
# ===============================
ref_path  = r"Images/references/ref_front.jpg"          # reference image
test_path = r"Images/tests/test_front.jpg"          # test image
roi_mask_path = r"Images/roi_masks/device_roi_mask_ref.jpg"

# ===============================
# LOAD IMAGES
# ===============================
ref = cv2.imread(ref_path)
test = cv2.imread(test_path)
roi_mask = cv2.imread(roi_mask_path, cv2.IMREAD_GRAYSCALE)

if ref is None or test is None or roi_mask is None:
    print("❌ Error loading images or ROI mask")
    exit()

# Ensure same size for test and reference
if ref.shape != test.shape:
    test = cv2.resize(test, (ref.shape[1], ref.shape[0]))

# Ensure ROI mask matches reference image size
if roi_mask.shape[:2] != ref.shape[:2]:
    roi_mask = cv2.resize(roi_mask, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_NEAREST)

h, w = ref.shape[:2]
print(f"Image size: {w} x {h}")

# ===============================
# SANITY CHECK ROI
# ===============================
roi_pixels = np.sum(roi_mask == 255)
roi_percent = roi_pixels / roi_mask.size * 100

print(f"ROI covers {roi_percent:.2f}% of image")

if roi_pixels < 1000:
    print("❌ ROI too small – something is wrong")
    exit()

# ===============================
# CONVERT TO LAB (OpenCV LAB)
# ===============================
ref_lab  = cv2.cvtColor(ref,  cv2.COLOR_BGR2LAB)
test_lab = cv2.cvtColor(test, cv2.COLOR_BGR2LAB)

# ===============================
# EXTRACT DEVICE PIXELS ONLY
# ===============================
ref_pixels  = ref_lab[roi_mask == 255]
test_pixels = test_lab[roi_mask == 255]

# Mean LAB in OpenCV scale
ref_mean  = ref_pixels.mean(axis=0)
test_mean = test_pixels.mean(axis=0)

# ===============================
# OpenCV LAB → REAL CIE LAB
# ===============================
def opencv_to_cielab(lab):
    L = lab[0] * 100.0 / 255.0
    a = lab[1] - 128.0
    b = lab[2] - 128.0
    return np.array([L, a, b])

ref_lab_real  = opencv_to_cielab(ref_mean)
test_lab_real = opencv_to_cielab(test_mean)

# ===============================
# ΔE (CIE76)
# ===============================
delta_e = np.linalg.norm(ref_lab_real - test_lab_real)

# ===============================
# RESULT
# ===============================
DELTA_E_LIMIT = 2.0   # adjust if needed

print("\n========== COLOR RESULT ==========")
print(f"Reference LAB : {ref_lab_real}")
print(f"Test LAB      : {test_lab_real}")
print(f"ΔE Value      : {delta_e:.2f}")

if delta_e <= DELTA_E_LIMIT:
    print("✅ PASS : Color within tolerance")
else:
    print("❌ FAIL : Color mismatch")

COLOR_DEFECT = delta_e > DELTA_E_LIMIT
# ===============================
# VISUAL CONFIRMATION (DISPLAY ONLY)
# ===============================
DISPLAY_SCALE = 0.4  # change if needed

# ---- Reference image with ROI ----
ref_overlay = ref.copy()
contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(ref_overlay, contours, -1, (0, 255, 0), 2)

ref_disp = cv2.resize(
    ref_overlay, None,
    fx=DISPLAY_SCALE, fy=DISPLAY_SCALE,
    interpolation=cv2.INTER_AREA
)

# ---- Test image with ROI ----
test_overlay = test.copy()
cv2.drawContours(test_overlay, contours, -1, (0, 255, 0), 2)

test_disp = cv2.resize(
    test_overlay, None,
    fx=DISPLAY_SCALE, fy=DISPLAY_SCALE,
    interpolation=cv2.INTER_AREA
)

# ---- ROI mask ----
mask_disp = cv2.resize(
    roi_mask, None,
    fx=DISPLAY_SCALE, fy=DISPLAY_SCALE,
    interpolation=cv2.INTER_NEAREST
)

# ---- Show all windows together ----
cv2.imshow("Test Image + ROI", test_disp)
cv2.imshow("ROI Mask", mask_disp)

cv2.waitKey(0)
cv2.destroyAllWindows()




# import cv2
# import numpy as np

# # ===============================
# # PATHS
# # ===============================
# TEST_IMAGE_PATH = "Images/references/ref_front.jpg"
# ROI_MASK_PATH = "Images/roi_masks/device_roi_mask_ref.jpg"

# # ===============================
# # PARAMETERS (TUNED FOR YOUR PART)
# # ===============================
# DARK_THRESHOLD = 185        # higher = detect lighter paint loss
# MIN_DEFECT_AREA = 30        # smallest valid paint chip
# MAX_DEFECT_AREA = 4000      # reject large regions (edges/shadows)

# # ===============================
# # LOAD IMAGE & ROI MASK
# # ===============================
# img = cv2.imread(TEST_IMAGE_PATH)
# roi_mask = cv2.imread(ROI_MASK_PATH, cv2.IMREAD_GRAYSCALE)

# roi_mask = roi_mask.astype(np.uint8)
# _, roi_mask = cv2.threshold(roi_mask, 127, 255, cv2.THRESH_BINARY)

# if img is None or roi_mask is None:
#     print("❌ Error loading image or ROI mask")
#     exit()

# # ===============================
# # ENSURE ROI MASK MATCHES IMAGE
# # ===============================
# if roi_mask.shape[:2] != img.shape[:2]:
#     roi_mask = cv2.resize(
#         roi_mask,
#         (img.shape[1], img.shape[0]),
#         interpolation=cv2.INTER_NEAREST
#     )

# # Force binary uint8 mask
# roi_mask = roi_mask.astype(np.uint8)
# _, roi_mask = cv2.threshold(roi_mask, 127, 255, cv2.THRESH_BINARY)

# # ===============================
# # APPLY ROI
# # ===============================
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# gray = cv2.GaussianBlur(gray, (5, 5), 0)

# gray = cv2.bitwise_and(gray, gray, mask=roi_mask)

# # ===============================
# # DARK PAINT LOSS DETECTION
# # ===============================
# _, paint_mask = cv2.threshold(
#     gray,
#     DARK_THRESHOLD,
#     255,
#     cv2.THRESH_BINARY_INV
# )

# # Keep detection strictly inside ROI
# paint_mask = cv2.bitwise_and(paint_mask, paint_mask, mask=roi_mask)

# # ===============================
# # MORPHOLOGY (CRITICAL)
# # ===============================
# kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
# kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

# paint_mask = cv2.morphologyEx(paint_mask, cv2.MORPH_OPEN, kernel_open)
# paint_mask = cv2.morphologyEx(paint_mask, cv2.MORPH_CLOSE, kernel_close)

# # ===============================
# # CONTOUR ANALYSIS
# # ===============================
# contours, _ = cv2.findContours(
#     paint_mask,
#     cv2.RETR_EXTERNAL,
#     cv2.CHAIN_APPROX_SIMPLE
# )

# paint_removed = False
# all_boxes = []

# for cnt in contours:
#     area = cv2.contourArea(cnt)

#     if area < MIN_DEFECT_AREA or area > MAX_DEFECT_AREA:
#         continue

#     paint_removed = True
#     x, y, w, h = cv2.boundingRect(cnt)
#     all_boxes.append((x, y, w, h))

#     # Draw individual defect boxes
#     cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

# # ===============================
# # OPTIONAL: DRAW ONE BIG BOX
# # ===============================
# if paint_removed and len(all_boxes) > 1:
#     xs = [x for x, y, w, h in all_boxes]
#     ys = [y for x, y, w, h in all_boxes]
#     xe = [x + w for x, y, w, h in all_boxes]
#     ye = [y + h for x, y, w, h in all_boxes]

#     cv2.rectangle(
#         img,
#         (min(xs), min(ys)),
#         (max(xe), max(ye)),
#         (0, 0, 255),
#         2
#     )

# # ===============================
# # FINAL STATUS
# # ===============================
# status = "PAINT REMOVED" if paint_removed else "NO PAINT DEFECT"
# color = (0, 0, 255) if paint_removed else (0, 255, 0)

# cv2.putText(
#     img,
#     status,
#     (30, 50),
#     cv2.FONT_HERSHEY_SIMPLEX,
#     1.2,
#     color,
#     3
# )
# # ===============================
# # VISUAL CHECK: ROI HIGHLIGHTED
# # ===============================

# overlay = img.copy()
# overlay[roi_mask == 255] = (0, 0, 255)

# vis = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)

# cv2.imshow("TEST IMAGE + ROI (RED)", vis)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # ===============================
# # DISPLAY RESULTS
# # ===============================
# cv2.imshow("ROI Mask (Should be TOP ONLY)", roi_mask)
# cv2.imshow("Paint Mask (Final)", paint_mask)
# cv2.imshow("Final Result", img)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

# print("PAINT REMOVED:", paint_removed)





# import cv2
# import numpy as np

# # ===============================
# # PATHS
# # ===============================
# TEST_IMAGE_PATH = "Images/tests/test_front.jpg"
# ROI_MASK_PATH   = "Images/roi_masks/device_roi_mask_ref.jpg"

# # ===============================
# # PARAMETERS
# # ===============================
# DARK_THRESHOLD = 185
# MIN_DEFECT_AREA = 40
# MAX_DEFECT_AREA = 5000

# # ===============================
# # LOAD
# # ===============================
# img = cv2.imread(TEST_IMAGE_PATH)
# roi_mask = cv2.imread(ROI_MASK_PATH, cv2.IMREAD_GRAYSCALE)

# if img is None or roi_mask is None:
#     print("❌ Image / ROI load error")
#     exit()

# # Resize mask if needed
# if roi_mask.shape[:2] != img.shape[:2]:
#     roi_mask = cv2.resize(
#         roi_mask,
#         (img.shape[1], img.shape[0]),
#         interpolation=cv2.INTER_NEAREST
#     )

# # Binary mask
# _, roi_mask = cv2.threshold(roi_mask, 127, 255, cv2.THRESH_BINARY)

# # ===============================
# # PROCESS FULL IMAGE FIRST
# # ===============================
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# gray = cv2.GaussianBlur(gray, (5, 5), 0)

# _, paint_mask = cv2.threshold(
#     gray,
#     DARK_THRESHOLD,
#     255,
#     cv2.THRESH_BINARY_INV
# )

# # ===============================
# # APPLY ROI (THIS IS THE KEY)
# # ===============================
# paint_mask = cv2.bitwise_and(paint_mask, paint_mask, mask=roi_mask)

# # ===============================
# # MORPHOLOGY
# # ===============================
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
# paint_mask = cv2.morphologyEx(paint_mask, cv2.MORPH_OPEN, kernel)

# # ===============================
# # CONTOUR CHECK
# # ===============================
# contours, _ = cv2.findContours(
#     paint_mask,
#     cv2.RETR_EXTERNAL,
#     cv2.CHAIN_APPROX_SIMPLE
# )

# paint_removed = False

# for cnt in contours:
#     area = cv2.contourArea(cnt)
#     if MIN_DEFECT_AREA < area < MAX_DEFECT_AREA:
#         paint_removed = True
#         x,y,w,h = cv2.boundingRect(cnt)
#         cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)

# # ===============================
# # RESULT
# # ===============================
# PAINT_DEFECT = paint_removed

# status = "PAINT REMOVED" if PAINT_DEFECT else "NO PAINT DEFECT"
# color  = (0,0,255) if PAINT_DEFECT else (0,255,0)

# cv2.putText(img, status, (30,50),
#             cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

# # ===============================
# # VISUAL ROI CONFIRMATION
# # ===============================
# overlay = img.copy()
# overlay[roi_mask == 255] = (0,255,0)

# vis = cv2.addWeighted(img, 0.8, overlay, 0.2, 0)

# cv2.imshow("PAINT CHECK (ROI BASED)", vis)
# cv2.imshow("PAINT MASK", paint_mask)
# cv2.imshow("ROI MASK", roi_mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# print("PAINT REMOVED:", PAINT_DEFECT)






# # ===============================
# # Scratch test script
# # ===============================

# import cv2
# import numpy as np

# # ===============================
# # PATHS
# # ===============================
# REF_IMAGE_PATH = "Images/references/ref_front.jpg"
# TEST_IMAGE_PATH = "Images/tests/test_front.jpg"
# ROI_MASK_PATH  = "Images/roi_masks/device_roi_mask_ref.jpg"

# # ===============================
# # PARAMETERS
# # ===============================

# # Adjusted thresholds to reduce false positives
# BLOCK_SIZE = 16          # pixel block size
# VAR_RATIO_THRESHOLD = 1.7
# MIN_DISTURBED_BLOCKS = 10

# # ===============================
# # LOAD
# # ===============================
# ref = cv2.imread(REF_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
# test = cv2.imread(TEST_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
# roi_mask = cv2.imread(ROI_MASK_PATH, cv2.IMREAD_GRAYSCALE)

# if ref is None or test is None or roi_mask is None:
#     print("Error loading files")
#     exit()
# if ref.shape != test.shape:
#     test = cv2.resize(test, (ref.shape[1], ref.shape[0]))
# if roi_mask.shape[:2] != ref.shape[:2]:
#     roi_mask = cv2.resize(roi_mask, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_NEAREST)

# # ===============================
# # SMOOTH (important)
# # ===============================
# ref = cv2.GaussianBlur(ref, (5,5), 0)
# test = cv2.GaussianBlur(test, (5,5), 0)

# h, w = ref.shape
# disturbance_map = np.zeros_like(ref, dtype=np.uint8)

# disturbed_blocks = 0

# # ===============================
# # BLOCK-WISE VARIANCE
# # ===============================
# for y in range(0, h - BLOCK_SIZE, BLOCK_SIZE):
#     for x in range(0, w - BLOCK_SIZE, BLOCK_SIZE):

#         block_mask = roi_mask[y:y+BLOCK_SIZE, x:x+BLOCK_SIZE]
#         if cv2.countNonZero(block_mask) < 0.5 * BLOCK_SIZE * BLOCK_SIZE:
#             continue

#         ref_block = ref[y:y+BLOCK_SIZE, x:x+BLOCK_SIZE]
#         test_block = test[y:y+BLOCK_SIZE, x:x+BLOCK_SIZE]

#         var_ref = np.var(ref_block)
#         var_test = np.var(test_block)

#         if (var_test - var_ref) > 20:
#             disturbed_blocks += 1
#             disturbance_map[y:y+BLOCK_SIZE, x:x+BLOCK_SIZE] = 255
# MIN_DISTURBED_BLOCKS = 40
# disturbance_map = cv2.bitwise_and(
#     disturbance_map,
#     disturbance_map,
#     mask=roi_mask
# )

# # Debug: print number of disturbed blocks
# print(f"Disturbed blocks flagged: {disturbed_blocks}")

# # ===============================
# # FINAL DECISION
# # ===============================
# scratch_detected = disturbed_blocks >= MIN_DISTURBED_BLOCKS

# # ===============================
# # VISUALIZATION
# # ===============================
# vis = cv2.cvtColor(test, cv2.COLOR_GRAY2BGR)
# vis[disturbance_map > 0] = (0, 0, 255)

# status = "SCRATCH DETECTED" if scratch_detected else "NO SCRATCH"
# color  = (0,0,255) if scratch_detected else (0,255,0)

# cv2.putText(vis, status, (30,50),
#             cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

# # ===============================
# # DISPLAY
# # ===============================

# test_disp = cv2.bitwise_and(test, test, mask=roi_mask)
# cv2.imshow("Test Image + ROI", test_disp)
# cv2.imshow("Disturbance Map", disturbance_map)
# # Save disturbance map for debugging
# cv2.imwrite("debug_disturbance_map.jpg", disturbance_map)
# cv2.imshow("Final Result", vis)

# cv2.waitKey(0)
# cv2.destroyAllWindows()


# print("Disturbed blocks:", disturbed_blocks)
# print("SCRATCH DETECTED:", scratch_detected)

# SCRATCH_DEFECT = scratch_detected


# # ===============================
# # FINAL VISUAL CONFIRMATION
# # ===============================

# if COLOR_DEFECT or PAINT_DEFECT or SCRATCH_DEFECT:
#     FINAL_RESULT = "FAIL"
#     result_color = (0, 0, 255)
# else:
#     FINAL_RESULT = "PASS"
#     result_color = (0, 255, 0)

# final_img = cv2.imread(TEST_IMAGE_PATH)  # reuse test image safely

# cv2.putText(
#     final_img,
#     FINAL_RESULT,
#     (40, 60),
#     cv2.FONT_HERSHEY_SIMPLEX,
#     1.5,
#     result_color,
#     3
# )

# cv2.imshow("FINAL INSPECTION RESULT", final_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# print("\n========== FINAL SUMMARY ==========")
# print("COLOR DEFECT  :", COLOR_DEFECT)
# print("PAINT DEFECT  :", PAINT_DEFECT)
# print("SCRATCH DEFECT:", SCRATCH_DEFECT)
# print("FINAL RESULT  :", FINAL_RESULT)



