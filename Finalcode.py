import cv2
import numpy as np
import os
# Set headless mode for server operation
HEADLESS_MODE = True  # Set to False for local testing with GUI


def create_roi_for_image(image_path, roi_name="roi_mask"):
    """Create ROI mask for any given image"""
   
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Error loading image: {image_path}")
        print(f"❌ Please check if the file exists and path is correct")
        return None
   
    h, w = img.shape[:2]
    print(f"Image size: {w} x {h}")
   
    # Display scale for interaction
    DISPLAY_SCALE = 0.4 if max(w, h) > 800 else 0.7
   
    display = cv2.resize(img, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE,
                        interpolation=cv2.INTER_AREA)
    display_clone = display.copy()
   
    points_display = []
    points_original = []
   
    def mouse_callback(event, x, y, flags, param):
        nonlocal display_clone
       
        if event == cv2.EVENT_LBUTTONDOWN:
            points_display.append((x, y))
           
            # Map to original coordinates
            ox = int(x / DISPLAY_SCALE)
            oy = int(y / DISPLAY_SCALE)
            points_original.append((ox, oy))
           
            # Visual feedback
            cv2.circle(display_clone, (x, y), 4, (0, 0, 255), -1)
           
            if len(points_display) > 1:
                cv2.line(display_clone, points_display[-2],
                        points_display[-1], (0, 255, 0), 2)
           
            # Close polygon if we have enough points
            if len(points_display) > 2:
                cv2.line(display_clone, points_display[-1],
                        points_display[0], (0, 255, 0), 2)
           
            cv2.imshow("Draw ROI - Press ENTER when done", display_clone)
   
    cv2.namedWindow("Draw ROI - Press ENTER when done", cv2.WINDOW_NORMAL)
    cv2.imshow("Draw ROI - Press ENTER when done", display_clone)
    cv2.setMouseCallback("Draw ROI - Press ENTER when done", mouse_callback)
   
    print("🟢 Click to draw ROI polygon. Press ENTER when finished.")
   
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # ENTER
            break
        elif key == 27:  # ESC
            print("❌ ROI creation cancelled")
            cv2.destroyAllWindows()
            return None
   
    cv2.destroyAllWindows()
   
    if len(points_original) < 3:
        print("❌ Need at least 3 points for ROI")
        return None
   
    # Create mask
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array(points_original, dtype=np.int32)
    cv2.fillPoly(roi_mask, [pts], 255)
   
    # Save mask with descriptive name
    shared_folder = "roi_masks/shared"
    os.makedirs(shared_folder, exist_ok=True)
    mask_filename = f"{roi_name}.jpg"  # Remove image-specific suffix
    mask_filepath = os.path.join(shared_folder, mask_filename)
    cv2.imwrite(mask_filepath, roi_mask)

    print(f"✅ ROI mask saved as: {mask_filename}")
   
    # Visual confirmation
    overlay = img.copy()
    cv2.polylines(overlay, [pts], True, (0, 255, 0), 3)
    overlay_small = cv2.resize(overlay, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
   
    cv2.imshow("ROI Confirmation - Press any key to continue", overlay_small)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
   
    return mask_filepath, roi_mask

def align_images(ref, test):

    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(5000)

    kp1, des1 = orb.detectAndCompute(ref_gray, None)
    kp2, des2 = orb.detectAndCompute(test_gray, None)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)

    matches = sorted(matches, key=lambda x: x.distance)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches[:80]]).reshape(-1,1,2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches[:80]]).reshape(-1,1,2)

    H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
    if H is None:
        print("Homography failed")
        return None

    aligned = cv2.warpPerspective(test, H, (ref.shape[1], ref.shape[0]))

    return aligned



def color_inspection(ref_path, test_path, roi_mask_path=None, delta_e_limit=2.0):
    """Improved color inspection with proper ROI handling"""
   
    # Load images
    ref = cv2.imread(ref_path)
    test = cv2.imread(test_path)
   
    if ref is None or test is None:
        print("❌ Error loading images")
        print(f"Reference: {ref_path} - {'✅ OK' if ref is not None else '❌ MISSING'}")
        print(f"Test: {test_path} - {'✅ OK' if test is not None else '❌ MISSING'}")
        return False, 0
   
    # Get reference dimensions
    ref_h, ref_w = ref.shape[:2]
   
    # Resize test image to match reference
    test = cv2.resize(test, (ref_w, ref_h))
   
    # Load or create ROI mask
    if roi_mask_path and os.path.exists(roi_mask_path):
        roi_mask = cv2.imread(roi_mask_path, cv2.IMREAD_GRAYSCALE)
        roi_mask = cv2.resize(roi_mask, (ref_w, ref_h), interpolation=cv2.INTER_NEAREST)
        print(f"✅ Using ROI mask: {roi_mask_path}")
    else:
        print("⚠️ No ROI mask found, using entire image")
        roi_mask = np.ones((ref_h, ref_w), dtype=np.uint8) * 255
   
    # Ensure binary mask
    _, roi_mask = cv2.threshold(roi_mask, 127, 255, cv2.THRESH_BINARY)
   
    # Check if ROI is valid
    roi_pixels = np.sum(roi_mask == 255)
    if roi_pixels < 100:
        print("❌ ROI too small or invalid")
        return False, 0
   
    print(f"ROI covers {roi_pixels / roi_mask.size * 100:.1f}% of image")
   
    # Convert to LAB color space
    ref_lab = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB)
    test_lab = cv2.cvtColor(test, cv2.COLOR_BGR2LAB)
   
    # Extract ROI pixels only
    ref_roi_pixels = ref_lab[roi_mask == 255]
    test_roi_pixels = test_lab[roi_mask == 255]
   
    if len(ref_roi_pixels) == 0 or len(test_roi_pixels) == 0:
        print("❌ No valid pixels in ROI")
        return False, 0
   
    # Calculate mean LAB values
    ref_mean = ref_roi_pixels.mean(axis=0)
    test_mean = test_roi_pixels.mean(axis=0)
   
    # Convert OpenCV LAB to CIE LAB
    def opencv_to_cielab(lab):
        L = lab[0] * 100.0 / 255.0
        a = lab[1] - 128.0
        b = lab[2] - 128.0
        return np.array([L, a, b])
   
    ref_cielab = opencv_to_cielab(ref_mean)
    test_cielab = opencv_to_cielab(test_mean)
   
    # Calculate Delta E
    delta_e = np.linalg.norm(ref_cielab - test_cielab)
   
    # Visualization
    DISPLAY_SCALE = 0.5
    
    # Show ROI on both images
    ref_vis = ref.copy()
    test_vis = test.copy()
    
    contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(ref_vis, contours, -1, (0, 255, 0), 2)
    cv2.drawContours(test_vis, contours, -1, (0, 255, 0), 2)
    
    # Add text results
    color_result = "PASS" if delta_e <= delta_e_limit else "FAIL"
    color_bgr = (0, 255, 0) if delta_e <= delta_e_limit else (0, 0, 255)
    
    cv2.putText(test_vis, f"Delta E: {delta_e:.2f}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_bgr, 2)
    cv2.putText(test_vis, color_result, (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color_bgr, 2)

    # Only show visualizations if not in headless mode
    if not HEADLESS_MODE:
        # Display results
        ref_disp = cv2.resize(ref_vis, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
        test_disp = cv2.resize(test_vis, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
        roi_disp = cv2.resize(roi_mask, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
        
        cv2.imshow("Test + ROI", test_disp)
    
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    print(f"\n=== COLOR ANALYSIS ===")
    print(f"Reference LAB: {ref_cielab}")
    print(f"Test LAB:      {test_cielab}")
    print(f"Delta E:       {delta_e:.2f}")
    print(f"Result:        {'PASS' if delta_e <= delta_e_limit else 'FAIL'}")
    
    return delta_e > delta_e_limit, delta_e
   
    contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(ref_vis, contours, -1, (0, 255, 0), 2)
    cv2.drawContours(test_vis, contours, -1, (0, 255, 0), 2)
   
    # Add text results
    color_result = "PASS" if delta_e <= delta_e_limit else "FAIL"
    color_bgr = (0, 255, 0) if delta_e <= delta_e_limit else (0, 0, 255)
   
    cv2.putText(test_vis, f"Delta E: {delta_e:.2f}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_bgr, 2)
    cv2.putText(test_vis, color_result, (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color_bgr, 2)
   
    # Display results
    ref_disp = cv2.resize(ref_vis, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
    test_disp = cv2.resize(test_vis, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
    roi_disp = cv2.resize(roi_mask, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
   


    cv2.imshow("Test + ROI", test_disp)
   
    print(f"\n=== COLOR ANALYSIS ===")
    print(f"Reference LAB: {ref_cielab}")
    print(f"Test LAB:      {test_cielab}")
    print(f"Delta E:       {delta_e:.2f}")
    print(f"Result:        {'PASS' if delta_e <= delta_e_limit else 'FAIL'}")
   
    cv2.waitKey(0)
    cv2.destroyAllWindows()
   
    return delta_e > delta_e_limit, delta_e


import cv2
import numpy as np
from sklearn.cluster import KMeans



def advanced_paint_defect_detection(ref_path, test_path, roi_mask_path=None, sensitivity='medium'):
    """
    Advanced paint defect detection using multiple methods
    """
    
    ref = cv2.imread(ref_path)
    test = cv2.imread(test_path)

    aligned_test = align_images(ref, test)

    if aligned_test is None:
        print("Alignment failed")
        return False, 0, []
    

    
    original_img = aligned_test.copy()
    h, w = original_img.shape[:2]
    
    # Load ROI mask
    if roi_mask_path and os.path.exists(roi_mask_path):
        roi_mask = cv2.imread(roi_mask_path, cv2.IMREAD_GRAYSCALE)
        roi_mask = cv2.resize(roi_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        roi_mask = np.ones((h, w), dtype=np.uint8) * 255
    
    # Ensure binary mask
    _, roi_mask = cv2.threshold(roi_mask, 127, 255, cv2.THRESH_BINARY)
    
    # Apply ROI mask
    diff = cv2.absdiff(ref, aligned_test)

    masked_diff = cv2.bitwise_and(diff, diff, mask=roi_mask)

    gray = cv2.cvtColor(masked_diff, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    gray = cv2.GaussianBlur(gray, (3,3), 0)

    _, roi_mask = cv2.threshold(roi_mask, 127, 255, cv2.THRESH_BINARY)

    kernel = np.ones((15,15), np.uint8)
    roi_mask = cv2.erode(roi_mask, kernel)

    # ===============================
    # DARK PAINT DETECTOR (BLACKHAT)
    # ===============================
    bh_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    gray_flat = cv2.medianBlur(gray, 9)
    blackhat = cv2.morphologyEx(gray_flat, cv2.MORPH_BLACKHAT, bh_kernel)
    _, blackhat_mask = cv2.threshold(blackhat, 12, 255, cv2.THRESH_BINARY)


    # DO NOT blur — it destroys tiny paint defects
    gray_blur = gray
    
    # Calculate local standard deviation (texture measure)
    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    mean = cv2.filter2D(gray_blur.astype(np.float32), -1, kernel)
    squared_diff = (gray_blur.astype(np.float32) - mean) ** 2
    variance = cv2.filter2D(squared_diff, -1, kernel)
    texture_map = np.sqrt(variance)
    

    
    # ===============================
    # METHOD 2: COLOR UNIFORMITY
    # ===============================
    # Convert to LAB for better color analysis
    lab = cv2.cvtColor(masked_diff, cv2.COLOR_BGR2LAB)
    
    # Calculate local color variation
    lab_blur = cv2.medianBlur(lab, 7)    
    color_diff = np.zeros_like(gray, dtype=np.float32)
    
    for i in range(3):  # L, a, b channels
        channel = lab[:,:,i].astype(np.float32)
        channel_blur = lab_blur[:,:,i].astype(np.float32)
        diff = np.abs(channel - channel_blur)
        color_diff += diff
    
    color_diff_norm = cv2.normalize(color_diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    

    # ===============================
    # COMBINE ALL METHODS
    # ===============================
    
    # Set sensitivity parameters
    if sensitivity == "high":
        texture_thresh, color_thresh, edge_weight, outlier_weight = 30, 20, 0.3, 0.4

    elif sensitivity == "medium":
        texture_thresh, color_thresh, edge_weight, outlier_weight = 40, 30, 0.2, 0.3

    else:  # low
        texture_thresh, color_thresh, edge_weight, outlier_weight = 50, 40, 0.1, 0.2
    # Create individual masks
    _, color_mask = cv2.threshold(color_diff_norm, color_thresh, 255, cv2.THRESH_BINARY)
    


    # Combine masks with weights
    combined_mask = np.zeros_like(gray, dtype=np.float32)
    combined_mask += color_mask.astype(np.float32) / 255.0 * 0.3
    combined_mask += blackhat_mask.astype(np.float32) / 255.0 * 0.6
    # Normalize and threshold
    
    combined_mask = cv2.normalize(
        combined_mask, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)
    _, strong_mask = cv2.threshold(combined_mask, 32, 255, cv2.THRESH_BINARY)
    strong_pixels = np.sum(strong_mask == 255)

    _, weak_mask = cv2.threshold(combined_mask, 24, 255, cv2.THRESH_BINARY)

    kernel = np.ones((2,2), np.uint8)
    weak_mask = cv2.morphologyEx(weak_mask, cv2.MORPH_OPEN, kernel)

    weak_pixels = np.sum(weak_mask == 255)


    # Now threshold
    _, final_mask = cv2.threshold(combined_mask, 32, 255, cv2.THRESH_BINARY)
    

    # Apply ROI constraint
    final_mask = cv2.bitwise_and(final_mask, roi_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4,4))
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
    # ===============================
    # MORPHOLOGICAL CLEANING (reduced)
    # ===============================
    
    # damage_pixels = np.sum(final_mask == 255)

    # print("Damage pixels:", damage_pixels)

    # if damage_pixels > 250:
    #     paint_detected = True
    # else:
    #     paint_detected = False



    # ===============================
    # CONTOUR ANALYSIS
    # ===============================
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print("[DEBUG] All contour areas:", [cv2.contourArea(c) for c in contours])

    paint_defects = []
    result_img = original_img.copy()

    # More lenient area thresholds for subtle defects
    min_area = 120   # Lower minimum area for more sensitivity
    max_area = 1500  # Larger maximum area

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if min_area <= area <= max_area:
            paint_defects.append(cnt)
            # Draw bounding box
            x, y, w_box, h_box = cv2.boundingRect(cnt)
            cv2.rectangle(result_img, (x, y), (x + w_box, y + h_box), (0, 0, 255), 2)
            # Add defect number
            cv2.putText(result_img, f"D{i+1}", (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    damage_pixels = np.sum(final_mask == 255)

    paint_detected = (
        strong_pixels > 120
        or
        weak_pixels > 600
    )


    # Add status text
    status = f"PAINT DEFECTS: {damage_pixels}" if paint_detected else "NO PAINT DEFECT"
    color = (0, 0, 255) if paint_detected else (0, 255, 0)
    cv2.putText(result_img, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    # ===============================
    # DETAILED VISUALIZATION
    # ===============================

    DISPLAY_SCALE = 0.6

    # Create visualization grid
    # Ensure combined_mask is uint8 for colormap
    combined_mask_uint8 = cv2.normalize(combined_mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    displays = [
        ("1. Combined Detection", cv2.applyColorMap(combined_mask_uint8, cv2.COLORMAP_JET)),
        ("2. Final Mask", cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR)),
        ("3. FINAL RESULT", result_img)
    ]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6,6))
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)

    # Only show visualizations if not in headless mode
    if not HEADLESS_MODE:
        for name, disp_img in displays:
            disp_resized = cv2.resize(disp_img, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
            cv2.imshow(name, disp_resized)

    print(f"\n=== ADVANCED PAINT DETECTION ===")
    print(f"Defects found: {len(paint_defects)}")
    print(f"Defect areas: {[cv2.contourArea(cnt) for cnt in paint_defects]}")
    print(f"Result: {'DEFECT DETECTED' if paint_detected else 'NO DEFECT'}")
    print(f"Sensitivity: {sensitivity}")

    if not HEADLESS_MODE:
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return paint_detected, len(paint_defects), paint_defects
    # Create visualization grid
    # Ensure combined_mask is uint8 for colormap
    combined_mask_uint8 = cv2.normalize(combined_mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    displays = [
        ("1. Combined Detection", cv2.applyColorMap(combined_mask_uint8, cv2.COLORMAP_JET)),
        ("2. Final Mask", cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR)),
        ("3. FINAL RESULT", result_img)
    ]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6,6))
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)

    for name, disp_img in displays:
        disp_resized = cv2.resize(disp_img, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
        cv2.imshow(name, disp_resized)
    
    print(f"\n=== ADVANCED PAINT DETECTION ===")
    print(f"Defects found: {len(paint_defects)}")
    print(f"Defect areas: {[cv2.contourArea(cnt) for cnt in paint_defects]}")
    print(f"Result: {'DEFECT DETECTED' if paint_detected else 'NO DEFECT'}")
    print(f"Sensitivity: {sensitivity}")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return paint_detected, len(paint_defects), paint_defects

# ===============================
# PARAMETER TUNING FUNCTION
# ===============================
def tune_detection_parameters(test_path, roi_mask_path=None):
    """
    Interactive parameter tuning for paint defect detection
    """
    
    print("🔧 Starting parameter tuning mode...")
    print("Try different sensitivity levels:")
    
    for sensitivity in ['high', 'medium', 'low']:
        print(f"\n--- Testing {sensitivity.upper()} sensitivity ---")
        detected, count, defects = advanced_paint_defect_detection(
            test_path, roi_mask_path, sensitivity)
        
        print(f"Sensitivity: {sensitivity} | Defects: {count} | Detected: {detected}")
        
        response = input("Press ENTER to continue, 'q' to quit, or 's' to stop here: ")
        if response.lower() == 'q':
            break
        elif response.lower() == 's':
            return sensitivity
    
    return 'medium'






def advanced_scratch_detection(ref_path, test_path, roi_mask_path=None, sensitivity='medium'):
    """
    Advanced scratch detection using multiple methods
    Detects linear defects, scratches, and surface damage
    """
    
    # Load test image
    ref = cv2.imread(ref_path)
    test = cv2.imread(test_path)

    if ref is None or test is None:
        print("❌ Error loading images")
        return False, 0, []

    aligned_test = align_images(ref, test)

    if aligned_test is None:
        print("Alignment failed")
        return False, 0, []

    # ⭐ THIS is the magic line
    img = cv2.absdiff(ref, aligned_test)

    original_img = aligned_test.copy()
    h, w = img.shape[:2]

    gray_diff = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh_diff = cv2.threshold(gray_diff, 20, 255, cv2.THRESH_BINARY)

    img = cv2.cvtColor(thresh_diff, cv2.COLOR_GRAY2BGR)


    
    # Load ROI mask
    if roi_mask_path and os.path.exists(roi_mask_path):
        roi_mask = cv2.imread(roi_mask_path, cv2.IMREAD_GRAYSCALE)
        roi_mask = cv2.resize(roi_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        roi_mask = np.ones((h, w), dtype=np.uint8) * 255
    
    # Ensure binary mask
    _, roi_mask = cv2.threshold(roi_mask, 127, 255, cv2.THRESH_BINARY)
    
    # Apply ROI mask
    masked_img = cv2.bitwise_and(img, img, mask=roi_mask)
    
    # ===============================
    # METHOD 1: LINE DETECTION (Hough Transform)
    # ===============================
    gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
    
    # Enhance contrast for better scratch detection
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    
    # Edge detection with multiple thresholds
    edges1 = cv2.Canny(blurred, 30, 90)
    edges2 = cv2.Canny(blurred, 50, 150)
    edges = cv2.bitwise_or(edges1, edges2)
    
    # Apply ROI to edges
    edges = cv2.bitwise_and(edges, roi_mask)
    
    # Hough Line Transform for detecting linear scratches
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 
                            threshold=30, minLineLength=20, maxLineGap=10)
    
    line_mask = np.zeros_like(gray)
    detected_lines = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            
            # Filter lines by length (scratches are typically longer)
            if length > 15:
                cv2.line(line_mask, (x1, y1), (x2, y2), 255, 2)
                detected_lines.append(line[0])
    
    # ===============================
    # METHOD 2: MORPHOLOGICAL LINE DETECTION
    # ===============================
    # Detect horizontal scratches
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    detect_horizontal = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    
    # Detect vertical scratches
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    detect_vertical = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
    
    # Detect diagonal scratches (45 degrees)
    diagonal_kernel1 = np.array([[1, 0, 0, 0, 0],
                                 [0, 1, 0, 0, 0],
                                 [0, 0, 1, 0, 0],
                                 [0, 0, 0, 1, 0],
                                 [0, 0, 0, 0, 1]], dtype=np.uint8)
    detect_diagonal1 = cv2.morphologyEx(edges, cv2.MORPH_OPEN, diagonal_kernel1, iterations=1)
    
    # Detect diagonal scratches (-45 degrees)
    diagonal_kernel2 = np.flip(diagonal_kernel1, axis=1)
    detect_diagonal2 = cv2.morphologyEx(edges, cv2.MORPH_OPEN, diagonal_kernel2, iterations=1)
    
    # Combine all directional detections
    morphological_mask = cv2.bitwise_or(detect_horizontal, detect_vertical)
    morphological_mask = cv2.bitwise_or(morphological_mask, detect_diagonal1)
    morphological_mask = cv2.bitwise_or(morphological_mask, detect_diagonal2)
    
    # ===============================
    # METHOD 3: GRADIENT-BASED DETECTION
    # ===============================
    # Sobel gradients for detecting edges
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    
    # Gradient magnitude
    gradient_mag = np.sqrt(sobelx**2 + sobely**2)
    gradient_mag = cv2.normalize(gradient_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Threshold gradient for scratch-like features
    _, gradient_mask = cv2.threshold(gradient_mag, 50, 255, cv2.THRESH_BINARY)
    
    # Apply morphological operations to isolate linear features
    line_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    gradient_lines = cv2.morphologyEx(gradient_mask, cv2.MORPH_OPEN, line_kernel)
    
    # ===============================
    # METHOD 4: VARIANCE-BASED DETECTION
    # ===============================
    # Calculate local variance to find irregular areas
    kernel_size = 7
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size**2)
    
    mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
    squared_diff = (gray.astype(np.float32) - mean) ** 2
    variance = cv2.filter2D(squared_diff, -1, kernel)
    
    # Normalize variance
    variance_norm = cv2.normalize(variance, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # ===============================
    # COMBINE ALL METHODS
    # ===============================
    
    # Set sensitivity parameters
    if sensitivity == 'high':
        line_weight, morph_weight, grad_weight, var_thresh = 0.4, 0.3, 0.2, 40
    elif sensitivity == 'medium':
        line_weight, morph_weight, grad_weight, var_thresh = 0.35, 0.3, 0.25, 50
    else:  # low
        line_weight, morph_weight, grad_weight, var_thresh = 0.3, 0.25, 0.2, 60
    
    # Threshold variance
    _, variance_mask = cv2.threshold(variance_norm, var_thresh, 255, cv2.THRESH_BINARY)
    
    # Combine all detection methods
    combined_mask = np.zeros_like(gray, dtype=np.float32)
    combined_mask += line_mask.astype(np.float32) / 255.0 * line_weight
    combined_mask += morphological_mask.astype(np.float32) / 255.0 * morph_weight
    combined_mask += gradient_lines.astype(np.float32) / 255.0 * grad_weight
    combined_mask += variance_mask.astype(np.float32) / 255.0 * 0.1
    
    # Normalize and threshold
    combined_mask = cv2.normalize(combined_mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, final_mask = cv2.threshold(combined_mask, 60, 255, cv2.THRESH_BINARY)  # Lower threshold for more sensitivity
    
    # Apply ROI constraint
    final_mask = cv2.bitwise_and(final_mask, roi_mask)
    
    # ===============================
    # MORPHOLOGICAL REFINEMENT
    # ===============================
    # Remove small noise
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel_open)
    
    # Connect nearby scratch segments
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel_close)
    
    kernel_expand = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    final_mask = cv2.dilate(final_mask, kernel_expand, iterations=1)
    # ===============================
    # CONTOUR ANALYSIS
    # ===============================
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    scratches = []
    result_img = original_img.copy()
    
    # Filter contours based on scratch characteristics
    min_area = 15
    max_area = 3000
    min_aspect_ratio = 2.5  # Scratches are elongated
    
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        
        if area < min_area or area > max_area:
            continue
        
        # Get bounding rectangle
        x, y, w_box, h_box = cv2.boundingRect(cnt)
        
        # Calculate aspect ratio
        aspect_ratio = max(w_box, h_box) / (min(w_box, h_box) + 1)
        
        # Check if it's elongated (scratch-like)
        if aspect_ratio >= min_aspect_ratio or area > 100:
            scratches.append(cnt)
            
            # Draw bounding box
            cv2.rectangle(result_img, (x, y), (x + w_box, y + h_box), (255, 0, 0), 2)
            
            # Draw the actual contour
            cv2.drawContours(result_img, [cnt], -1, (0, 255, 255), 1)
            
            # Add scratch number
            cv2.putText(result_img, f"S{i+1}", (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    scratch_detected = len(scratches) > 0
    
    # Add status text
    status = f"SCRATCHES: {len(scratches)}" if scratch_detected else "NO SCRATCHES"
    color = (255, 0, 0) if scratch_detected else (0, 255, 0)
    cv2.putText(result_img, status, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    # ===============================
    # DETAILED VISUALIZATION
    # ===============================
    DISPLAY_SCALE = 0.6

    # Create visualization grid
    displays = [
        ("1. Combined Detection", cv2.applyColorMap(combined_mask, cv2.COLORMAP_JET)),
        ("2. Final Mask", cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR)),
        ("3. SCRATCH RESULT", result_img)
    ]

    # Only show visualizations if not in headless mode
    if not HEADLESS_MODE:
        for name, disp_img in displays:
            disp_resized = cv2.resize(disp_img, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
            cv2.imshow(name, disp_resized)

    print(f"\n=== ADVANCED SCRATCH DETECTION ===")
    print(f"Scratches found: {len(scratches)}")
    print(f"Scratch areas: {[cv2.contourArea(cnt) for cnt in scratches]}")
    print(f"Result: {'SCRATCH DETECTED' if scratch_detected else 'NO SCRATCH'}")
    print(f"Sensitivity: {sensitivity}")

    if not HEADLESS_MODE:
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return scratch_detected, len(scratches), scratches

    # Create visualization grid
    displays = [
        ("1. Combined Detection", cv2.applyColorMap(combined_mask, cv2.COLORMAP_JET)),
        ("2. Final Mask", cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR)),
        ("3. SCRATCH RESULT", result_img)
    ]
    
    for name, disp_img in displays:
        disp_resized = cv2.resize(disp_img, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
        cv2.imshow(name, disp_resized)
    
    print(f"\n=== ADVANCED SCRATCH DETECTION ===")
    print(f"Scratches found: {len(scratches)}")
    print(f"Scratch areas: {[cv2.contourArea(cnt) for cnt in scratches]}")
    print(f"Result: {'SCRATCH DETECTED' if scratch_detected else 'NO SCRATCH'}")
    print(f"Sensitivity: {sensitivity}")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return scratch_detected, len(scratches), scratches








# import cv2
# import numpy as np
# import os


# def advanced_scratch_detection(test_path, roi_mask_path=None, sensitivity='medium'):
#     """
#     Enhanced scratch detection with improved accuracy
#     Detects linear defects, scratches, and surface damage
    
#     Parameters:
#     -----------
#     test_path : str
#         Path to the test image
#     roi_mask_path : str, optional
#         Path to ROI mask image
#     sensitivity : str
#         Detection sensitivity: 'low', 'medium', or 'high'
    
#     Returns:
#     --------
#     scratch_detected : bool
#         Whether scratches were detected
#     num_scratches : int
#         Number of scratches found
#     scratches : list
#         List of scratch contours
#     """
    
#     # Load test image
#     img = cv2.imread(test_path)
#     if img is None:
#         print(f"❌ Error loading test image: {test_path}")
#         return False, 0, []
    
#     original_img = img.copy()
#     h, w = img.shape[:2]
    
#     # Load ROI mask
#     if roi_mask_path and os.path.exists(roi_mask_path):
#         roi_mask = cv2.imread(roi_mask_path, cv2.IMREAD_GRAYSCALE)
#         roi_mask = cv2.resize(roi_mask, (w, h), interpolation=cv2.INTER_NEAREST)
#     else:
#         roi_mask = np.ones((h, w), dtype=np.uint8) * 255
    
#     # Ensure binary mask
#     _, roi_mask = cv2.threshold(roi_mask, 127, 255, cv2.THRESH_BINARY)
    
#     # Apply ROI mask
#     masked_img = cv2.bitwise_and(img, img, mask=roi_mask)
    
#     # ===============================
#     # PREPROCESSING
#     # ===============================
#     gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
    
#     # Apply bilateral filter to reduce noise while preserving edges
#     denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    
#     # Adaptive histogram equalization for better contrast
#     clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
#     enhanced = clahe.apply(denoised)
    
#     # ===============================
#     # METHOD 1: ADAPTIVE EDGE DETECTION
#     # ===============================
#     # Multi-scale edge detection
#     edges_fine = cv2.Canny(enhanced, 40, 120)
#     edges_coarse = cv2.Canny(enhanced, 20, 60)
#     edges = cv2.bitwise_or(edges_fine, edges_coarse)
    
#     # Apply ROI to edges
#     edges = cv2.bitwise_and(edges, roi_mask)
    
#     # Remove isolated pixels
#     kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
#     edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_clean)
    
#     # ===============================
#     # METHOD 2: DIRECTIONAL LINE DETECTION
#     # ===============================
#     line_masks = []
    
#     # Horizontal scratches (improved)
#     h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
#     h_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, h_kernel, iterations=1)
#     line_masks.append(h_lines)
    
#     # Vertical scratches (improved)
#     v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
#     v_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, v_kernel, iterations=1)
#     line_masks.append(v_lines)
    
#     # Multiple diagonal angles
#     for angle in [45, 135, 30, 150, 60, 120]:
#         kernel_size = 15
#         diag_kernel = create_directional_kernel(kernel_size, angle)
#         diag_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, diag_kernel, iterations=1)
#         line_masks.append(diag_lines)
    
#     # Combine directional detections
#     directional_mask = np.zeros_like(gray)
#     for mask in line_masks:
#         directional_mask = cv2.bitwise_or(directional_mask, mask)
    
#     # ===============================
#     # METHOD 3: HOUGH LINE TRANSFORM (IMPROVED)
#     # ===============================
#     line_mask = np.zeros_like(gray)
#     detected_lines = []
    
#     # Detect lines with better parameters
#     lines = cv2.HoughLinesP(edges, 1, np.pi/180, 
#                             threshold=25, minLineLength=25, maxLineGap=8)
    
#     if lines is not None:
#         for line in lines:
#             x1, y1, x2, y2 = line[0]
#             length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            
#             # Calculate line angle
#             angle = np.abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
            
#             # Filter by length and check if it's relatively straight
#             if length > 20:
#                 # Check line intensity along the line
#                 line_profile = get_line_intensity(enhanced, (x1, y1), (x2, y2))
                
#                 # If line has consistent low or high intensity, it's likely a scratch
#                 if np.std(line_profile) < 40:  # Consistent intensity
#                     cv2.line(line_mask, (x1, y1), (x2, y2), 255, 2)
#                     detected_lines.append(line[0])
    
#     # ===============================
#     # METHOD 4: RIDGE DETECTION (NEW)
#     # ===============================
#     # Detect dark scratches (valleys) and bright scratches (ridges)
    
#     # Top-hat transform for bright scratches
#     tophat_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
#     tophat = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, tophat_kernel)
    
#     # Black-hat transform for dark scratches
#     blackhat = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, tophat_kernel)
    
#     # Combine
#     ridges = cv2.add(tophat, blackhat)
    
#     # Threshold ridges
#     _, ridge_mask = cv2.threshold(ridges, 15, 255, cv2.THRESH_BINARY)
    
#     # Keep only linear features
#     ridge_mask = filter_linear_features(ridge_mask)
    
#     # ===============================
#     # METHOD 5: LAPLACIAN OF GAUSSIAN
#     # ===============================
#     # Detect edges using LoG
#     blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
#     laplacian = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)
#     laplacian = np.absolute(laplacian)
#     laplacian = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
#     _, lap_mask = cv2.threshold(laplacian, 30, 255, cv2.THRESH_BINARY)
#     lap_mask = filter_linear_features(lap_mask)
    
#     # ===============================
#     # SENSITIVITY CONFIGURATION
#     # ===============================
#     if sensitivity == 'high':
#         weights = {
#             'directional': 0.30,
#             'hough': 0.25,
#             'ridge': 0.20,
#             'laplacian': 0.15,
#             'edges': 0.10
#         }
#         combine_threshold = 50
#         min_area = 10
#         min_aspect_ratio = 2.0
#     elif sensitivity == 'medium':
#         weights = {
#             'directional': 0.30,
#             'hough': 0.30,
#             'ridge': 0.20,
#             'laplacian': 0.10,
#             'edges': 0.10
#         }
#         combine_threshold = 60
#         min_area = 15
#         min_aspect_ratio = 2.5
#     else:  # low
#         weights = {
#             'directional': 0.25,
#             'hough': 0.35,
#             'ridge': 0.20,
#             'laplacian': 0.10,
#             'edges': 0.10
#         }
#         combine_threshold = 70
#         min_area = 20
#         min_aspect_ratio = 3.0
    
#     # ===============================
#     # INTELLIGENT COMBINATION
#     # ===============================
#     combined_mask = np.zeros_like(gray, dtype=np.float32)
#     combined_mask += directional_mask.astype(np.float32) / 255.0 * weights['directional']
#     combined_mask += line_mask.astype(np.float32) / 255.0 * weights['hough']
#     combined_mask += ridge_mask.astype(np.float32) / 255.0 * weights['ridge']
#     combined_mask += lap_mask.astype(np.float32) / 255.0 * weights['laplacian']
#     combined_mask += edges.astype(np.float32) / 255.0 * weights['edges']
    
#     # Normalize
#     combined_mask = cv2.normalize(combined_mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
#     # Adaptive thresholding
#     _, final_mask = cv2.threshold(combined_mask, combine_threshold, 255, cv2.THRESH_BINARY)
    
#     # Apply ROI constraint
#     final_mask = cv2.bitwise_and(final_mask, roi_mask)
    
#     # ===============================
#     # MORPHOLOGICAL REFINEMENT
#     # ===============================
#     # Remove small isolated noise
#     kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
#     final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
    
#     # Connect nearby scratch segments
#     kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
#     final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel_close)
    
#     # Remove remaining noise
#     final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
    
#     # ===============================
#     # CONTOUR ANALYSIS WITH IMPROVED FILTERING
#     # ===============================
#     contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     scratches = []
#     result_img = original_img.copy()
    
#     max_area = 5000
    
#     for i, cnt in enumerate(contours):
#         area = cv2.contourArea(cnt)
        
#         if area < min_area or area > max_area:
#             continue
        
#         # Get bounding rectangle
#         x, y, w_box, h_box = cv2.boundingRect(cnt)
        
#         # Calculate aspect ratio
#         aspect_ratio = max(w_box, h_box) / (min(w_box, h_box) + 1)
        
#         # Fit ellipse if possible
#         if len(cnt) >= 5:
#             ellipse = cv2.fitEllipse(cnt)
#             (center, axes, angle) = ellipse
#             major_axis = max(axes)
#             minor_axis = min(axes)
#             ellipse_ratio = major_axis / (minor_axis + 1)
#         else:
#             ellipse_ratio = aspect_ratio
        
#         # Calculate solidity (how compact the shape is)
#         hull = cv2.convexHull(cnt)
#         hull_area = cv2.contourArea(hull)
#         solidity = area / (hull_area + 1e-5)
        
#         # Calculate extent (ratio of contour area to bounding box area)
#         rect_area = w_box * h_box
#         extent = area / (rect_area + 1e-5)
        
#         # IMPROVED SCRATCH CRITERIA:
#         # 1. Elongated shape (high aspect ratio)
#         # 2. Not too compact (solidity check)
#         # 3. Reasonable extent
#         is_scratch = False
        
#         if aspect_ratio >= min_aspect_ratio or ellipse_ratio >= min_aspect_ratio:
#             # Primary criterion: elongated shape
#             if solidity < 0.95 and extent > 0.15:
#                 is_scratch = True
#         elif area > 50 and aspect_ratio >= (min_aspect_ratio * 0.7):
#             # Larger defects with slightly lower aspect ratio
#             if solidity < 0.90 and extent > 0.20:
#                 is_scratch = True
        
#         if is_scratch:
#             # Verify it's in a valid region
#             mask_value = roi_mask[y + h_box//2, x + w_box//2]
#             if mask_value > 0:
#                 scratches.append(cnt)
                
#                 # Draw bounding box
#                 cv2.rectangle(result_img, (x, y), (x + w_box, y + h_box), (0, 0, 255), 2)
                
#                 # Draw the actual contour
#                 cv2.drawContours(result_img, [cnt], -1, (0, 255, 255), 2)
                
#                 # Add scratch information
#                 scratch_info = f"S{len(scratches)} A:{int(area)}"
#                 cv2.putText(result_img, scratch_info, (x, y-5),
#                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
#     scratch_detected = len(scratches) > 0
    
#     # Add status text
#     status = f"SCRATCHES DETECTED: {len(scratches)}" if scratch_detected else "NO SCRATCHES DETECTED"
#     color = (0, 0, 255) if scratch_detected else (0, 255, 0)
#     cv2.putText(result_img, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
#     cv2.putText(result_img, f"Sensitivity: {sensitivity.upper()}", (20, 70), 
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
#     # ===============================
#     # DETAILED VISUALIZATION
#     # ===============================
#     DISPLAY_SCALE = 0.6
    
#     # Create visualization panels
#     edge_vis = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
#     directional_vis = cv2.cvtColor(directional_mask, cv2.COLOR_GRAY2BGR)
#     combined_vis = cv2.applyColorMap(combined_mask, cv2.COLORMAP_JET)
#     final_vis = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR)
    
#     displays = [
#         ("1. Edge Detection", edge_vis),
#         ("2. Directional Lines", directional_vis),
#         ("3. Combined Score", combined_vis),
#         ("4. Final Mask", final_vis),
#         ("5. RESULT", result_img)
#     ]
    
#     for name, disp_img in displays:
#         disp_resized = cv2.resize(disp_img, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
#         cv2.imshow(name, disp_resized)
    
#     # ===============================
#     # DETAILED REPORTING
#     # ===============================
#     print(f"\n{'='*50}")
#     print(f"ADVANCED SCRATCH DETECTION RESULTS")
#     print(f"{'='*50}")
#     print(f"Image: {test_path}")
#     print(f"Sensitivity: {sensitivity.upper()}")
#     print(f"Image size: {w}x{h}")
#     print(f"\nDetection Summary:")
#     print(f"  Total scratches found: {len(scratches)}")
#     print(f"  Detection status: {'⚠️  SCRATCH DETECTED' if scratch_detected else '✓ NO SCRATCH'}")
    
#     if scratches:
#         print(f"\nScratch Details:")
#         for i, cnt in enumerate(scratches):
#             area = cv2.contourArea(cnt)
#             x, y, w_box, h_box = cv2.boundingRect(cnt)
#             aspect_ratio = max(w_box, h_box) / (min(w_box, h_box) + 1)
#             print(f"  Scratch {i+1}: Area={int(area)}px², Aspect={aspect_ratio:.2f}, Size={w_box}x{h_box}")
    
#     print(f"{'='*50}\n")
    
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
    
#     return scratch_detected, len(scratches), scratches


# # ===============================
# # HELPER FUNCTIONS
# # ===============================

# def create_directional_kernel(size, angle):
#     """Create a directional line kernel for morphological operations"""
#     kernel = np.zeros((size, size), dtype=np.uint8)
#     center = size // 2
    
#     angle_rad = np.radians(angle)
#     for i in range(size):
#         offset = int((i - center) * np.tan(angle_rad))
#         if 0 <= center + offset < size:
#             kernel[i, center + offset] = 1
    
#     return kernel


# def get_line_intensity(img, pt1, pt2, num_samples=50):
#     """Get intensity values along a line"""
#     x1, y1 = pt1
#     x2, y2 = pt2
    
#     x_coords = np.linspace(x1, x2, num_samples).astype(int)
#     y_coords = np.linspace(y1, y2, num_samples).astype(int)
    
#     # Clip coordinates to image boundaries
#     h, w = img.shape[:2]
#     x_coords = np.clip(x_coords, 0, w-1)
#     y_coords = np.clip(y_coords, 0, h-1)
    
#     intensities = img[y_coords, x_coords]
#     return intensities


# def filter_linear_features(mask, min_length=15):
#     """Filter mask to keep only linear features"""
#     # Find contours
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     filtered_mask = np.zeros_like(mask)
    
#     for cnt in contours:
#         area = cv2.contourArea(cnt)
#         if area < 5:
#             continue
        
#         # Get bounding rectangle
#         x, y, w, h = cv2.boundingRect(cnt)
#         aspect_ratio = max(w, h) / (min(w, h) + 1)
        
#         # Keep elongated features
#         if aspect_ratio >= 2.0 or max(w, h) >= min_length:
#             cv2.drawContours(filtered_mask, [cnt], -1, 255, -1)
    
#     return filtered_mask


# # ===============================
# # EXAMPLE USAGE
# # ===============================

# if __name__ == "__main__":
#     # Example usage
#     test_image = "path/to/your/test_image.jpg"
#     roi_mask = "path/to/your/roi_mask.png"  # Optional
    
#     # Run detection with different sensitivity levels
#     for sens in ['low', 'medium', 'high']:
#         print(f"\nTesting with sensitivity: {sens}")
#         detected, count, scratch_contours = advanced_scratch_detection(
#             test_image, 
#             roi_mask_path=roi_mask,
#             sensitivity=sens
#         )









def complete_inspection_single_face(ref_image_path, test_image_path, face_name, device_type='C20', roi_mask_path=None, force_recreate_roi=False):
    """Complete inspection pipeline for a single face"""
    
    print(f"\n{'='*60}")
    print(f"🔍 INSPECTING FACE: {face_name}")
    print(f"{'='*60}")
    print(f"Reference: {ref_image_path}")
    print(f"Test: {test_image_path}")
    
    # Check if files exist
    if not os.path.exists(ref_image_path):
        print(f"❌ Reference image not found: {ref_image_path}")
        return None
        
    if not os.path.exists(test_image_path):
        print(f"❌ Test image not found: {test_image_path}")
        return None
    
    # Generate ROI mask filename based on DEVICE TYPE and face name
    if roi_mask_path is None:
        shared_folder = "roi_masks/shared"
        face_name_clean = face_name.replace(' ', '_').lower()
        roi_mask_path = os.path.join(shared_folder, f"roi_mask_{device_type.lower()}_{face_name_clean}.jpg")

    # Create ROI only if it doesn't exist or force_recreate_roi is True
    if force_recreate_roi or not os.path.exists(roi_mask_path):
        print(f"📐 Creating new ROI for {device_type} {face_name}...")
        face_name_clean = face_name.replace(' ', '_').lower()
        roi_result = create_roi_for_image(ref_image_path, f"roi_mask_{device_type.lower()}_{face_name_clean}")

        if roi_result is None:
            print(f"❌ ROI creation failed for {face_name}. Aborting inspection.")
            return None
    
        roi_mask_path, _ = roi_result
    else:
        print(f"✅ Using existing ROI mask: {roi_mask_path}")
    
    # Run all inspections
    results = {
        'face_name': face_name,
        'ref_path': ref_image_path,
        'test_path': test_image_path
    }
    
    # 1. Color inspection
    print(f"\n🎨 Running color inspection for {face_name}...")
    color_defect, delta_e = color_inspection(ref_image_path, test_image_path, roi_mask_path)
    results['color_defect'] = color_defect
    results['delta_e'] = delta_e
    
    # 2. Paint defect inspection
    print(f"\n🎯 Running paint defect inspection for {face_name}...")
    paint_detected, num_defects, defect_areas = advanced_paint_defect_detection(
                                                    ref_image_path,
                                                    test_image_path,
                                                    roi_mask_path
                                                )

    # 3. Scratch detection
    print(f"\n🔍 Running scratch detection for {face_name}...")
    scratch_detected, num_scratches, scratches = advanced_scratch_detection(test_image_path, roi_mask_path)

    # --- Exclude paint defects overlapping with scratches ---
    def contour_overlap(c1, c2):
        x1, y1, w1, h1 = cv2.boundingRect(c1)
        x2, y2, w2, h2 = cv2.boundingRect(c2)
        # Compute intersection
        xi1, yi1 = max(x1, x2), max(y1, y2)
        xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        inter_area = (xi2 - xi1) * (yi2 - yi1)
        area1 = w1 * h1
        area2 = w2 * h2
        # Overlap ratio wrt smaller box
        return inter_area / min(area1, area2)

    filtered_defects = []
    for defect in defect_areas:
        overlap = False
        for scratch in scratches:
            if contour_overlap(defect, scratch) > 0.3:  # 30% overlap threshold
                overlap = True
                break
        if not overlap:
            filtered_defects.append(defect)

    filtered_paint_detected = len(filtered_defects) > 0
    filtered_num_defects = len(filtered_defects)

    results['paint_defect'] = filtered_paint_detected
    results['num_defects'] = filtered_num_defects
    results['defect_areas'] = filtered_defects
    results['scratch_defect'] = scratch_detected
    results['num_scratches'] = num_scratches
    results['scratches'] = scratches

    # Face result
    face_failed = any([color_defect, filtered_paint_detected, scratch_detected])
    results['face_failed'] = face_failed

    print(f"\n{'='*60}")
    print(f"📊 RESULTS FOR {face_name}")
    print(f"{'='*60}")
    print(f"Color Defect:    {'FAIL' if color_defect else 'PASS'} (ΔE: {delta_e:.2f})")
    print(f"Paint Defect:    {'FAIL' if filtered_paint_detected else 'PASS'} ({filtered_num_defects} defects)")
    print(f"Scratch Defect:  {'FAIL' if scratch_detected else 'PASS'} ({num_scratches} scratches)")
    print(f"Face Result:     {'❌ FAIL' if face_failed else '✅ PASS'}")
    print(f"{'='*60}")

    return results


def create_all_roi_masks(face_configs):
    """
    Create ROI masks for all faces at once before testing
    
    Parameters:
    -----------
    face_configs : list of dict
        Each dict should have: {'face_name': str, 'ref_path': str}
    
    Returns:
    --------
    dict : Dictionary mapping face_name to roi_mask_path
    """
    
    print("\n" + "="*80)
    print("📐 ROI MASK CREATION MODE - CREATE ALL ROI MASKS FIRST")
    print(f"📦 Total faces: {len(face_configs)}")
    print("="*80)
    
    roi_masks = {}
    
    for i, config in enumerate(face_configs, 1):
        face_name = config['face_name']
        ref_path = config['ref_path']
        
        print(f"\n[{i}/{len(face_configs)}] Creating ROI for: {face_name.upper()}")
        print(f"Reference image: {ref_path}")
        
        # Check if file exists
        if not os.path.exists(ref_path):
            print(f"❌ Reference image not found: {ref_path}")
            continue
        
        # Check if shared ROI mask already exists
        shared_folder = "roi_masks/shared"
        face_name_clean = face_name.replace(' ', '_').lower()
        roi_mask_path = os.path.join(shared_folder, f"roi_mask_{face_name_clean}.jpg")

        if os.path.exists(roi_mask_path):
            response = input(f"⚠️  ROI mask already exists for {face_name}. Recreate? (y/n): ")
            if response.lower() != 'y':
                print(f"✅ Using existing ROI mask: {roi_mask_path}")
                roi_masks[face_name] = roi_mask_path
                continue
        
        # Create ROI
        roi_result = create_roi_for_image(ref_path, f"roi_mask_{face_name_clean}")
        
        if roi_result is None:
            print(f"❌ ROI creation failed for {face_name}")
            continue
        
        roi_mask_path, _ = roi_result
        roi_masks[face_name] = roi_mask_path
        print(f"✅ ROI created for {face_name}: {roi_mask_path}")
    
    print("\n" + "="*80)
    print(f"📊 ROI CREATION SUMMARY:")
    print(f"   Total ROIs created/confirmed: {len(roi_masks)}/{len(face_configs)}")
    print(f"   ROI masks: {list(roi_masks.values())}")
    print("="*80)
    
    return roi_masks


def complete_inspection_all_faces(face_configs, device_type='C20', create_new_roi=False):
    """
    Complete inspection for all faces of the device
    OPTIMIZED WORKFLOW: 
    1. Create ALL ROI masks first (one after another)
    2. Then test ALL faces continuously without interruption
    
    Parameters:
    -----------
    face_configs : list of dict
        Each dict should have: {'face_name': str, 'ref_path': str, 'test_path': str}
    create_new_roi : bool
        Whether to create new ROI masks (True) or use existing ones (False)
    """
    
    print("\n" + "="*80)
    print(f"🚀 STARTING MULTI-FACE INSPECTION FOR {device_type}")
    print(f"📦 Total faces to inspect: {len(face_configs)}")
    print("="*80)
    
    # ===============================
    # PHASE 1: CREATE/VERIFY ALL ROI MASKS FIRST
    # ===============================
    roi_masks = {}
    
    if create_new_roi:
        print("\n" + "="*80)
        print("🔧 PHASE 1: CREATING ALL ROI MASKS")
        print("="*80)
        print("📐 You will now draw ROI for all faces one by one...")
        print("   After all ROIs are drawn, testing will run automatically.\n")
        
        for i, config in enumerate(face_configs, 1):
            face_name = config['face_name']
            ref_path = config['ref_path']
            
            print(f"\n{'─'*60}")
            print(f"[{i}/{len(face_configs)}] Creating ROI for: {face_name.upper()}")
            print(f"{'─'*60}")
            print(f"Reference image: {ref_path}")
            
            # Check if file exists
            if not os.path.exists(ref_path):
                print(f"❌ Reference image not found: {ref_path}")
                continue
            
            # Check if ROI mask already exists
            shared_folder = "roi_masks/shared"
            face_name_clean = face_name.replace(' ', '_').lower()
            roi_mask_path = os.path.join(shared_folder, f"roi_mask_{device_type.lower()}_{face_name_clean}.jpg")
            
            if os.path.exists(roi_mask_path):
                print(f"⚠️  ROI mask already exists: {roi_mask_path}")
                response = input(f"   Recreate ROI for {face_name}? (y/n): ")
                if response.lower() != 'y':
                    print(f"✅ Using existing ROI mask")
                    roi_masks[face_name] = roi_mask_path
                    continue
            
            # Create ROI
            face_name_clean = face_name.replace(' ', '_').lower()
            roi_result = create_roi_for_image(ref_path, f"roi_mask_{device_type.lower()}_{face_name_clean}")
            
            if roi_result is None:
                print(f"❌ ROI creation failed for {face_name}")
                continue
            
            roi_mask_path, _ = roi_result
            roi_masks[face_name] = roi_mask_path
            print(f"✅ ROI saved: {roi_mask_path}")
        
        print("\n" + "="*80)
        print(f"✅ ROI CREATION COMPLETED!")
        print(f"   Total ROIs created/confirmed: {len(roi_masks)}/{len(face_configs)}")
        print("="*80)
        
        if len(roi_masks) == 0:
            print("❌ No ROI masks were created. Aborting inspection.")
            return None
        
        print("\n🟢 All ROI masks ready. Starting automated testing...\n")
        input("Press ENTER to begin testing all faces...")
        
    else:
        print("\n" + "="*80)
        print("🔧 PHASE 1: VERIFYING EXISTING ROI MASKS")
        print("="*80)
        
        missing_roi = []
        
        for config in face_configs:
            face_name = config['face_name']
            shared_folder = "roi_masks/shared"
            face_name_clean = face_name.replace(' ', '_').lower()
            roi_mask_path = os.path.join(shared_folder, f"roi_mask_{device_type.lower()}_{face_name_clean}.jpg")
            
            if os.path.exists(roi_mask_path):
                roi_masks[face_name] = roi_mask_path
                print(f"✅ {face_name:10} - ROI mask found: {roi_mask_path}")
            else:
                missing_roi.append(face_name)
                print(f"❌ {face_name:10} - ROI mask NOT found!")
        
        if missing_roi:
            print(f"\n⚠️  Missing ROI masks for: {', '.join(missing_roi)}")
            response = input("Create missing ROI masks now? (y/n): ")
            
            if response.lower() == 'y':
                print("\n📐 Creating missing ROI masks...")
                
                for face_name in missing_roi:
                    config = next((c for c in face_configs if c['face_name'] == face_name), None)
                    if config is None:
                        continue
                    
                    ref_path = config['ref_path']
                    print(f"\n[ROI] Creating for: {face_name.upper()}")
                    
                    if not os.path.exists(ref_path):
                        print(f"❌ Reference image not found: {ref_path}")
                        continue
                    
                    face_name_clean = face_name.replace(' ', '_').lower()
                    roi_result = create_roi_for_image(ref_path, f"roi_mask_{device_type.lower()}_{face_name_clean}")
                    
                    if roi_result is not None:
                        roi_mask_path, _ = roi_result
                        roi_masks[face_name] = roi_mask_path
                        print(f"✅ ROI created: {roi_mask_path}")
            else:
                print(f"⚠️  Will proceed without ROI masks for: {', '.join(missing_roi)}")
        
        print("\n" + "="*80)
        print(f"✅ ROI VERIFICATION COMPLETED!")
        print(f"   Available ROI masks: {len(roi_masks)}/{len(face_configs)}")
        print("="*80)
        print("\n🟢 Starting automated testing...\n")
    
    # ===============================
    # PHASE 2: TEST ALL FACES CONTINUOUSLY
    # ===============================
    print("\n" + "="*80)
    print("🧪 PHASE 2: AUTOMATED TESTING OF ALL FACES")
    print("="*80)
    print("⚡ Testing will now run continuously for all faces...")
    print("   No manual intervention required!\n")
    
    all_results = []
    failed_faces = []
    
    for i, config in enumerate(face_configs, 1):
        face_name = config['face_name']
        ref_path = config['ref_path']
        test_path = config['test_path']
        
        # Skip if no ROI mask
        if face_name not in roi_masks:
            print(f"\n⚠️  [{i}/{len(face_configs)}] Skipping {face_name.upper()} - No ROI mask available\n")
            continue
        
        roi_mask_path = roi_masks[face_name]
        
        print(f"\n{'='*80}")
        print(f"🔍 [{i}/{len(face_configs)}] TESTING: {face_name.upper()}")
        print(f"{'='*80}")
        
        # Check if files exist
        if not os.path.exists(ref_path):
            print(f"❌ Reference image not found: {ref_path}")
            continue
            
        if not os.path.exists(test_path):
            print(f"❌ Test image not found: {test_path}")
            continue
        
        # Run all inspections for this face
        results = {
            'face_name': face_name,
            'ref_path': ref_path,
            'test_path': test_path,
            'roi_mask_path': roi_mask_path
        }
        
        # 1. Color inspection
        print(f"\n🎨 Color inspection...")
        color_defect, delta_e = color_inspection(ref_path, test_path, roi_mask_path)
        results['color_defect'] = color_defect
        results['delta_e'] = delta_e
        
        # 2. Paint defect inspection
        print(f"\n🎯 Paint defect inspection...")
        paint_detected, num_defects, defect_areas = advanced_paint_defect_detection(ref_path, test_path, roi_mask_path)

        # 3. Scratch detection
        print(f"\n🔍 Scratch detection...")
        scratch_detected, num_scratches, scratches = advanced_scratch_detection(ref_path, test_path, roi_mask_path)

        # --- Exclude paint defects overlapping with scratches ---
        def contour_overlap(c1, c2):
            x1, y1, w1, h1 = cv2.boundingRect(c1)
            x2, y2, w2, h2 = cv2.boundingRect(c2)
            xi1, yi1 = max(x1, x2), max(y1, y2)
            xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
            if xi2 <= xi1 or yi2 <= yi1:
                return 0.0
            inter_area = (xi2 - xi1) * (yi2 - yi1)
            area1 = w1 * h1
            area2 = w2 * h2
            return inter_area / min(area1, area2)

        filtered_defects = []
        for defect in defect_areas:
            overlap = False
            for scratch in scratches:
                if contour_overlap(defect, scratch) > 0.3:
                    overlap = True
                    break
            if not overlap:
                filtered_defects.append(defect)

        filtered_paint_detected = len(filtered_defects) > 0
        filtered_num_defects = len(filtered_defects)

        results['paint_defect'] = filtered_paint_detected
        results['num_defects'] = filtered_num_defects
        results['defect_areas'] = filtered_defects
        results['scratch_defect'] = scratch_detected
        results['num_scratches'] = num_scratches
        results['scratches'] = scratches

        # Face result
        face_failed = any([color_defect, filtered_paint_detected, scratch_detected])
        results['face_failed'] = face_failed

        # Quick summary for this face
        print(f"\n{'─'*60}")
        print(f"📊 {face_name.upper()} RESULTS:")
        print(f"   Color:   {'FAIL ❌' if color_defect else 'PASS ✅'} (ΔE: {delta_e:.2f})")
        print(f"   Paint:   {'FAIL ❌' if filtered_paint_detected else 'PASS ✅'} ({filtered_num_defects} defects)")
        print(f"   Scratch: {'FAIL ❌' if scratch_detected else 'PASS ✅'} ({num_scratches} scratches)")
        print(f"   Result:  {'❌ FAIL' if face_failed else '✅ PASS'}")
        print(f"{'─'*60}")

        all_results.append(results)
        
        if face_failed:
            failed_faces.append(face_name)
    
    # ===============================
    # FINAL OVERALL RESULTS
    # ===============================
    overall_device_failed = len(failed_faces) > 0
    
    print("\n\n" + "="*80)
    print("🏁 FINAL OVERALL DEVICE INSPECTION RESULTS")
    print("="*80)
    
    print(f"\n{'FACE':<12} {'STATUS':<10} {'COLOR':<15} {'PAINT':<15} {'SCRATCH':<15}")
    print("─"*80)
    
    for result in all_results:
        face = result['face_name'].upper()
        status = "❌ FAIL" if result['face_failed'] else "✅ PASS"
        color_str = f"{'FAIL' if result['color_defect'] else 'PASS'} ({result['delta_e']:.2f})"
        paint_str = f"{'FAIL' if result['paint_defect'] else 'PASS'} ({result['num_defects']})"
        scratch_str = f"{'FAIL' if result['scratch_defect'] else 'PASS'} ({result['num_scratches']})"
        
        print(f"{face:<12} {status:<10} {color_str:<15} {paint_str:<15} {scratch_str:<15}")
    
    print("\n" + "="*80)
    print(f"📊 SUMMARY:")
    print(f"   Total Faces Tested:  {len(all_results)}")
    print(f"   Passed:              {len(all_results) - len(failed_faces)} ✅")
    print(f"   Failed:              {len(failed_faces)} ❌")
    if failed_faces:
        print(f"   Failed Faces:        {', '.join([f.upper() for f in failed_faces])}")
    print("="*80)
    print(f"\n🎯 FINAL VERDICT: {'❌ DEVICE REJECTED' if overall_device_failed else '✅ DEVICE ACCEPTED'}")
    print("="*80 + "\n")
    
    return {
        'all_results': all_results,
        'failed_faces': failed_faces,
        'overall_failed': overall_device_failed,
        'total_faces': len(all_results),
        'passed_faces': len(all_results) - len(failed_faces)
    }




# Usage with your original image paths:
if __name__ == "__main__":
    
    # ========================================
    # DEVICE TYPE SELECTION
    # ========================================
    print("\n" + "="*80)
    print("DEVICE QUALITY INSPECTION SYSTEM")
    print("="*80)
    print("\nSelect Device Type:")
    print("1. C20")
    print("2. C50/C60")
    
    device_choice = input("\nEnter choice (1 or 2): ").strip()
    
    # Define configurations for both devices
    face_configs_c20 = [
        {
            'face_name': 'front',
            'ref_path': 'Images/references/C20/ref_c20_front.jpg',
            'test_path': 'Images/tests/test_front.jpg'
        },
        {
            'face_name': 'bottom',
            'ref_path': 'Images/references/C20/ref_c20_bottom.jpg',
            'test_path': 'Images/tests/test_bottom.jpg'
        },
        {
            'face_name': 'left',
            'ref_path': 'Images/references/C20/ref_c20_left.jpg',
            'test_path': 'Images/tests/test_left.jpg'
        },
        {
            'face_name': 'right',
            'ref_path': 'Images/references/C20/ref_c20_right.jpg',
            'test_path': 'Images/tests/test_right.jpg'
        },
        {
            'face_name': 'top',
            'ref_path': 'Images/references/C20/ref_c20_top.jpg',
            'test_path': 'Images/tests/test_top.jpg'
        }
    ]
    
    face_configs_c50 = [
        {
            'face_name': 'front',
            'ref_path': 'Images/references/C50/ref_c50_front.jpg',
            'test_path': 'Images/tests/test_blackfront.jpg'
        },
        {
            'face_name': 'bottom',
            'ref_path': 'Images/references/C50/ref_c50_bottom.jpg',
            'test_path': 'Images/tests/test_blackbottom.jpg'
        },
        {
            'face_name': 'left',
            'ref_path': 'Images/references/C50/ref_c50_left.jpg',
            'test_path': 'Images/tests/test_blackleft.jpg'
        },
        {
            'face_name': 'right',
            'ref_path': 'Images/references/C50/ref_c50_right.jpg',
            'test_path': 'Images/tests/test_blackright.jpg'
        },
        {
            'face_name': 'top',
            'ref_path': 'Images/references/C50/ref_c50_top.jpg',
            'test_path': 'Images/tests/test_blacktop.jpg'
        }
    ]
    
    # Select device configuration
    if device_choice == '1':
        device_type = 'C20'
        face_configs = face_configs_c20
    elif device_choice == '2':
        device_type = 'C50'
        face_configs = face_configs_c50
    else:
        print("❌ Invalid choice. Exiting.")
        exit()
    
    print(f"\n✅ Selected Device: {device_type}")
    print(f"📁 Reference images folder: Images/references/{device_type}/")
    
    # Check if ROI masks exist for this device
    shared_folder = "roi_masks/shared"
    os.makedirs(shared_folder, exist_ok=True)
    
    roi_masks_exist = all(
        os.path.exists(os.path.join(shared_folder, f"roi_mask_{device_type.lower()}_{config['face_name'].replace(' ', '_').lower()}.jpg"))
        for config in face_configs
    )
    
    if roi_masks_exist:
        print(f"✅ All ROI masks found for {device_type}.")
        create_new_roi = False
    else:
        print(f"⚠️  Some ROI masks are missing for {device_type}. You will be prompted to create them.")
        create_new_roi = True
    
    # Run multi-face inspection with device type
    final_results = complete_inspection_all_faces(
        face_configs,
        device_type=device_type,
        create_new_roi=create_new_roi
    )
    
    if final_results:
        print(f"\n✅ {device_type} inspection completed successfully!")
        print(f"Device Status: {'REJECTED ❌' if final_results['overall_failed'] else 'ACCEPTED ✅'}")