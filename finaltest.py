import cv2
import numpy as np
import os
import argparse
import glob
from pathlib import Path

# If True, skip interactive ROI drawing when a mask file already exists
# Default: True — prefer using existing masks to avoid repeated prompts. Set to False to force interactive drawing.
SKIP_ROI_IF_EXISTS = True
# If True, force redraw even if mask exists (overrides SKIP_ROI_IF_EXISTS)
FORCE_DRAW_ROI = False

DISPLAY_SCALE = 0.4   # scale for ROI drawing window
DELTA_E_VALUES = {}

# -------------------------------
# Helper functions (extracted)
# -------------------------------

def detect_paint(ref_img, test_img, roi_mask, face="FACE", display_scale=0.4,
                 diff_threshold=None, mean_diff_threshold=None, min_defect_area=30, max_defect_area=4000,
                 erosion_kernel_size=1, min_defect_percent=0.001, max_defect_percent=0.9,
                 verbose=False, save_debug=False, debug_dir="debug_output", display=True, fallback_to_intensity=True, intensity_dark_threshold=185,
                 paint_method='intensity'):
    """Detect paint removal by comparing reference and test images inside `roi_mask`.
    If `diff_threshold` or `mean_diff_threshold` are None, thresholds are computed adaptively
    from the reference-vs-test difference inside the ROI (robust to device/lighting).
    Returns (paint_removed: bool, info: dict).
    """
    info = {}

    # Ensure ROI matches image
    if roi_mask.shape[:2] != test_img.shape[:2]:
        roi_mask = cv2.resize(roi_mask, (test_img.shape[1], test_img.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Force binary mask
    roi_mask = roi_mask.astype(np.uint8)
    _, roi_mask = cv2.threshold(roi_mask, 127, 255, cv2.THRESH_BINARY)

    # If user requests intensity-only method (same as final.py), delegate to intensity detector
    if paint_method == 'intensity':
        return detect_paint_intensity(test_img, roi_mask, face=face, display_scale=display_scale,
                                      dark_threshold=intensity_dark_threshold, min_defect_area=min_defect_area,
                                      max_defect_area=max_defect_area, kernel_open_size=(3,3), kernel_close_size=(7,7),
                                      erosion_kernel_size=1, min_defect_percent=min_defect_percent,
                                      verbose=verbose, save_debug=save_debug, debug_dir=debug_dir, display=display)

    # Prepare grayscale images (diff-based method)
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    if ref_gray.shape != test_gray.shape:
        test_gray = cv2.resize(test_gray, (ref_gray.shape[1], ref_gray.shape[0]))

    # Smooth to reduce noise
    ref_blur = cv2.GaussianBlur(ref_gray, (5,5), 0)
    test_blur = cv2.GaussianBlur(test_gray, (5,5), 0)

    # Difference: where test is darker than reference
    diff = cv2.subtract(ref_blur, test_blur).astype(np.float32)

    # Adaptive thresholds (if not provided) using robust statistics inside ROI
    diff_vals = diff[roi_mask == 255].ravel()
    if diff_vals.size == 0:
        diff_vals = np.array([0.0], dtype=np.float32)

    mean_val = float(np.mean(diff_vals))
    std_val = float(np.std(diff_vals))

    p90 = float(np.percentile(diff_vals, 90.0))
    p95 = float(np.percentile(diff_vals, 95.0))
    p97 = float(np.percentile(diff_vals, 97.0))
    p99 = float(np.percentile(diff_vals, 99.0))

    if diff_threshold is None:
        # conservative: use a high percentile and mean+4*std (but capped if distribution small)
        diff_threshold = int(max(12, min(255, max(10, p99, mean_val + 4.0 * std_val))))
    if mean_diff_threshold is None:
        mean_diff_threshold = int(max(8, min(255, max(8, p95, mean_val + 3.0 * std_val))))

    info['adaptive_diff_threshold'] = diff_threshold
    info['adaptive_mean_diff_threshold'] = mean_diff_threshold
    info['diff_mean'] = mean_val
    info['diff_std'] = std_val
    info['diff_p90'] = p90
    info['diff_p95'] = p95
    info['diff_p97'] = p97
    info['diff_p99'] = p99

    if verbose:
        print(f"[paint-adapt] {face}: mean={mean_val:.2f}, std={std_val:.2f}, p90={p90:.2f}, p95={p95:.2f}, p99={p99:.2f}")
        print(f"[paint-adapt] {face}: chosen diff_threshold={diff_threshold}, mean_diff_threshold={mean_diff_threshold}")

    # Threshold the difference using chosen threshold
    _, paint_mask = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY)
    paint_mask = paint_mask.astype(np.uint8)

    # Keep detection strictly inside ROI
    paint_mask[roi_mask == 0] = 0

    # Diagnostics
    info['img_shape'] = test_img.shape
    info['mask_shape'] = roi_mask.shape
    info['mask_unique'] = np.unique(roi_mask).tolist()
    info['paint_pixels_total'] = int(np.sum(paint_mask == 255))

    # Morphology to clean noise
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    paint_mask = cv2.morphologyEx(paint_mask, cv2.MORPH_OPEN, kernel_open)
    paint_mask = cv2.morphologyEx(paint_mask, cv2.MORPH_CLOSE, kernel_close)

    # Erode ROI to avoid boundary artifacts (allow disabling erosion to capture edge defects)
    if erosion_kernel_size <= 1:
        eroded_roi = roi_mask.copy()
    else:
        ek = erosion_kernel_size if erosion_kernel_size % 2 == 1 else erosion_kernel_size + 1
        shrink_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ek, ek))
        eroded_roi = cv2.erode(roi_mask, shrink_kernel, iterations=1)

    info['paint_pixels_in_eroded'] = int(np.sum(cv2.bitwise_and(paint_mask, paint_mask, mask=eroded_roi) == 255))
    eroded_area = int(np.sum(eroded_roi == 255))

    # Filter paint mask by eroded ROI
    paint_mask_filtered = cv2.bitwise_and(paint_mask, paint_mask, mask=eroded_roi)

    # Contour analysis
    contours, _ = cv2.findContours(paint_mask_filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    paint_removed = False
    all_boxes = []
    paint_vis = cv2.bitwise_and(test_img, test_img, mask=roi_mask)

    total_defect_area = 0
    contour_details = []
    considered_cnts = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_defect_area or area > max_defect_area:
            continue

        # Check centroid inside eroded ROI
        M = cv2.moments(cnt)
        if M.get('m00', 0) == 0:
            continue
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        if eroded_roi[cy, cx] == 0:
            continue

        # Mean diff inside contour - helps reject lighting/global changes
        mask_cnt = np.zeros_like(paint_mask_filtered, dtype=np.uint8)
        cv2.drawContours(mask_cnt, [cnt], -1, 255, -1)
        mean_diff = float(np.mean(diff[mask_cnt == 255]))

        contour_details.append({'area': area, 'mean_diff': mean_diff})
        considered_cnts.append(cnt)

        # Relaxed filtering: accept contours with mean_diff >= 60% of the adaptive threshold
        if mean_diff < (mean_diff_threshold * 0.6):
            # likely minor variation, skip
            if verbose:
                print(f"[paint-skip] {face}: contour area={area:.1f}, mean_diff={mean_diff:.1f} < relaxed_mean_diff_threshold={(mean_diff_threshold*0.6):.1f}")
            continue

        paint_removed = True
        total_defect_area += area
        x, y, w, h = cv2.boundingRect(cnt)
        all_boxes.append((x, y, w, h))
        cv2.rectangle(paint_vis, (x, y), (x + w, y + h), (255, 0, 0), 2)

    info['contours'] = contour_details

    # Weak-signal fallback: if nothing passed strict thresholds but there exist contours
    # with mean_diff >= 75% of mean_diff_threshold and combined area is meaningful,
    # mark as a weak detection (helps not miss obvious defects under slightly low stats)
    if (not paint_removed) and len(contour_details) > 0:
        candidate_area = 0
        candidate_boxes = []
        for idx, d in enumerate(contour_details):
            if (d['mean_diff'] >= (0.75 * mean_diff_threshold)) and (d['area'] >= min_defect_area):
                candidate_area += d['area']
                cnt = considered_cnts[idx]
                x, y, w, h = cv2.boundingRect(cnt)
                candidate_boxes.append((x, y, w, h))

        if eroded_area > 0:
            candidate_percent = candidate_area / float(eroded_area)
        else:
            candidate_percent = 0

        if candidate_percent >= (min_defect_percent * 0.5) and candidate_area > 0:
            paint_removed = True
            all_boxes.extend(candidate_boxes)
            info['weak_detection'] = True
            info['weak_candidate_percent'] = candidate_percent
            if verbose:
                print(f"[paint-weak] {face}: weak_candidate_percent={candidate_percent:.4f} >= {(min_defect_percent * 0.5):.4f} -> marking weak paint_removed True")

    # If detected defect area is excessive relative to ROI, treat as suspicious (likely lighting)
    eroded_area = np.sum(eroded_roi == 255)
    if eroded_area > 0:
        defect_percent = total_defect_area / float(eroded_area)
    else:
        defect_percent = 0

    if defect_percent > max_defect_percent:
        # Too large to be a reasonable paint chip -> ignore as false positive
        paint_removed = False
        all_boxes = []

    # If detected defect area is very small relative to ROI, ignore as noise
    if defect_percent < min_defect_percent:
        paint_removed = False
        all_boxes = []

    # Recompute num_defects after filtering
    info['num_defects'] = len(all_boxes)
    info['boxes'] = all_boxes
    info['defect_percent'] = defect_percent

    # Optional big box
    if paint_removed and len(all_boxes) > 1:
        xs = [x for x, y, w, h in all_boxes]
        ys = [y for x, y, w, h in all_boxes]
        xe = [x + w for x, y, w, h in all_boxes]
        ye = [y + h for x, y, w, h in all_boxes]
        cv2.rectangle(paint_vis, (min(xs), min(ys)), (max(xe), max(ye)), (0, 0, 255), 2)

    # Recompute num_defects after filtering
    info['num_defects'] = len(all_boxes)
    info['boxes'] = all_boxes
    info['defect_percent'] = defect_percent

    # Save debug images if requested
    if save_debug:
        try:
            os.makedirs(debug_dir, exist_ok=True)
            # Save normalized diff for visualization
            diff_vis = np.clip((diff - diff.min()) / (diff.max() - diff.min() + 1e-8) * 255.0, 0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(debug_dir, f"{face}_diff.png"), diff_vis)
            cv2.imwrite(os.path.join(debug_dir, f"{face}_paint_mask.png"), paint_mask)
            cv2.imwrite(os.path.join(debug_dir, f"{face}_paint_filtered.png"), paint_mask_filtered)
            cv2.imwrite(os.path.join(debug_dir, f"{face}_paint_vis.png"), paint_vis)
            if verbose:
                print(f"[paint-debug] saved debug images to {debug_dir}")
        except Exception as e:
            print(f"[paint-debug] could not save debug images: {e}")

    # If we skipped everything due to thresholds, print summary when verbose
    if verbose and not paint_removed:
        print(f"[paint-summary] {face}: paint_pixels_total={info['paint_pixels_total']}, in_eroded={info['paint_pixels_in_eroded']}\n             contours_considered={len(contour_details)}, total_defect_area={total_defect_area:.1f}, defect_percent={defect_percent:.4f}")

    # Put status text
    status = "PAINT REMOVED" if paint_removed else "NO PAINT DEFECT"
    color = (0, 0, 255) if paint_removed else (0, 255, 0)
    cv2.putText(paint_vis, status, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    # Display diagnostics (only if display requested)
    try:
        if display:
            cv2.imshow(f"{face} ROI Mask", cv2.resize(roi_mask, None, fx=display_scale, fy=display_scale, interpolation=cv2.INTER_NEAREST))
            cv2.imshow(f"{face} Paint Mask", cv2.resize(paint_mask, None, fx=display_scale, fy=display_scale, interpolation=cv2.INTER_NEAREST))
            cv2.imshow(f"{face} Paint Filtered", cv2.resize(paint_mask_filtered, None, fx=display_scale, fy=display_scale, interpolation=cv2.INTER_NEAREST))
            cv2.imshow(f"{face} Paint Result", cv2.resize(paint_vis, None, fx=display_scale, fy=display_scale, interpolation=cv2.INTER_AREA))
    except Exception as e:
        print("Could not show paint windows:", e)

    print(f"[paint] {face}: mask pixels total={info['paint_pixels_total']}, in_eroded={info['paint_pixels_in_eroded']}, defects={info['num_defects']}, adaptive_thresh={info.get('adaptive_diff_threshold')}")

    # Wait and close (only if display requested)
    if display:
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return paint_removed, info


def detect_paint_intensity(test_img, roi_mask, face="FACE", display_scale=0.4,
                           dark_threshold=185, min_defect_area=30, max_defect_area=4000,
                           kernel_open_size=(3,3), kernel_close_size=(7,7),
                           erosion_kernel_size=1, min_defect_percent=0.001,
                           verbose=False, save_debug=False, debug_dir="debug_output", display=True):
    """Intensity-based paint detection (conservative fallback copied from `final.py`).
    Operates on the test image alone (no reference required). Returns (paint_removed, info).
    """
    info = {}

    # Ensure ROI matches image
    if roi_mask.shape[:2] != test_img.shape[:2]:
        roi_mask = cv2.resize(roi_mask, (test_img.shape[1], test_img.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Force binary mask
    roi_mask = roi_mask.astype(np.uint8)
    _, roi_mask = cv2.threshold(roi_mask, 127, 255, cv2.THRESH_BINARY)

    # Preprocessing
    roi = cv2.bitwise_and(test_img, test_img, mask=roi_mask)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)

    # Threshold dark areas (paint loss dark patches become white)
    _, paint_mask = cv2.threshold(gray, dark_threshold, 255, cv2.THRESH_BINARY_INV)

    # Keep detection strictly inside ROI
    paint_mask = cv2.bitwise_and(paint_mask, paint_mask, mask=roi_mask)

    info['paint_pixels_total'] = int(np.sum(paint_mask == 255))

    # Morphology
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_open_size)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_close_size)

    paint_mask = cv2.morphologyEx(paint_mask, cv2.MORPH_OPEN, kernel_open)
    paint_mask = cv2.morphologyEx(paint_mask, cv2.MORPH_CLOSE, kernel_close)

    # Erode ROI to avoid boundaries (allow disabling erosion to capture edge defects)
    if erosion_kernel_size <= 1:
        eroded_roi = roi_mask.copy()
    else:
        ek = erosion_kernel_size if erosion_kernel_size % 2 == 1 else erosion_kernel_size + 1
        shrink_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ek, ek))
        eroded_roi = cv2.erode(roi_mask, shrink_kernel, iterations=1)

    paint_mask_filtered = cv2.bitwise_and(paint_mask, paint_mask, mask=eroded_roi)

    contours, _ = cv2.findContours(paint_mask_filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    paint_removed = False
    all_boxes = []
    paint_vis = cv2.bitwise_and(test_img, test_img, mask=roi_mask)
    total_defect_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_defect_area or area > max_defect_area:
            continue

        # Accept contour if area is in valid range (no centroid-in-eroded check — match single-image logic)
        paint_removed = True
        total_defect_area += area
        x, y, w, h = cv2.boundingRect(cnt)
        all_boxes.append((x, y, w, h))
        cv2.rectangle(paint_vis, (x, y), (x + w, y + h), (255, 0, 0), 2)

    eroded_area = int(np.sum(eroded_roi == 255))
    defect_percent = (total_defect_area / float(eroded_area)) if eroded_area > 0 else 0

    # Intensity-based detection: follow single-image logic — accept any valid contour(s)
    info['num_defects'] = len(all_boxes)
    info['boxes'] = all_boxes
    info['defect_percent'] = defect_percent
    info['method'] = 'intensity'

    status = "PAINT REMOVED (INTENSITY)" if paint_removed else "NO PAINT DEFECT"
    color = (0, 0, 255) if paint_removed else (0, 255, 0)
    cv2.putText(paint_vis, status, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    if save_debug:
        try:
            os.makedirs(debug_dir, exist_ok=True)
            cv2.imwrite(os.path.join(debug_dir, f"{face}_intensity_mask.png"), paint_mask)
            cv2.imwrite(os.path.join(debug_dir, f"{face}_intensity_filtered.png"), paint_mask_filtered)
            cv2.imwrite(os.path.join(debug_dir, f"{face}_intensity_vis.png"), paint_vis)
        except Exception as e:
            if verbose:
                print(f"[paint-intensity-debug] could not save debug images: {e}")

    if display:
        try:
            cv2.imshow(f"{face} Intensity Mask", cv2.resize(paint_mask, None, fx=display_scale, fy=display_scale, interpolation=cv2.INTER_NEAREST))
            cv2.imshow(f"{face} Intensity Filtered", cv2.resize(paint_mask_filtered, None, fx=display_scale, fy=display_scale, interpolation=cv2.INTER_NEAREST))
            cv2.imshow(f"{face} Intensity Result", cv2.resize(paint_vis, None, fx=display_scale, fy=display_scale, interpolation=cv2.INTER_AREA))
            if display:
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        except Exception:
            pass

    return paint_removed, info


def detect_scratch(ref, test, roi_mask, face="FACE", display_scale=0.4,
                   block_size=16, edge_diff_threshold=None, min_disturbed_blocks=None, region_block_threshold=None,
                   method='variance', var_ratio_threshold=1.4, display=True):
    """Detect scratches using block-wise methods.
    method = 'both' (default) uses both Laplacian and variance checks for robustness.
    method = 'laplacian' uses Laplacian block differences.
    method = 'variance' uses variance ratio approach from final.py.
    Returns (scratch_detected, info).
    """
    info = {}

    # Ensure sizes
    if ref.shape != test.shape:
        test = cv2.resize(test, (ref.shape[1], ref.shape[0]))
    roi_mask = cv2.resize(roi_mask, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Smooth
    ref_blur = cv2.GaussianBlur(ref, (5,5), 0)
    test_blur = cv2.GaussianBlur(test, (5,5), 0)

    # Use Laplacian to emphasize edges (less sensitive to background texture)
    lap_ref = cv2.Laplacian(ref_blur, cv2.CV_64F)
    lap_test = cv2.Laplacian(test_blur, cv2.CV_64F)

    # Compute laplacian diffs inside ROI to set adaptive threshold if needed
    lap_diff = np.abs(lap_test - lap_ref)
    lap_vals = lap_diff[roi_mask == 255].ravel()
    if lap_vals.size == 0:
        lap_vals = np.array([0.0], dtype=np.float32)

    mean_lap = float(np.mean(lap_vals))
    std_lap = float(np.std(lap_vals))

    if edge_diff_threshold is None:
        # conservative: high percentile + mean+4*std
        edge_diff_threshold = max(10.0, float(np.percentile(lap_vals, 99.0)), mean_lap + 4.0 * std_lap)

    # Count total usable blocks inside ROI to compute default min_disturbed_blocks if needed
    total_blocks = 0
    h, w = ref_blur.shape
    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):
            block_mask = roi_mask[y:y+block_size, x:x+block_size]
            if cv2.countNonZero(block_mask) >= 0.5 * block_size * block_size:
                total_blocks += 1

    if min_disturbed_blocks is None:
        # For variance-based detection in final.py a fixed MIN_DISTURBED_BLOCKS=6 was used.
        # Use that as default when method is 'variance', else fall back to fraction-of-blocks.
        if method == 'variance':
            min_disturbed_blocks = 6
        else:
            min_disturbed_blocks = max(4, int(max(1, total_blocks * 0.02)))

    if region_block_threshold is None:
        region_block_threshold = 2

    info['adaptive_edge_diff_threshold'] = edge_diff_threshold
    info['adaptive_min_disturbed_blocks'] = min_disturbed_blocks
    info['computed_total_blocks'] = total_blocks
    info['lap_mean'] = mean_lap
    info['lap_std'] = std_lap

    disturbance_map = np.zeros_like(ref_blur, dtype=np.uint8)
    disturbed_blocks = 0

    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):
            block_mask = roi_mask[y:y+block_size, x:x+block_size]
            if cv2.countNonZero(block_mask) < 0.5 * block_size * block_size:
                continue

            lap_ref_block = lap_ref[y:y+block_size, x:x+block_size]
            lap_test_block = lap_test[y:y+block_size, x:x+block_size]

            # mean absolute difference in Laplacian domain
            block_diff = np.mean(np.abs(lap_test_block - lap_ref_block))

            if block_diff > edge_diff_threshold:
                disturbed_blocks += 1
                disturbance_map[y:y+block_size, x:x+block_size] = 255

    # Analyze clusters of disturbed blocks
    contours_blocks, _ = cv2.findContours(disturbance_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    region_blocks = [cv2.contourArea(c) / float(block_size * block_size) for c in contours_blocks]

    # Require both enough disturbed blocks and at least one sufficiently large cluster to confirm a laplacian-based scratch
    has_large_region = any(rb >= region_block_threshold for rb in region_blocks)
    lap_scratch_detected = (disturbed_blocks >= min_disturbed_blocks) and has_large_region

    info['lap_disturbed_blocks'] = disturbed_blocks
    info['disturbed_blocks'] = disturbed_blocks  # kept for compatibility (prefer lap_disturbed_blocks)
    info['lap_region_blocks'] = region_blocks
    info['lap_adaptive_edge_thresh'] = edge_diff_threshold

    # Optionally compute variance-based detection (from final.py)
    var_scratch_detected = False
    var_disturbed_blocks = 0
    if method in ('variance', 'both'):
        # Compute variance ratio over blocks
        var_disturb_map = np.zeros_like(ref_blur, dtype=np.uint8)
        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block_mask = roi_mask[y:y+block_size, x:x+block_size]
                if cv2.countNonZero(block_mask) < 0.5 * block_size * block_size:
                    continue

                ref_block = ref_blur[y:y+block_size, x:x+block_size]
                test_block = test_blur[y:y+block_size, x:x+block_size]

                var_ref = np.var(ref_block)
                var_test = np.var(test_block)

                if (var_test + 1) / (var_ref + 1) > var_ratio_threshold:
                    var_disturbed_blocks += 1
                    var_disturb_map[y:y+block_size, x:x+block_size] = 255

        var_scratch_detected = var_disturbed_blocks >= min_disturbed_blocks
        info['var_disturbed_blocks'] = var_disturbed_blocks

    # Final decision depending on requested method
    if method == 'laplacian':
        scratch_detected = lap_scratch_detected
    elif method == 'variance':
        scratch_detected = var_scratch_detected
    else:  # both
        scratch_detected = lap_scratch_detected or var_scratch_detected

    info['scratch_detected'] = bool(scratch_detected)

    # Visualization: combine maps if useful
    vis = cv2.cvtColor(test_blur, cv2.COLOR_GRAY2BGR)
    vis[disturbance_map > 0] = (0, 0, 255)
    if method in ('variance', 'both'):
        vis[var_disturb_map > 0] = (0, 255, 255)  # highlight var-detected blocks in yellow

    status = "SCRATCH DETECTED" if scratch_detected else "NO SCRATCH"
    color = (0,0,255) if scratch_detected else (0,255,0)
    cv2.putText(vis, status, (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    try:
        if display:
            cv2.imshow(f"{face} Test + ROI", cv2.bitwise_and(test_blur, test_blur, mask=roi_mask))
            cv2.imshow(f"{face} Disturbance Map", disturbance_map)
            cv2.imshow(f"{face} Scratch Result", cv2.resize(vis, None, fx=display_scale, fy=display_scale, interpolation=cv2.INTER_AREA))
    except Exception as e:
        print("Could not show scratch windows:", e)

    print(f"[scratch] {face}: lap_disturbed={disturbed_blocks}, var_disturbed={info.get('var_disturbed_blocks','?')}, lap_thresh={edge_diff_threshold}, var_ratio={var_ratio_threshold}, min_blocks={min_disturbed_blocks}")

    if display:
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return scratch_detected, info


def auto_tune_paint(ref_img, test_img, roi_mask, face="FACE", display_scale=0.4,
                    start_factors=(1.0, 0.8, 0.6, 0.4, 0.2),
                    verbose=False, debug_base_dir="debug_autotune"):
    """Auto-tune paint detection thresholds by trying scaled-down thresholds starting
    from adaptive estimates. Returns a dict of chosen params if a detection is found, else None.
    """
    # Compute diff stats similar to detect_paint without invoking display-wait
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    if ref_gray.shape != test_gray.shape:
        test_gray = cv2.resize(test_gray, (ref_gray.shape[1], ref_gray.shape[0]))

    ref_blur = cv2.GaussianBlur(ref_gray, (5,5), 0)
    test_blur = cv2.GaussianBlur(test_gray, (5,5), 0)
    diff = cv2.subtract(ref_blur, test_blur).astype(np.float32)

    diff_vals = diff[roi_mask == 255].ravel()
    if diff_vals.size == 0:
        diff_vals = np.array([0.0], dtype=np.float32)

    mean_val = float(np.mean(diff_vals))
    std_val = float(np.std(diff_vals))
    p95 = float(np.percentile(diff_vals, 95.0))
    p99 = float(np.percentile(diff_vals, 99.0))

    # Start from a robust 'suggested' baseline
    baseline_diff = int(max(12, min(255, max(p99, mean_val + 4.0 * std_val))))
    baseline_mean = int(max(8, min(255, max(p95, mean_val + 3.0 * std_val))))

    if verbose:
        print(f"[autotune] {face}: baseline_diff={baseline_diff}, baseline_mean={baseline_mean}, mean={mean_val:.2f}, std={std_val:.2f}, p99={p99:.2f}")

    # Try scale factors to be more permissive
    attempt = 0
    for f in start_factors:
        attempt_diff = max(4, int(baseline_diff * f))
        attempt_mean = max(4, int(baseline_mean * f))

        attempt_dir = os.path.join(debug_base_dir, f"{face}_attempt_{attempt_diff}_{attempt_mean}")
        os.makedirs(attempt_dir, exist_ok=True)

        # Run detect_paint with quiet display=False and save debug images
        pr, info = detect_paint(
            ref_img, test_img, roi_mask, face=face, display_scale=display_scale,
            diff_threshold=attempt_diff, mean_diff_threshold=attempt_mean,
            min_defect_area=10, max_defect_area=5000,
            erosion_kernel_size=1, min_defect_percent=0.001, max_defect_percent=0.9,
            verbose=verbose, save_debug=True, debug_dir=attempt_dir, display=False
        )

        if verbose:
            print(f"[autotune] {face}: attempt diff={attempt_diff}, mean={attempt_mean} -> paint_removed={pr}, info.num_defects={info.get('num_defects')} , defect_percent={info.get('defect_percent')}")

        # Accept only if real defects found (not too large and not zero)
        if pr and info.get('num_defects', 0) > 0 and info.get('defect_percent', 0.0) > 0:
            return {
                'diff_threshold': attempt_diff,
                'mean_diff_threshold': attempt_mean,
                'min_defect_area': 10,
                'max_defect_area': 5000,
                'erosion_kernel_size': 3,
                'min_defect_percent': 0.001,
                'debug_dir': attempt_dir,
                'auto_tuned': True
            }

        attempt += 1

    # If we reached here, no tuned thresholds produced detection
    if verbose:
        print(f"[autotune] {face}: no tuned thresholds produced detection")

    return None

# ===============================
# MULTI-FACE IMAGE PATHS
# ===============================
FACES = {
    # "top": {
    #     "ref": "Images/references/ref_top.jpg",
    #     "test": "Images/tests/test_top.jpg",
    #     "roi": "Images/roi_masks/top_mask.jpg"
    # },
    # "left": {
    #     "ref": "Images/references/ref_left.jpg",
    #     "test": "Images/tests/test_left.jpg",
    #     "roi": "Images/roi_masks/left_mask.jpg"
    # },
    # "right": {
    #     "ref": "Images/references/ref_right.jpg",
    #     "test": "Images/tests/test_right.jpg",
    #     "roi": "Images/roi_masks/right_mask.jpg"
    # },
    "bottom": {
        "ref": "Images/references/ref_bottom.jpg",
        "test": "Images/tests/test_bottom.jpg",
        "roi": "Images/roi_masks/bottom_mask.jpg",
        "params": {"verbose": True, "save_debug": True, "debug_dir": "debug_bottom", "autotune": True, "debug_base": "debug_autotune"}
    },
    # "front": {
    #     "ref": "Images/references/ref_front.jpg",
    #     "test": "Images/tests/test_front.jpg",
    #     "roi": "Images/roi_masks/front_mask.jpg"
    # }
}
# ===============================
# LOAD ORIGINAL IMAGE
# ===============================

def mouse_callback(event, x, y, flags, param):
    global display_clone, points_display, points_original

    if event == cv2.EVENT_LBUTTONDOWN:
        points_display.append((x, y))

        # map back to original image coordinates
        ox = int(x / DISPLAY_SCALE)
        oy = int(y / DISPLAY_SCALE)
        points_original.append((ox, oy))

        cv2.circle(display_clone, (x, y), 4, (0, 0, 255), -1)

        if len(points_display) > 1:
            cv2.line(
                display_clone,
                points_display[-2],
                points_display[-1],
                (0, 255, 0),
                2
            )

        cv2.imshow("Draw ROI", display_clone)


for face, paths in FACES.items():
    print(f"\n🟢 Draw ROI for {face.upper()}")

    # If mask already exists and skipping is enabled, don't prompt the user
    if SKIP_ROI_IF_EXISTS and os.path.exists(paths["roi"]) and not FORCE_DRAW_ROI:
        print(f"✅ ROI mask already exists for {face.upper()}, skipping drawing: {paths['roi']}")
        continue
    if FORCE_DRAW_ROI and os.path.exists(paths["roi"]):
        print(f"⚠️ FORCE_DRAW_ROI is True — existing mask will be overwritten for {face.upper()}")

    img = cv2.imread(paths["ref"])
    if img is None:
        continue

    h, w = img.shape[:2]

    points_display = []
    points_original = []

    display = cv2.resize(img, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
    display_clone = display.copy()

    cv2.namedWindow("Draw ROI", cv2.WINDOW_NORMAL)
    cv2.imshow("Draw ROI", display_clone)
    cv2.setMouseCallback("Draw ROI", mouse_callback)

    while True:
        if cv2.waitKey(1) == 13:
            break

    roi_mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array(points_original, dtype=np.int32)
    cv2.fillPoly(roi_mask, [pts], 255)

    cv2.imwrite(paths["roi"], roi_mask)
    print(f"✅ Saved {paths['roi']}")

    cv2.destroyAllWindows()
# ===============================
# DISPLAY SCALE (CHANGE IF NEEDED)
# ===============================








# ===============================
# Color Test Script
# ===============================
# ===============================
# PATHS
# ===============================
FINAL_FAIL = False
COLOR_DEFECT = {}   # face → True/False
PAINT_DEFECT = {}
SCRATCH_DEFECT = {}

# Logging & export options
EXPORT_CSV = False
CSV_PATH = "results.csv"

def _log_face_results(face):
    de = DELTA_E_VALUES.get(face, float('nan'))
    c = 'FAIL' if COLOR_DEFECT.get(face, False) else 'PASS'
    p = 'DEFECT' if PAINT_DEFECT.get(face, False) else 'OK'
    s = 'SCRATCH' if SCRATCH_DEFECT.get(face, False) else 'OK'
    overall = 'FAIL' if (COLOR_DEFECT.get(face, False) or PAINT_DEFECT.get(face, False) or SCRATCH_DEFECT.get(face, False)) else 'PASS'
    print(f"{face.upper():8} | COLOR: {c} (ΔE={de:.2f}) | PAINT: {p} | SCRATCH: {s} | {overall}")

# -----------------------------
# Command-line / batch processing
# -----------------------------
parser = argparse.ArgumentParser(description='Run inspection using fixed references/ROIs and variable test images (per-device).')
parser.add_argument('--device-dir', help='Directory containing per-device subfolders with test images (each device folder should contain images named per face)')
parser.add_argument('--test-file', help='Single test image path to inspect for one face (use with --face optionally)')
parser.add_argument('--face', help='Face name (when using --test-file), e.g. front, left, right')
parser.add_argument('--no-display', action='store_true', help='Do not show OpenCV windows (useful for batch runs)')
parser.add_argument('--verbose', action='store_true', help='Verbose output')
args = parser.parse_args()

# Helper to find test image for a face inside a device directory
def find_test_image_for_face(device_path, face_name):
    device_path = Path(device_path)
    if not device_path.exists():
        return None
    # Exact name tries
    exts = ['jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff']
    for ext in exts:
        p = device_path / f"{face_name}.{ext}"
        if p.exists():
            return str(p)
    # Fallback: case-insensitive substring match
    for p in device_path.iterdir():
        if p.is_file() and face_name.lower() in p.name.lower():
            return str(p)
    return None

# Build work items: list of tuples (device_name, face, paths)
work_items = []

if args.device_dir:
    root = Path(args.device_dir)
    for dev in sorted(root.iterdir()):
        if not dev.is_dir():
            continue
        dev_name = dev.name
        for face, base_paths in FACES.items():
            test_candidate = find_test_image_for_face(dev, face)
            if test_candidate:
                paths = dict(base_paths)
                paths['test'] = test_candidate
                work_items.append((dev_name, face, paths))
            else:
                if args.verbose:
                    print(f"[skip] {dev_name}/{face}: no test image found in {dev}")

elif args.test_file:
    tf = Path(args.test_file)
    if not tf.exists():
        print(f"Test file not found: {tf}")
        import sys
        sys.exit(1)
    if args.face:
        face = args.face
    else:
        face = next((f for f in FACES.keys() if f.lower() in tf.name.lower()), None)
    if face is None:
        print("Could not infer face from filename; please provide --face")
        import sys
        sys.exit(1)
    base_paths = FACES[face]
    paths = dict(base_paths)
    paths['test'] = str(tf)
    work_items.append((tf.stem, face, paths))

else:
    # Default behavior: use the test paths specified in FACES (single-device mode)
    for face, base_paths in FACES.items():
        paths = dict(base_paths)
        work_items.append(('default', face, paths))

if not work_items:
    print("No test images found to process. Provide --device-dir or --test-file, or add test paths to FACES.")
    import sys
    sys.exit(0)

# Iterate over devices/faces
for device_name, face, paths in work_items:
    print(f"\n🔍 Inspecting {device_name} / {face.upper()} face")

    ref_path = paths["ref"]
    test_path = paths["test"]
    roi_mask_path = paths["roi"]
    # override display flag from CLI
    if args.no_display:
        paths['display'] = False
    else:
        paths['display'] = paths.get('display', True)

    # ===============================
    # LOAD IMAGES
    # ===============================
    ref = cv2.imread(ref_path)
    test = cv2.imread(test_path)
    roi_mask = cv2.imread(roi_mask_path, cv2.IMREAD_GRAYSCALE)

    if ref is None or test is None or roi_mask is None:
        print("❌ Error loading images or ROI mask")
        exit()

    # Debug print to confirm which images were actually loaded
    print(f"[load] {face.upper()}: ref={ref_path} shape={None if ref is None else ref.shape} | test={test_path} shape={None if test is None else test.shape} | roi={roi_mask_path} exists={os.path.exists(roi_mask_path)}")

    # Ensure same size
    if ref.shape != test.shape:
        test = cv2.resize(test, (ref.shape[1], ref.shape[0]))

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
    DELTA_E_VALUES[face] = delta_e
    color_defect = (delta_e > DELTA_E_LIMIT)
    COLOR_DEFECT[face] = bool(color_defect)
    if color_defect:
        FINAL_FAIL = True
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




        # ===============================
    #   Paint Test Script (refactored)
    # ===============================
    # PARAMETERS (TUNED FOR YOUR PART)
    DARK_THRESHOLD = 185        # higher = detect lighter paint loss
    MIN_DEFECT_AREA = 30        # smallest valid paint chip
    MAX_DEFECT_AREA = 4000      # reject large regions (edges/shadows)

    params = paths.get('params', {})
    # If autotune is requested for this face, try to find permissive-but-valid thresholds
    tuned = None
    if params.get('autotune', False):
        tuned = auto_tune_paint(ref, test, roi_mask, face=face.upper(), display_scale=DISPLAY_SCALE, verbose=params.get('verbose', False), debug_base_dir=params.get('debug_base', 'debug_autotune'))
        if tuned is not None:
            print(f"[autotune] {face}: using tuned params: {tuned}")

    # Allow per-face selection of paint method (default 'intensity' to match final.py)
    paint_method = params.get('paint_method', 'intensity')

    # If we got tuned params, use them; otherwise use provided params (None triggers adaptive inside detect_paint)
    if tuned:
        paint_removed, paint_info = detect_paint(
            ref,
            test,
            roi_mask,
            face=face.upper(),
            display_scale=DISPLAY_SCALE,
            diff_threshold=tuned.get('diff_threshold'),
            mean_diff_threshold=tuned.get('mean_diff_threshold'),
            min_defect_percent=tuned.get('min_defect_percent', params.get('min_defect_percent', 0.001)),
            min_defect_area=tuned.get('min_defect_area', params.get('min_defect_area', MIN_DEFECT_AREA)),
            max_defect_area=tuned.get('max_defect_area', params.get('max_defect_area', MAX_DEFECT_AREA)),
            erosion_kernel_size=tuned.get('erosion_kernel_size', params.get('erosion_kernel_size', 1)),
            verbose=params.get('verbose', False),
            save_debug=True,
            debug_dir=tuned.get('debug_dir', params.get('debug_dir', 'debug_output')),
            display=params.get('display', True),
            fallback_to_intensity=params.get('fallback_to_intensity', True),
            intensity_dark_threshold=params.get('intensity_dark_threshold', 185)
        )
    else:
        paint_removed, paint_info = detect_paint(
            ref,
            test,
            roi_mask,
            face=face.upper(),
            display_scale=DISPLAY_SCALE,
            diff_threshold=params.get('diff_threshold', None),
            mean_diff_threshold=params.get('mean_diff_threshold', None),
            min_defect_percent=params.get('min_defect_percent', 0.001),
            min_defect_area=params.get('min_defect_area', MIN_DEFECT_AREA),
            max_defect_area=params.get('max_defect_area', MAX_DEFECT_AREA),
            erosion_kernel_size=params.get('erosion_kernel_size', 1),
            verbose=params.get('verbose', False),
            save_debug=params.get('save_debug', False),
            debug_dir=params.get('debug_dir', 'debug_output'),
            display=params.get('display', True),
            fallback_to_intensity=params.get('fallback_to_intensity', True),
            intensity_dark_threshold=params.get('intensity_dark_threshold', 185),
            paint_method=paint_method
        )
    print("PAINT REMOVED:", paint_removed)
    PAINT_DEFECT[face] = bool(paint_removed)
    if paint_removed:
        FINAL_FAIL = True


    # ===============================
    #   Scratch (refactored call)
    # ===============================

    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)

    scratch_detected, scratch_info = detect_scratch(
        ref_gray,
        test_gray,
        roi_mask,
        face=face.upper(),
        display_scale=DISPLAY_SCALE,
        block_size=params.get('block_size', 16),
        edge_diff_threshold=params.get('edge_diff_threshold', None),
        min_disturbed_blocks=params.get('min_disturbed_blocks', None),
        region_block_threshold=params.get('region_block_threshold', None),
        method=params.get('scratch_method', 'variance'),
        var_ratio_threshold=params.get('var_ratio_threshold', 1.4),
        display=params.get('display', True)
    )

    # Print counts similar to single-image script
    print("Disturbed blocks (lap,var):", scratch_info.get('lap_disturbed_blocks', '?'), scratch_info.get('var_disturbed_blocks', '?'))
    print("SCRATCH DETECTED:", scratch_detected)

    # ---- STORE RESULT ----
    SCRATCH_DEFECT[face] = bool(scratch_detected)
    if scratch_detected:
        FINAL_FAIL = True

    print(f"{face.upper()} → Disturbed blocks (lap,var):", scratch_info.get('lap_disturbed_blocks', '?'), scratch_info.get('var_disturbed_blocks', '?'))
    print(f"{face.upper()} → SCRATCH DETECTED:", scratch_detected)

    # Per-face concise log
    _log_face_results(face)

    # Ensure windows are cleared before next face
    cv2.destroyAllWindows()

# ===============================
# FINAL VISUAL CONFIRMATION
# ===============================

# if COLOR_DEFECT or PAINT_DEFECT or SCRATCH_DEFECT:
#     FINAL_FAIL = True

# FINAL_RESULT = "FAIL" if FINAL_FAIL else "PASS"
# result_color = (0, 0, 255) if FINAL_FAIL else (0, 255, 0)

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

# FINAL_RESULT = "FAIL" if FINAL_FAIL else "PASS"

# print("\n========== FINAL SUMMARY ==========")
# print("FINAL RESULT:", FINAL_RESULT)
# (Flags already initialized before the main inspection loop)

# for face, paths in FACES.items():
#     print(f"\n🔍 Inspecting {face.upper()} face")
    
#     # ... your existing color test code ...
#     COLOR_DEFECT = delta_e > DELTA_E_LIMIT
#     COLOR_DEFECTS[face] = COLOR_DEFECT
#     if COLOR_DEFECT:
#         FINAL_FAIL = True
    
#     # ... your existing paint test code ...
#     PAINT_DEFECT = paint_removed
#     PAINT_DEFECTS[face] = PAINT_DEFECT
#     if PAINT_DEFECT:
#         FINAL_FAIL = True
    
#     # ... your existing scratch test code ...
#     SCRATCH_DEFECT = scratch_detected
#     SCRATCH_DEFECTS.append((face, SCRATCH_DEFECT))
#     if SCRATCH_DEFECT:
#         FINAL_FAIL = True
    
#     # Close all windows after each face
#     cv2.destroyAllWindows()

# ===============================
# NOW SAFE - Use FINAL_FAIL
# ===============================
FINAL_RESULT = "FAIL" if FINAL_FAIL else "PASS"
print("\n========== FINAL SUMMARY ==========")
print("COLOR DEFECTS:", [face for face, defect in COLOR_DEFECT.items() if defect])
print("PAINT DEFECTS:", [face for face, defect in PAINT_DEFECT.items() if defect])
print("SCRATCH DEFECTS:", [face for face, defect in SCRATCH_DEFECT.items() if defect])
print("FINAL RESULT:", FINAL_RESULT)

print("\n========== FINAL SUMMARY ==========")

