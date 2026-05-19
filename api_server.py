from flask import Flask, request, jsonify, send_file, render_template_string
from flask_cors import CORS
import cv2
import numpy as np
import os
import base64
import json
from datetime import datetime
import threading
import uuid
import subprocess

# Import your inspection functions
from delta import (
    complete_inspection_single_face,
    create_roi_for_image,
    color_inspection,
    advanced_paint_defect_detection,
    advanced_scratch_detection
)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Session storage
sessions = {}
test_results = {}

# Ensure directories exist
os.makedirs("Images/tests", exist_ok=True)
os.makedirs("roi_masks/shared", exist_ok=True)

# Device configurations - CORRECTED FACE NAMES
DEVICE_CONFIGS = {
    'C20': {
        'front': 'Images/references/C20/ref_c20_front.jpg',
        'bottom': 'Images/references/C20/ref_c20_bottom.jpg',
        'left': 'Images/references/C20/ref_c20_left.jpg',
        'right': 'Images/references/C20/ref_c20_right.jpg',
        'top': 'Images/references/C20/ref_c20_top.jpg'
    },
    'C50': {
        'front': 'Images/references/C50/ref_c50_front.jpg',
        'bottom': 'Images/references/C50/ref_c50_bottom.jpg',
        'left': 'Images/references/C50/ref_c50_left.jpg',
        'right': 'Images/references/C50/ref_c50_right.jpg',
        'top': 'Images/references/C50/ref_c50_top.jpg'
    }
}

# CORRECTED FACE NAME MAPPING
FACE_NAMES = {
    'Front': 'front',
    'Bottom': 'bottom',
    'Left Side': 'left',
    'Right Side': 'right',
    'Top': 'top'
}

# Read the HTML file content
def load_html_template():
    try:
        with open('index.html', 'r', encoding='utf-8') as f:
            html_content = f.read()
            # Replace API_BASE_URL to use relative paths (same server)
            html_content = html_content.replace(
                "const API_BASE_URL = 'http://localhost:5000/api';",
                "const API_BASE_URL = '/api';"
            )
            return html_content
    except FileNotFoundError:
        return """
        <html>
        <body style="background: #0a0e1a; color: #fff; font-family: monospace; padding: 2rem;">
            <h1>❌ Error: index.html not found</h1>
            <p>Please ensure index.html is in the same directory as api_server.py</p>
            <p>Current directory: {}</p>
        </body>
        </html>
        """.format(os.getcwd())


# ========================================
# ROUTE: Serve the HTML UI (Main Page)
# ========================================
@app.route('/')
def index():
    """Serve the main HTML UI"""
    html_content = load_html_template()
    return render_template_string(html_content)


# ========================================
# API ROUTES
# ========================================

@app.route('/api/login', methods=['POST'])
def login():
    """Handle user login with job ID"""
    data = request.json
    job_id = data.get('jobId', '')
    
    if not job_id:
        return jsonify({'success': False, 'error': 'Job ID is required'}), 400
    
    # Create session
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        'jobId': job_id,
        'startTime': datetime.now().isoformat(),
        'deviceType': None,
        'capturedImages': []
    }
    
    print(f"[OK] Login successful - Job ID: {job_id}, Session ID: {session_id}")
    
    return jsonify({
        'success': True,
        'sessionId': session_id,
        'message': 'Login successful'
    })


@app.route('/api/select-device', methods=['POST'])
def select_device():
    """Select device type (C20 or C50/C60)"""
    data = request.json
    session_id = data.get('sessionId')
    device_type = data.get('deviceType')  # "C20" or "C50/C60"
    
    if session_id not in sessions:
        return jsonify({'success': False, 'error': 'Invalid session'}), 400
    
    # Normalize device type
    if device_type == 'C50/C60':
        device_type = 'C50'
    
    sessions[session_id]['deviceType'] = device_type
    sessions[session_id]['capturedImages'] = []
    
    print(f"[OK] Device selected - Session: {session_id}, Device: {device_type}")
    
    return jsonify({
        'success': True,
        'message': f'Device {device_type} selected'
    })


@app.route('/api/capture-image', methods=['POST'])
def capture_image():
    """Capture image from file upload or camera"""
    
    # Check if it's a file upload
    if 'image' in request.files:
        # Handle file upload
        session_id = request.form.get('sessionId')
        face_name = request.form.get('faceName')
        device_type = request.form.get('deviceType')
        
        if session_id not in sessions:
            return jsonify({'success': False, 'error': 'Invalid session'}), 400
        
        # Normalize device type
        if device_type == 'C50/C60':
            device_type = 'C50'
        
        # Get uploaded file
        file = request.files['image']
        
        if not file:
            return jsonify({'success': False, 'error': 'No file provided'}), 400
        
        # Map face name to internal name
        internal_face_name = FACE_NAMES.get(face_name, face_name.lower().replace(' ', '_'))
        
        # Create device-specific directory
        device_test_dir = f"Images/tests/{device_type}"
        os.makedirs(device_test_dir, exist_ok=True)
        
        # Save uploaded image with device-specific path
        test_image_path = f"{device_test_dir}/test_{internal_face_name}.jpg"
        file.save(test_image_path)
        
        print(f"[OK] Image uploaded - Device: {device_type}, Face: {face_name}, Path: {test_image_path}")
        
        # Read image and convert to base64 for frontend display
        img = cv2.imread(test_image_path)
        if img is None:
            return jsonify({'success': False, 'error': 'Failed to read uploaded image'}), 400
            
        _, buffer = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Store captured image info
        sessions[session_id]['capturedImages'].append({
            'faceName': face_name,
            'internalName': internal_face_name,
            'imagePath': test_image_path,
            'timestamp': datetime.now().isoformat()
        })
        
        return jsonify({
            'success': True,
            'imageData': f'data:image/jpeg;base64,{img_base64}',
            'imagePath': test_image_path,
            'message': f'{face_name} image captured'
        })
    
    # Handle JSON request (use existing test images)
    else:
        data = request.json
        session_id = data.get('sessionId')
        face_name = data.get('faceName')
        device_type = data.get('deviceType')
        
        if session_id not in sessions:
            return jsonify({'success': False, 'error': 'Invalid session'}), 400
        
        # Normalize device type
        if device_type == 'C50/C60':
            device_type = 'C50'
        
        # Map face name to internal name
        internal_face_name = FACE_NAMES.get(face_name, face_name.lower().replace(' ', '_'))
        
        # Use device-specific test image path
        test_image_path = f"Images/tests/{device_type}/test_black{internal_face_name}.jpg"
        
        # Check if test image exists
        if not os.path.exists(test_image_path):
            return jsonify({
                'success': False,
                'error': f'Test image not found: {test_image_path}. Please capture or upload an image first for {device_type}.'
            }), 400
        
        print(f"[OK] Using existing test image - Device: {device_type}, Face: {face_name}, Path: {test_image_path}")
        
        # Read image and convert to base64 for frontend display
        img = cv2.imread(test_image_path)
        _, buffer = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Store captured image info
        sessions[session_id]['capturedImages'].append({
            'faceName': face_name,
            'internalName': internal_face_name,
            'imagePath': test_image_path,
            'timestamp': datetime.now().isoformat()
        })
        
        return jsonify({
            'success': True,
            'imageData': f'data:image/jpeg;base64,{img_base64}',
            'imagePath': test_image_path,
            'message': f'{face_name} image loaded'
        })

@app.route('/api/capture-from-camera', methods=['POST'])
def capture_from_camera():
    """Capture image directly from Raspberry Pi camera"""
    data = request.json
    session_id = data.get('sessionId')
    face_name = data.get('faceName')
    device_type = data.get('deviceType')
    
    if session_id not in sessions:
        return jsonify({'success': False, 'error': 'Invalid session'}), 400
    
    # Normalize device type
    if device_type == 'C50/C60':
        device_type = 'C50'
    
    # Map face name to internal name
    internal_face_name = FACE_NAMES.get(face_name, face_name.lower().replace(' ', '_'))
    
    # Create device-specific directory
    device_test_dir = f"Images/tests/{device_type}"
    os.makedirs(device_test_dir, exist_ok=True)
    
    # Define output image path
    test_image_path = f"{device_test_dir}/test_{internal_face_name}.jpg"
    
    # Prepare the rpicam-still command
    camera_command = [
        'rpicam-still',
        '--roi', '0.25,0.25,0.5,0.5',
        '--width', '4680',
        '--height', '2592',
        '--awb', 'daylight',
        '--exposure', 'normal',
        '--autofocus-mode', 'auto',
        '--timeout', '5000',
        '-o', test_image_path
    ]
    
    try:
        print(f"[CAMERA] Capturing image with Pi Camera...")
        print(f"[CAMERA] Command: {' '.join(camera_command)}")
        
        # Execute the camera command
        result = subprocess.run(
            camera_command,
            capture_output=True,
            text=True,
            timeout=10  # 10 second timeout
        )
        
        if result.returncode != 0:
            error_msg = result.stderr if result.stderr else "Camera capture failed"
            print(f"[ERROR] Camera error: {error_msg}")
            return jsonify({
                'success': False,
                'error': f'Camera capture failed: {error_msg}'
            }), 500
        
        # Check if image was created
        if not os.path.exists(test_image_path):
            return jsonify({
                'success': False,
                'error': 'Image file not created by camera'
            }), 500
        
        print(f"[OK] Camera image captured - Device: {device_type}, Face: {face_name}, Path: {test_image_path}")
        
        # Read image and convert to base64 for frontend display
        img = cv2.imread(test_image_path)
        if img is None:
            return jsonify({
                'success': False,
                'error': 'Failed to read captured image'
            }), 500
        
        _, buffer = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Store captured image info
        sessions[session_id]['capturedImages'].append({
            'faceName': face_name,
            'internalName': internal_face_name,
            'imagePath': test_image_path,
            'timestamp': datetime.now().isoformat(),
            'captureMethod': 'camera'
        })
        
        return jsonify({
            'success': True,
            'imageData': f'data:image/jpeg;base64,{img_base64}',
            'imagePath': test_image_path,
            'message': f'{face_name} image captured from camera'
        })
        
    except subprocess.TimeoutExpired:
        print(f"[ERROR] Camera capture timeout")
        return jsonify({
            'success': False,
            'error': 'Camera capture timed out'
        }), 500
        
    except FileNotFoundError:
        print(f"[ERROR] rpicam-still command not found")
        return jsonify({
            'success': False,
            'error': 'rpicam-still not found. Is libcamera installed?'
        }), 500
        
    except Exception as e:
        print(f"[ERROR] Camera capture exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Camera error: {str(e)}'
        }), 500


@app.route('/api/process-images', methods=['POST'])
def process_images():
    """Process all captured images and run inspection"""
    data = request.json
    session_id = data.get('sessionId')
    device_type = data.get('deviceType')
    
    if session_id not in sessions:
        return jsonify({'success': False, 'error': 'Invalid session'}), 400
    
    # Normalize device type
    if device_type == 'C50/C60':
        device_type = 'C50'
    
    session = sessions[session_id]
    captured_images = session.get('capturedImages', [])
    
    if not captured_images:
        return jsonify({'success': False, 'error': 'No images captured'}), 400
    
    print(f"\n{'='*80}")
    print(f"[PROCESSING IMAGES] - Session: {session_id}, Device: {device_type}")
    print(f"{'='*80}")
    
    # Run inspection for each face
    results = []
    
    for img_data in captured_images:
        face_name = img_data['internalName']
        test_path = img_data['imagePath']
        
        print(f"\n[PROCESSING] {img_data['faceName']}")
        
        # Get reference image path for this device and face
        ref_path = DEVICE_CONFIGS.get(device_type, {}).get(face_name)
        
        if not ref_path or not os.path.exists(ref_path):
            print(f"[ERROR] Reference image not found: {ref_path}")
            results.append({
                'face': img_data['faceName'],
                'error': f'Reference image not found for {device_type} {face_name}',
                'overallPass': False
            })
            continue
        
        # Check if ROI exists
        shared_folder = "roi_masks/shared"
        face_name_clean = face_name.replace(' ', '_').lower()
        roi_mask_path = os.path.join(shared_folder, f"roi_mask_{device_type.lower()}_{face_name_clean}.jpg")
        
        if not os.path.exists(roi_mask_path):
            print(f"[ERROR] ROI mask not found: {roi_mask_path}")
            results.append({
                'face': img_data['faceName'],
                'error': f'ROI mask not found. Please create ROI masks first using: python3 delta.py',
                'overallPass': False
            })
            continue
        
        try:
            print(f"   Reference: {ref_path}")
            print(f"   Test: {test_path}")
            print(f"   ROI Mask: {roi_mask_path}")
            
            # Run color inspection
            print(f"   [COLOR] Running color inspection...")
            color_defect, delta_e = color_inspection(ref_path, test_path, roi_mask_path)
            print(f"      Delta E: {delta_e:.2f} - {'FAIL' if color_defect else 'PASS'}")
            
            # Run paint defect detection
            print(f"   [PAINT] Running paint defect detection...")
            paint_detected, num_defects, defect_areas = advanced_paint_defect_detection(
                ref_path, test_path, roi_mask_path
            )
            print(f"      Defects: {num_defects} - {'FAIL' if paint_detected else 'PASS'}")
            
            # Run scratch detection
            print(f"   [SCRATCH] Running scratch detection...")
            scratch_detected, num_scratches, scratches = advanced_scratch_detection(
                ref_path, test_path, roi_mask_path
            )
            print(f"      Scratches: {num_scratches} - {'FAIL' if scratch_detected else 'PASS'}")
            
            # Filter overlapping defects (same logic as backend)
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
            
            # Overall pass/fail
            overall_pass = not any([color_defect, filtered_paint_detected, scratch_detected])
            
            print(f"   [RESULT] {'PASS' if overall_pass else 'FAIL'}")
            
            results.append({
                'face': img_data['faceName'],
                'overallPass': overall_pass,
                'colorDifference': {
                    'value': round(delta_e, 2),
                    'pass': not color_defect,
                    'threshold': 2.0
                },
                'paintRemoval': {
                    'value': filtered_num_defects,
                    'pass': not filtered_paint_detected,
                    'threshold': 0
                },
                'scratches': {
                    'value': num_scratches,
                    'pass': not scratch_detected,
                    'threshold': 0
                }
            })
            
        except Exception as e:
            print(f"[ERROR] Error processing {img_data['faceName']}: {str(e)}")
            import traceback
            traceback.print_exc()
            
            results.append({
                'face': img_data['faceName'],
                'error': str(e),
                'overallPass': False
            })
    
    # Store results
    test_results[session_id] = {
        'timestamp': datetime.now().isoformat(),
        'deviceType': device_type,
        'jobId': session['jobId'],
        'results': results
    }
    
    print(f"\n{'='*80}")
    print(f"[DONE] PROCESSING COMPLETE")
    print(f"{'='*80}\n")
    
    return jsonify({
        'success': True,
        'results': results
    })


@app.route('/api/export-results', methods=['POST'])
def export_results():
    """Export test results as JSON"""
    data = request.json
    session_id = data.get('sessionId')
    
    if session_id not in test_results:
        return jsonify({'success': False, 'error': 'No results found'}), 400
    
    print(f"[EXPORT] Exporting results for session: {session_id}")
    
    return jsonify({
        'success': True,
        'data': test_results[session_id]
    })


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'active_sessions': len(sessions)
    })


if __name__ == '__main__':
    print("\n" + "="*80)
    print("[START] DEVICE QUALITY CONTROL SYSTEM")
    print("="*80)
    print("\n🌐 Server starting on: http://0.0.0.0:5000")
    print("\n📱 Access the UI at:")
    print("   • Local:    http://localhost:5000")
    print("   • Network:  http://<your-ip>:5000")
    print("\n📋 API Endpoints:")
    print("   POST /api/login             - Create session with job ID")
    print("   POST /api/select-device     - Select device type (C20/C50)")
    print("   POST /api/capture-image     - Capture/upload test image")
    print("   POST /api/capture-from-camera - Capture from Pi camera")
    print("   POST /api/process-images    - Run quality inspection")
    print("   POST /api/export-results    - Export results as JSON")
    print("   GET  /api/health            - Server health check")
    print("\n✅ Prerequisites:")
    print("   [*] Reference images in Images/references/C20/ and Images/references/C50/")
    print("   [*] ROI masks created (run: python3 delta.py to create)")
    print("   [*] flask-cors installed (pip install flask-cors)")
    print("   [*] index.html in the same directory as api_server.py")
    print("\n🚀 Starting server...")
    print("="*80 + "\n")
    
    # Get local IP address for display
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        print(f"💡 Your local IP: {local_ip}")
        print(f"   Access UI from other devices: http://{local_ip}:5000\n")
    except:
        pass
    
    app.run(host='0.0.0.0', port=5000, debug=True)