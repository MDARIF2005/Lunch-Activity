from flask import Flask, render_template, request, redirect, url_for, jsonify, Response, send_file
import webbrowser
import os
import cv2
import threading
import time
import speech_recognition as sr
import qrcode
from io import BytesIO
import base64
from datetime import datetime, timedelta
import json
from werkzeug.utils import secure_filename
import schedule
import time as time_module
import numpy as np

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['VIDEO_FOLDER'] = 'static/video'
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024  # 64MB max file size

# Ensure upload directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['VIDEO_FOLDER'], exist_ok=True)

# Global variables to store launch data and camera
launch_data = {
    "type": "website",  # Default to website
    "value": "https://www.google.com",  # Default website
    "trigger": "hand"  # Default trigger
}

camera = None
camera_lock = threading.Lock()
hand_detected_at = 0.0
hand_inside_at = 0.0
current_centroid = (0, 0)
# Target region as percentages of frame size
hand_target_region = {"x": 0.35, "y": 0.30, "w": 0.30, "h": 0.40}
hand_detected_at = 0.0

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp', 'mp4', 'webm', 'ogg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Global variables for hand detection and countdown
countdown_active = False
countdown_start = 0.0
hand_sketch = None
launch_triggered = False

def load_hand_sketch():
    """Load the hand sketch image for overlay with transparency"""
    global hand_sketch
    try:
        sketch_path = os.path.join('static', 'images', 'hand_sketch.png')
        if os.path.exists(sketch_path):
            # Load with alpha channel to preserve transparency
            hand_sketch = cv2.imread(sketch_path, cv2.IMREAD_UNCHANGED)
            if hand_sketch is not None:
                # If image has 3 channels (BGR), add alpha channel
                if hand_sketch.shape[2] == 3:
                    # Create alpha channel - make white background transparent
                    alpha = np.ones((hand_sketch.shape[0], hand_sketch.shape[1]), dtype=np.uint8) * 255
                    
                    # Make white pixels transparent (assuming white background)
                    gray = cv2.cvtColor(hand_sketch, cv2.COLOR_BGR2GRAY)
                    white_mask = gray > 240  # White pixels
                    alpha[white_mask] = 0  # Make white transparent
                    
                    # Combine BGR with alpha
                    hand_sketch = np.dstack([hand_sketch, alpha])
                    print("Hand sketch loaded with transparency")
                else:
                    print("Hand sketch loaded with existing alpha channel")
            else:
                print("Failed to load hand sketch")
        else:
            print("Hand sketch file not found")
    except Exception as e:
        print(f"Error loading hand sketch: {e}")

# Load hand sketch on startup
load_hand_sketch()

def gen_frames():
    """Generate video frames for camera feed with hand sketch overlay and countdown"""
    global camera, hand_detected_at, hand_inside_at, current_centroid, hand_target_region
    global countdown_start, countdown_active, hand_sketch, launch_triggered
    
    with camera_lock:
        if camera is None:
            # Try multiple camera indices
            for i in range(3):
                try:
                    camera = cv2.VideoCapture(i)
                    if camera.isOpened():
                        print(f"Camera {i} initialized successfully")
                        break
                    else:
                        camera.release()
                        camera = None
                except:
                    camera = None
                    continue
    
    while True:
        if camera is None:
            # Create demo frame when no camera is available
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame[:] = (50, 50, 50)  # Dark gray background
            
            # Add gradient effect
            for i in range(480):
                intensity = int(50 + (i / 480) * 100)
                frame[i, :] = (intensity, intensity, intensity)
            
            # Add demo text
            cv2.putText(frame, "Camera not available", (200, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, "Using demo mode", (220, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
            
            # No demo target area - just the hand sketch
            
            # Show hand sketch if available
            if hand_sketch is not None:
                try:
                    sketch_h, sketch_w = hand_sketch.shape[:2]
                    scale = min(200 / sketch_w, 200 / sketch_h)
                    new_w = int(sketch_w * scale)
                    new_h = int(sketch_h * scale)
                    
                    target_x = (640 - new_w) // 2
                    target_y = (480 - new_h) // 2
                    
                    sketch_resized = cv2.resize(hand_sketch, (new_w, new_h))
                    
                    if hand_sketch.shape[2] == 4:  # Has alpha channel
                        alpha = sketch_resized[:, :, 3] / 255.0
                        for c in range(3):
                            frame[target_y:target_y+new_h, target_x:target_x+new_w, c] = \
                                frame[target_y:target_y+new_h, target_x:target_x+new_w, c] * (1 - alpha) + \
                                sketch_resized[:, :, c] * alpha
                    else:
                        # Make it semi-transparent in demo mode too
                        alpha = 0.9
                        for c in range(3):
                            frame[target_y:target_y+new_h, target_x:target_x+new_w, c] = \
                                frame[target_y:target_y+new_h, target_x:target_x+new_w, c] * (1 - alpha) + \
                                sketch_resized[:, :, c] * alpha
                    
                    # No border - just the clean hand sketch outline
                except Exception as e:
                    print(f"Error overlaying hand sketch: {e}")
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.1)
            continue
        
        success, frame = camera.read()
        if not success:
            # Camera failed, create demo frame
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame[:] = (50, 50, 50)
            cv2.putText(frame, "Camera error - Demo mode", (180, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Get frame dimensions
            h_frame, w_frame = frame.shape[:2]
            
            # Improved hand detection using multiple color ranges
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Multiple skin color ranges for better detection
            lower_skin1 = (0, 20, 70)
            upper_skin1 = (20, 255, 255)
            lower_skin2 = (170, 20, 70)
            upper_skin2 = (180, 255, 255)
            lower_skin3 = (0, 30, 60)
            upper_skin3 = (15, 255, 255)
            
            mask1 = cv2.inRange(hsv, lower_skin1, upper_skin1)
            mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
            mask3 = cv2.inRange(hsv, lower_skin3, upper_skin3)
            skin_mask = cv2.bitwise_or(cv2.bitwise_or(mask1, mask2), mask3)
            
            # Better noise reduction
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
            skin_mask = cv2.GaussianBlur(skin_mask, (5, 5), 0)
            
            # Find contours
            contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            detected = False
            inside_target = False
            
            if contours:
                # Sort contours by area and take the largest
                contours = sorted(contours, key=cv2.contourArea, reverse=True)
                largest = contours[0]
                area = cv2.contourArea(largest)
                # Lower threshold for better detection
                if area > 5000:  # Reduced threshold
                    detected = True
                    x, y, w, h = cv2.boundingRect(largest)
                    cx = x + w // 2
                    cy = y + h // 2
                    current_centroid = (cx, cy)
                    # Check if hand is in target area (hand sketch area)
                    if hand_sketch is not None:
                        try:
                            hand_size = min(w_frame, h_frame) // 3
                            target_x = (w_frame - hand_size) // 2
                            target_y = (h_frame - hand_size) // 2
                            hand_left = x
                            hand_right = x + w
                            hand_top = y
                            hand_bottom = y + h
                            target_left = target_x
                            target_right = target_x + hand_size
                            target_top = target_y
                            target_bottom = target_y + hand_size
                            overlap = (hand_left < target_right and hand_right > target_left and 
                                     hand_top < target_bottom and hand_bottom > target_top)
                            if overlap:
                                overlap_left = max(hand_left, target_left)
                                overlap_right = min(hand_right, target_right)
                                overlap_top = max(hand_top, target_top)
                                overlap_bottom = min(hand_bottom, target_bottom)
                                overlap_width = max(0, overlap_right - overlap_left)
                                overlap_height = max(0, overlap_bottom - overlap_top)
                                overlap_area = overlap_width * overlap_height
                                hand_area = w * h
                                overlap_percentage = overlap_area / hand_area if hand_area > 0 else 0
                                print(f"[DEBUG] Overlap: {overlap_percentage:.2f}, Hand area: {hand_area}")
                                if overlap_percentage >= 0.02 and hand_area >= 1500:
                                    inside_target = True
                                    if hand_inside_at == 0:
                                        hand_inside_at = time.time()
                                        print(f"Hand detected in target area - overlap: {overlap_percentage:.2f}, area: {hand_area}")
                                    time_inside = time.time() - hand_inside_at
                                    if not countdown_active and time_inside >= 0.05:
                                        countdown_active = True
                                        countdown_start = time.time()
                                        print(f"Starting countdown - hand held for {time_inside:.2f} seconds")
                                    elif countdown_active:
                                        pass
                                else:
                                    inside_target = False
                                    hand_inside_at = 0
                                    countdown_active = False
                                    countdown_start = 0
                            else:
                                inside_target = False
                                hand_inside_at = 0
                                countdown_active = False
                                countdown_start = 0
                        except Exception as e:
                            print(f"Error in hand sketch detection: {e}")
                    else:
                        inside_target = False
                        hand_inside_at = 0
                        countdown_active = False
                        countdown_start = 0
                else:
                    # If area is too small, treat as not detected
                    inside_target = False
                    hand_inside_at = 0
                    countdown_active = False
                    countdown_start = 0
            else:
                # No contours, treat as not detected
                inside_target = False
                hand_inside_at = 0
                countdown_active = False
                countdown_start = 0
            
            # Update hand detection time
            if detected:
                hand_detected_at = time.time()
            
            # Handle countdown display and launch
            if countdown_active:
                elapsed = time.time() - countdown_start
                remaining = max(0, 3.0 - elapsed)
                if remaining > 0:
                    countdown_number = int(remaining) + 1
                    if countdown_number > 3:
                        countdown_number = 3
                    cv2.putText(frame, str(countdown_number), (w_frame//2 - 50, h_frame//2), 
                              cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 6)
                    cv2.putText(frame, "Hold steady...", (w_frame//2 - 80, h_frame//2 + 60), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    bar_width = 300
                    bar_height = 20
                    bar_x = w_frame//2 - bar_width//2
                    bar_y = h_frame//2 + 100
                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
                    progress = (3.0 - remaining) / 3.0
                    progress_width = int(bar_width * progress)
                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), (0, 255, 0), -1)
                else:
                    # Countdown finished - only trigger launch if hand is still inside target
                    if not launch_triggered and inside_target:
                        cv2.putText(frame, "LAUNCHING!", (w_frame//2 - 80, h_frame//2), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                        launch_triggered = True
                        print("Countdown completed - triggering launch (hand inside target)")
                    countdown_active = False
                    countdown_start = 0
                    hand_inside_at = 0
            
            # Always show the hand sketch as a target overlay (larger size)
            if hand_sketch is not None:
                try:
                    hand_size = min(w_frame, h_frame) // 3
                    target_x = (w_frame - hand_size) // 2
                    target_y = (h_frame - hand_size) // 2
                    sketch_h, sketch_w = hand_sketch.shape[:2]
                    scale = min(hand_size / sketch_w, hand_size / sketch_h)
                    new_w = int(sketch_w * scale)
                    new_h = int(sketch_h * scale)
                    sketch_x = target_x + (hand_size - new_w) // 2
                    sketch_y = target_y + (hand_size - new_h) // 2
                    sketch_resized = cv2.resize(hand_sketch, (new_w, new_h))
                    if hand_sketch.shape[2] == 4:
                        alpha = sketch_resized[:, :, 3] / 255.0
                        for c in range(3):
                            frame[sketch_y:sketch_y+new_h, sketch_x:sketch_x+new_w, c] = \
                                frame[sketch_y:sketch_y+new_h, sketch_x:sketch_x+new_w, c] * (1 - alpha) + \
                                sketch_resized[:, :, c] * alpha
                    else:
                        alpha = 0.9
                        for c in range(3):
                            frame[sketch_y:sketch_y+new_h, sketch_x:sketch_x+new_w, c] = \
                                frame[sketch_y:sketch_y+new_h, sketch_x:sketch_x+new_w, c] * (1 - alpha) + \
                                sketch_resized[:, :, c] * alpha
                except Exception as e:
                    print(f"Error overlaying hand sketch: {e}")
            
            # Display status text
            if detected:
                if inside_target:
                    cv2.putText(frame, "Hand detected on sketch - Hold steady!", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "Move your hand to the hand sketch", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            else:
                cv2.putText(frame, "Show your hand to trigger", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Proximity feature removed

@app.route('/')
def index():
    """Main selection page"""
    return render_template('index.html')

@app.route('/store_selection', methods=['POST'])
def store_selection():
    """Store user's selection and redirect to appropriate trigger workflow"""
    global launch_data
    
    launch_type = request.form.get('launch_type')
    trigger_type = request.form.get('trigger_type')
    
    if launch_type == 'website':
        value = request.form.get('website_url')
    elif launch_type == 'image':
        if 'image_file' in request.files:
            file = request.files['image_file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                ext = filename.rsplit('.', 1)[1].lower()
                if ext in ['mp4', 'webm', 'ogg']:
                    save_path = os.path.join(app.config['VIDEO_FOLDER'], filename)
                    file.save(save_path)
                    value = f'static/video/{filename}'
                else:
                    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(save_path)
                    value = f'static/uploads/{filename}'
            else:
                value = request.form.get('image_path')
        else:
            value = request.form.get('image_path')
    else:  # file
        if 'file_upload' in request.files:
            file = request.files['file_upload']
            if file:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                value = filepath
            else:
                return "No file uploaded", 400
        else:
            value = request.form.get('file_path')
    
    launch_data = {
        "type": launch_type,
        "value": value,
        "trigger": trigger_type
    }
    
    # Redirect to appropriate trigger workflow
    if trigger_type == 'button':
        return redirect(url_for('launch'))
    elif trigger_type == 'hand':
        return redirect(url_for('hand_gesture'))
    elif trigger_type == 'voice':
        return redirect(url_for('voice_command'))
    elif trigger_type == 'timer':
        return redirect(url_for('timer_trigger'))
    elif trigger_type == 'qr':
        return redirect(url_for('qr_trigger'))
    elif trigger_type == 'remote':
        return redirect(url_for('remote_trigger'))
    # Proximity trigger removed
    
    return redirect(url_for('index'))

@app.route('/launch')
def launch():
    """Execute the launch based on stored data"""
    global launch_data

    if launch_data["type"] == "website":
        return render_template('display.html', 
                             message=f"Website launched: {launch_data['value']}",
                             type="website",
                             website_url=launch_data["value"])
    elif launch_data["type"] == "image":
        return render_template('display.html', 
                             image_path=launch_data["value"],
                             type="image")
    elif launch_data["type"] == "file":
        return render_template('display.html', 
                             file_path=launch_data["value"],
                             type="file")

    return "No launch data available", 400

@app.route('/hand')
def hand_gesture():
    """Hand gesture trigger page"""
    # Reset launch flag when starting new hand gesture session
    global launch_triggered, countdown_active, countdown_start
    launch_triggered = False
    countdown_active = False
    countdown_start = 0
    return render_template('hand.html')

@app.route('/set_launch_data', methods=['POST'])
def set_launch_data():
    """Set launch data for testing"""
    global launch_data
    data = request.get_json()
    launch_data.update(data)
    return jsonify({"success": True, "launch_data": launch_data})

@app.route('/debug_launch_data')
def debug_launch_data():
    """Debug route to check current launch data"""
    global launch_data
    return jsonify(launch_data)

@app.route('/video_feed')
def video_feed():
    """Video feed for hand gesture detection"""
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/hand_trigger', methods=['POST'])
def hand_trigger():
    """Triggered when hand gesture is detected (from frontend JS after countdown)"""
    global countdown_active, countdown_start, launch_triggered, hand_inside_at
    countdown_active = False
    countdown_start = 0
    hand_inside_at = 0
    launch_triggered = False  # Reset so /hand_status doesn't re-trigger
    return redirect(url_for('launch'))

@app.route('/hand_status')
def hand_status():
    """Return recent hand detection status"""
    global hand_detected_at, hand_inside_at, countdown_active, countdown_start, launch_triggered
    now = time.time()
    detected = (now - hand_detected_at) < 1.0
    inside = (now - hand_inside_at) < 1.0
    inside_duration = max(0.0, now - hand_inside_at) if inside else 0.0
    
    # Calculate countdown remaining
    countdown_remaining = 0.0
    if countdown_active:
        elapsed = now - countdown_start
        countdown_remaining = max(0.0, 3.0 - elapsed)
    
    # Auto-reset launch_triggered after reporting launch once
    launch = launch_triggered
    if launch_triggered:
        launch_triggered = False
    return jsonify({
        "detected": detected, 
        "inside": inside, 
        "inside_duration": round(inside_duration, 3),
        "countdown_active": countdown_active,
        "countdown_remaining": round(countdown_remaining, 1),
        "launch": launch
    })

@app.route('/set_hand_target', methods=['POST'])
def set_hand_target():
    """Set the target region as percentages of the frame size."""
    global hand_target_region
    try:
        data = request.get_json(force=True)
        x = float(data.get('x', hand_target_region['x']))
        y = float(data.get('y', hand_target_region['y']))
        w = float(data.get('w', hand_target_region['w']))
        h = float(data.get('h', hand_target_region['h']))
        # clamp and ensure fits within frame
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))
        w = max(0.05, min(1.0, w))
        h = max(0.05, min(1.0, h))
        if x + w > 1.0:
            x = 1.0 - w
        if y + h > 1.0:
            y = 1.0 - h
        hand_target_region = {"x": x, "y": y, "w": w, "h": h}
        return jsonify({"ok": True, "target": hand_target_region})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

@app.route('/voice')
def voice_command():
    """Voice command trigger page"""
    return render_template('voice.html')

@app.route('/listen', methods=['POST'])
def listen():
    """Listen for voice commands"""
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    
    try:
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source, timeout=5)
        
        text = recognizer.recognize_google(audio).lower()
        
        if any(keyword in text for keyword in ['launch', 'open', 'start', 'go']):
            return jsonify({"success": True, "command": text})
        else:
            return jsonify({"success": False, "message": "Command not recognized"})
    
    except sr.UnknownValueError:
        return jsonify({"success": False, "message": "Could not understand audio"})
    except sr.RequestError:
        return jsonify({"success": False, "message": "Could not request results"})

@app.route('/timer')
def timer_trigger():
    """Timer trigger page"""
    return render_template('timer.html')

@app.route('/schedule_launch', methods=['POST'])
def schedule_launch():
    """Schedule a launch for a specific time"""
    launch_time = request.form.get('launch_time')
    try:
        # Parse the datetime
        launch_datetime = datetime.strptime(launch_time, '%Y-%m-%dT%H:%M')
        current_time = datetime.now()
        if launch_datetime <= current_time:
            return render_template('timer.html', error="Please select a future time.")
        # Pass the launch_time as ISO string to the template for countdown
        return render_template('timer.html', launch_time=launch_datetime.isoformat())
    except ValueError:
        return render_template('timer.html', error="Invalid datetime format.")

@app.route('/qr')
def qr_trigger():
    """QR code trigger page"""
    # By default point QR to selected value when possible
    global launch_data
    qr_data = None
    if launch_data.get("type") == "website" and launch_data.get("value"):
        qr_data = launch_data["value"]
    elif launch_data.get("type") == "image" and launch_data.get("value"):
        # If image is in static/, construct absolute URL
        img_path = launch_data["value"].replace('\\', '/')
        if img_path.startswith('static/'):
            base = request.host_url.rstrip('/')
            qr_data = f"{base}/{img_path}"
    
    # Fallback to confirmation endpoint when we don't have a direct URL
    if not qr_data:
        qr_data = f"{request.host_url.rstrip('/')}/qr_confirm/{int(time.time())}"
    
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(qr_data)
    qr.make(fit=True)
    
    img = qr.make_image(fill_color="black", back_color="white")
    
    # Convert to base64 for display
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    
    return render_template('qr.html', qr_code=img_base64, qr_data=qr_data)

@app.route('/qr_confirm/<int:timestamp>')
def qr_confirm(timestamp):
    """Confirm QR code scan and trigger launch"""
    return redirect(url_for('launch'))

@app.route('/remote')
def remote_trigger():
    """Remote trigger page"""
    return render_template('remote.html')

@app.route('/remote_launch', methods=['POST'])
def remote_launch():
    """Remote launch endpoint"""
    return redirect(url_for('launch'))

@app.route('/hand_launch', methods=['POST'])
def hand_launch():
    """Triggered when hand countdown reaches 0"""
    global countdown_active, countdown_start, launch_triggered
    countdown_active = False
    countdown_start = 0
    launch_triggered = False
    return redirect(url_for('launch'))

@app.route('/check_launch')
def check_launch():
    """Check if launch should be triggered"""
    global launch_triggered
    if launch_triggered:
        launch_triggered = False
        return jsonify({"launch": True})
    return jsonify({"launch": False})

# Proximity routes removed

@app.route('/cleanup')
def cleanup():
    """Clean up camera resources"""
    global camera
    with camera_lock:
        if camera is not None:
            camera.release()
            camera = None
    return "Camera cleaned up"

if __name__ == '__main__':
    try:
        app.run(debug=True, host='0.0.0.0', port=5000)
    finally:
        # Clean up camera on exit
        with camera_lock:
            if camera is not None:
                camera.release()
