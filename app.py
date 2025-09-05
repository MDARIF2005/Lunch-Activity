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
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables to store launch data and camera
launch_data = {
    "type": None,
    "value": None,
    "trigger": None
}

camera = None
camera_lock = threading.Lock()
hand_detected_at = 0.0
hand_inside_at = 0.0
current_centroid = (0, 0)
# Target region as percentages of frame size
hand_target_region = {"x": 0.35, "y": 0.30, "w": 0.30, "h": 0.40}
countdown_start = 0
countdown_active = False
hand_sketch = None

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_hand_sketch():
    """Load the hand sketch image for overlay"""
    global hand_sketch
    try:
        sketch_path = os.path.join('static', 'images', 'hand_sketch.png')
        if os.path.exists(sketch_path):
            hand_sketch = cv2.imread(sketch_path, cv2.IMREAD_UNCHANGED)
            if hand_sketch is not None:
                print("Hand sketch loaded successfully")
            else:
                print("Failed to load hand sketch")
        else:
            print(f"Hand sketch not found at {sketch_path}")
    except Exception as e:
        print(f"Error loading hand sketch: {e}")

# Load hand sketch on startup
load_hand_sketch()

def gen_frames():
    """Generate video frames for camera feed with hand sketch overlay and countdown"""
    global camera, hand_detected_at, hand_inside_at, current_centroid, hand_target_region
    global countdown_start, countdown_active, hand_sketch
    
    with camera_lock:
        if camera is None:
            # Try different camera indices
            for i in range(3):  # Try cameras 0, 1, 2
                camera = cv2.VideoCapture(i)
                if camera.isOpened():
                    print(f"Camera {i} initialized successfully")
                    break
                else:
                    camera.release()
                    camera = None
            
            if camera is None:
                print("Error: Could not open any camera")
    
    while True:
        if camera is None:
            # Create a test frame if camera is not available
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Camera not available", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, "Please check camera connection", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            h_frame, w_frame = frame.shape[:2]
        else:
            success, frame = camera.read()
            if not success:
                # Create a test frame if camera read fails
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "Camera read failed", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, "Try refreshing the page", (50, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                h_frame, w_frame = frame.shape[:2]
            else:
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Use original camera frame (no black background)
                h_frame, w_frame = frame.shape[:2]
                
                # Enhanced hand detection using HSV mask (focused on hands only)
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                
                # More restrictive skin color range to avoid faces
                lower_skin1 = (0, 30, 80)  # Higher saturation and value thresholds
                upper_skin1 = (20, 255, 255)
                lower_skin2 = (170, 30, 80)
                upper_skin2 = (180, 255, 255)
                
                mask1 = cv2.inRange(hsv, lower_skin1, upper_skin1)
                mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
                skin_mask = cv2.bitwise_or(mask1, mask2)
                
                # More aggressive noise reduction
                skin_mask = cv2.GaussianBlur(skin_mask, (9, 9), 0)
                skin_mask = cv2.erode(skin_mask, None, iterations=2)
                skin_mask = cv2.dilate(skin_mask, None, iterations=3)
                
                # Find contours
                contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            detected = False
            inside_target = False
            
            if contours:
                # Filter contours by area and aspect ratio to focus on hands
                hand_contours = []
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if 5000 < area < 50000:  # Hand-sized area range
                        x, y, w, h = cv2.boundingRect(contour)
                        aspect_ratio = w / h
                        # Hands typically have aspect ratio between 0.5 and 1.5
                        if 0.5 < aspect_ratio < 1.5:
                            hand_contours.append(contour)
                
                if hand_contours:
                    # Use the largest hand-like contour
                    largest = max(hand_contours, key=cv2.contourArea)
                    detected = True
                    x, y, w, h = cv2.boundingRect(largest)
                    cx = x + w // 2
                    cy = y + h // 2
                    current_centroid = (cx, cy)
                    
                    # Check if hand is in center area (where hand sketch is)
                    center_x = w_frame // 2
                    center_y = h_frame // 2
                    center_tolerance = min(w_frame, h_frame) // 4  # Quarter of screen size
                    
                    if (abs(cx - center_x) < center_tolerance and 
                        abs(cy - center_y) < center_tolerance):
                        inside_target = True
                        hand_inside_at = time.time()
                        
                        # Start countdown if not already active
                        if not countdown_active:
                            countdown_start = time.time()
                            countdown_active = True
                        
                        # Calculate countdown progress
                        elapsed = time.time() - countdown_start
                        remaining = max(0, 3.0 - elapsed)
                        
                        if remaining > 0:
                            # Draw countdown
                            countdown_text = f"Countdown: {remaining:.1f}s"
                            cv2.putText(frame, countdown_text, (w_frame//2 - 100, h_frame//2), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
                            
                            # Draw progress bar
                            bar_width = 300
                            bar_height = 20
                            bar_x = w_frame//2 - bar_width//2
                            bar_y = h_frame//2 + 50
                            
                            # Background bar
                            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
                            # Progress bar
                            progress = (3.0 - remaining) / 3.0
                            progress_width = int(bar_width * progress)
                            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), (0, 255, 0), -1)
                        else:
                            # Countdown finished - trigger launch
                            cv2.putText(frame, "LAUNCHING!", (w_frame//2 - 80, h_frame//2), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                    else:
                        # Reset countdown if hand moves out of center area
                        countdown_active = False
                        countdown_start = 0
                else:
                    # Reset countdown if no hand detected
                    countdown_active = False
                    countdown_start = 0
            
            # Always show the transparent hand sketch at center when hand is detected
            if detected and hand_sketch is not None:
                try:
                    # Resize hand sketch to fit nicely in the center
                    sketch_h, sketch_w = hand_sketch.shape[:2]
                    scale = min(w_frame * 0.4 / sketch_w, h_frame * 0.4 / sketch_h)
                    new_w = int(sketch_w * scale)
                    new_h = int(sketch_h * scale)
                    
                    # Position at center of frame
                    x_offset = (w_frame - new_w) // 2
                    y_offset = (h_frame - new_h) // 2
                    
                    sketch_resized = cv2.resize(hand_sketch, (new_w, new_h))
                    
                    if hand_sketch.shape[2] == 4:  # Has alpha channel
                        # Extract alpha channel for transparency
                        alpha = sketch_resized[:, :, 3] / 255.0
                        
                        # Blend with original camera frame (transparent overlay)
                        for c in range(3):
                            frame[y_offset:y_offset+new_h, x_offset:x_offset+new_w, c] = \
                                alpha * sketch_resized[:, :, c] + (1 - alpha) * frame[y_offset:y_offset+new_h, x_offset:x_offset+new_w, c]
                    else:
                        # If no alpha channel, create one by making white pixels transparent
                        # Convert to grayscale to find white background
                        gray = cv2.cvtColor(sketch_resized, cv2.COLOR_BGR2GRAY)
                        # Create alpha channel where white pixels (background) are transparent
                        alpha = np.where(gray > 240, 0, 1).astype(np.float32)  # White pixels become transparent
                        
                        # Apply transparency
                        for c in range(3):
                            frame[y_offset:y_offset+new_h, x_offset:x_offset+new_w, c] = \
                                alpha * sketch_resized[:, :, c] + (1 - alpha) * frame[y_offset:y_offset+new_h, x_offset:x_offset+new_w, c]
                except Exception as e:
                    print(f"Error overlaying hand sketch: {e}")
            
            # Show instruction when no hand is detected
            if not detected:
                cv2.putText(frame, "Show your hand to trigger", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            if detected:
                hand_detected_at = time.time()
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Proximity feature removed

@app.route('/')
def index():
    """Main selection page"""
    return render_template('index.html')

@app.route('/button')
def button_trigger():
    """Simple button page to manually launch the configured action"""
    return render_template('button.html')

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
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                value = filepath
            else:
                return "Invalid file type", 400
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
        return redirect(url_for('button_trigger'))
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
        webbrowser.open(launch_data["value"])
        return render_template('display.html', 
                             message=f"Website launched: {launch_data['value']}",
                             type="website")
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
    return render_template('hand.html')

@app.route('/video_feed')
def video_feed():
    """Video feed for hand gesture detection"""
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/hand_trigger', methods=['POST'])
def hand_trigger():
    """Triggered when hand gesture is detected"""
    return redirect(url_for('launch'))

@app.route('/hand_status')
def hand_status():
    """Return recent hand detection status"""
    global hand_detected_at, hand_inside_at, countdown_start, countdown_active
    now = time.time()
    detected = (now - hand_detected_at) < 1.0
    inside = (now - hand_inside_at) < 1.0
    inside_duration = max(0.0, now - hand_inside_at) if inside else 0.0
    
    # Calculate countdown status
    countdown_remaining = 0
    if countdown_active and inside:
        elapsed = now - countdown_start
        countdown_remaining = max(0, 3.0 - elapsed)
        if countdown_remaining <= 0:
            # Countdown finished, trigger launch
            return jsonify({"detected": detected, "inside": inside, "inside_duration": round(inside_duration, 3), 
                          "countdown_remaining": 0, "launch": True})
    
    return jsonify({"detected": detected, "inside": inside, "inside_duration": round(inside_duration, 3), 
                   "countdown_remaining": round(countdown_remaining, 1), "launch": False})

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
            return "Launch time must be in the future", 400
        
        # Calculate delay in seconds
        delay = (launch_datetime - current_time).total_seconds()
        
        # Schedule the launch
        timer = threading.Timer(delay, lambda: webbrowser.open(launch_data["value"]) if launch_data["type"] == "website" else None)
        timer.start()
        
        return render_template('display.html', 
                             message=f"Launch scheduled for {launch_time}",
                             type="timer")
    
    except ValueError:
        return "Invalid datetime format", 400

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
