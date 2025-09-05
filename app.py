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
hand_detected_at = 0.0

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Global variables for hand detection and countdown
countdown_active = False
countdown_start = 0.0
hand_sketch = None
launch_triggered = False

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
            print("Hand sketch file not found")
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
            
            # Add demo target area
            cv2.rectangle(frame, (220, 280), (420, 380), (0, 255, 0), 3)
            cv2.putText(frame, "TARGET AREA", (250, 270), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
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
                        frame[target_y:target_y+new_h, target_x:target_x+new_w] = sketch_resized
                    
                    # Add green border around target
                    cv2.rectangle(frame, (target_x-5, target_y-5), 
                                (target_x+new_w+5, target_y+new_h+5), (0, 255, 0), 3)
                    cv2.putText(frame, "TARGET AREA", (target_x, target_y-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
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
            
            # Basic hand-like skin detection using HSV mask
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower_skin1 = (0, 20, 70)
            upper_skin1 = (20, 255, 255)
            lower_skin2 = (170, 20, 70)
            upper_skin2 = (180, 255, 255)
            mask1 = cv2.inRange(hsv, lower_skin1, upper_skin1)
            mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
            skin_mask = cv2.bitwise_or(mask1, mask2)
            
            # Smooth mask and reduce noise
            skin_mask = cv2.GaussianBlur(skin_mask, (7, 7), 0)
            skin_mask = cv2.erode(skin_mask, None, iterations=1)
            skin_mask = cv2.dilate(skin_mask, None, iterations=2)
            
            # Find contours
            contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            detected = False
            inside_target = False
            
            if contours:
                largest = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest)
                if area > 8000:  # heuristic threshold
                    detected = True
                    x, y, w, h = cv2.boundingRect(largest)
                    cx = x + w // 2
                    cy = y + h // 2
                    current_centroid = (cx, cy)
                    
                    # Draw hand detection rectangle
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)
                    
                    # Check if hand is in target area (hand sketch area)
                    if hand_sketch is not None:
                        try:
                            # Calculate hand sketch position and size
                            sketch_h, sketch_w = hand_sketch.shape[:2]
                            scale = min(w_frame * 0.4 / sketch_w, h_frame * 0.4 / sketch_h)
                            new_w = int(sketch_w * scale)
                            new_h = int(sketch_h * scale)
                            
                            target_x = (w_frame - new_w) // 2
                            target_y = (h_frame - new_h) // 2
                            
                            # Check if hand center is within the hand sketch target area
                            if (target_x <= cx <= target_x + new_w and 
                                target_y <= cy <= target_y + new_h):
                                inside_target = True
                                hand_inside_at = time.time()
                                
                                # Start countdown if hand is in target area
                                if not countdown_active:
                                    countdown_active = True
                                    countdown_start = time.time()
                            else:
                                inside_target = False
                                countdown_active = False
                                countdown_start = 0
                        except Exception as e:
                            print(f"Error in hand sketch detection: {e}")
                    else:
                        # Fallback to center area if no hand sketch
                        center_x = w_frame // 2
                        center_y = h_frame // 2
                        center_tolerance = min(w_frame, h_frame) // 4
                        
                        if (abs(cx - center_x) < center_tolerance and 
                            abs(cy - center_y) < center_tolerance):
                            inside_target = True
                            hand_inside_at = time.time()
                            
                            if not countdown_active:
                                countdown_active = True
                                countdown_start = time.time()
                        else:
                            inside_target = False
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
                    # Draw countdown with big numbers
                    countdown_number = int(remaining) + 1
                    if countdown_number > 3:
                        countdown_number = 3
                    
                    # Draw big countdown number
                    cv2.putText(frame, str(countdown_number), (w_frame//2 - 50, h_frame//2), 
                              cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 6)
                    
                    # Draw progress text
                    cv2.putText(frame, "Hold steady...", (w_frame//2 - 80, h_frame//2 + 60), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                    # Draw progress bar
                    bar_width = 300
                    bar_height = 20
                    bar_x = w_frame//2 - bar_width//2
                    bar_y = h_frame//2 + 100
                    
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
                    
                    # Reset countdown to prevent multiple launches
                    countdown_active = False
                    countdown_start = 0
                    
                    # Set a flag to trigger launch on next frame
                    global launch_triggered
                    launch_triggered = True
            
            # Always show the hand sketch as a target overlay
            if hand_sketch is not None:
                try:
                    # Resize hand sketch to fit nicely in the center
                    sketch_h, sketch_w = hand_sketch.shape[:2]
                    scale = min(w_frame * 0.4 / sketch_w, h_frame * 0.4 / sketch_h)
                    new_w = int(sketch_w * scale)
                    new_h = int(sketch_h * scale)
                    
                    target_x = (w_frame - new_w) // 2
                    target_y = (h_frame - new_h) // 2
                    
                    # Create a simple white hand outline
                    # Draw a white hand shape using basic shapes
                    center_x = target_x + new_w // 2
                    center_y = target_y + new_h // 2
                    
                    # Draw hand outline as white lines
                    cv2.rectangle(frame, (target_x, target_y), (target_x + new_w, target_y + new_h), (255, 255, 255), 3)
                    
                    # Draw fingers
                    finger_width = new_w // 8
                    finger_height = new_h // 3
                    
                    # Thumb
                    cv2.rectangle(frame, (target_x, target_y + finger_height), 
                                (target_x + finger_width, target_y + finger_height * 2), (255, 255, 255), 3)
                    
                    # Index finger
                    cv2.rectangle(frame, (target_x + finger_width, target_y), 
                                (target_x + finger_width * 2, target_y + finger_height), (255, 255, 255), 3)
                    
                    # Middle finger
                    cv2.rectangle(frame, (target_x + finger_width * 2, target_y), 
                                (target_x + finger_width * 3, target_y + finger_height), (255, 255, 255), 3)
                    
                    # Ring finger
                    cv2.rectangle(frame, (target_x + finger_width * 3, target_y), 
                                (target_x + finger_width * 4, target_y + finger_height), (255, 255, 255), 3)
                    
                    # Pinky
                    cv2.rectangle(frame, (target_x + finger_width * 4, target_y), 
                                (target_x + finger_width * 5, target_y + finger_height), (255, 255, 255), 3)
                    
                    # Palm
                    cv2.rectangle(frame, (target_x + finger_width, target_y + finger_height), 
                                (target_x + finger_width * 4, target_y + new_h), (255, 255, 255), 3)
                    
                    # Add bright green border around the target area
                    cv2.rectangle(frame, (target_x-5, target_y-5), 
                                (target_x+new_w+5, target_y+new_h+5), (0, 255, 0), 3)
                    cv2.putText(frame, "TARGET AREA", (target_x, target_y-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                except Exception as e:
                    print(f"Error overlaying hand sketch: {e}")
            
            # Display status text
            if detected:
                if inside_target:
                    cv2.putText(frame, "Position your hand on the white hand sketch", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "Move your hand to the hand sketch target", (10, 30), 
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
    global hand_detected_at, hand_inside_at
    now = time.time()
    detected = (now - hand_detected_at) < 1.0
    inside = (now - hand_inside_at) < 1.0
    inside_duration = max(0.0, now - hand_inside_at) if inside else 0.0
    return jsonify({"detected": detected, "inside": inside, "inside_duration": round(inside_duration, 3)})

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
