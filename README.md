# Activity Launch App

A comprehensive Flask web application that allows users to launch websites, images, or files using various trigger mechanisms including hand gestures, voice commands, QR codes, timers, and proximity detection.

## Features

### Launch Types
- **Website**: Open any URL in the default browser
- **Image/Logo**: Display images with upload support
- **File**: Handle file uploads and access

### Trigger Mechanisms
- **Button Click**: Direct launch trigger
- **Hand Gesture**: OpenCV-based gesture detection
- **Voice Command**: Speech recognition using Google API
- **Timer**: Scheduled launches at specific times
- **QR Code**: Generate and scan QR codes for remote triggering
- **Remote**: Mobile-friendly remote launch interface
- **Proximity**: Face detection using OpenCV

## Installation

### Prerequisites
- Python 3.7 or higher
- Webcam (for gesture and proximity detection)
- Microphone (for voice commands)

### Setup

1. **Clone or download the project files**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   python app.py
   ```

4. **Access the application**:
   Open your browser and navigate to `http://localhost:5000`

## Usage

### Basic Workflow

1. **Configure Launch**: Select what you want to launch (website, image, or file)
2. **Choose Trigger**: Select how you want to trigger the launch
3. **Execute**: Follow the specific workflow for your chosen trigger

### Trigger-Specific Instructions

#### Button Trigger
- Simply click the "Launch Now" button to execute immediately

#### Hand Gesture Trigger
- Allow camera access when prompted
- Press 'Q' key or use the trigger button to launch
- Ensure good lighting for optimal detection

#### Voice Command Trigger
- Click "Start Listening" and speak clearly
- Use commands like "launch", "open", "start", or "go"
- Wait for confirmation before the launch executes

#### Timer Trigger
- Select a future date and time
- Keep the browser tab open until the scheduled time
- The launch will execute automatically

#### QR Code Trigger
- A QR code will be generated automatically
- Scan with your phone's camera or QR scanner app
- The launch will trigger when the QR code is accessed

#### Remote Trigger
- Share the remote URL with others or access from mobile
- Use the large "Launch Now" button for easy mobile access
- Generate QR codes for quick mobile access

#### Proximity Trigger
- Allow camera access for face detection
- The system will automatically detect faces
- Launch triggers when someone approaches the camera

## File Structure

```
Activity lunch/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── templates/            # HTML templates
│   ├── base.html         # Base template with common layout
│   ├── index.html        # Main configuration page
│   ├── display.html      # Launch result display
│   ├── hand.html         # Hand gesture detection page
│   ├── voice.html        # Voice command page
│   ├── timer.html        # Timer configuration page
│   ├── qr.html           # QR code display page
│   ├── remote.html       # Remote launch page
│   └── proximity.html    # Proximity detection page
└── static/               # Static files
    ├── css/
    │   └── style.css     # Custom styles
    └── uploads/          # Uploaded files (created automatically)
```

## Technical Details

### Dependencies
- **Flask**: Web framework
- **OpenCV**: Computer vision for camera and face detection
- **SpeechRecognition**: Voice command processing
- **QRCode**: QR code generation
- **Pillow**: Image processing
- **Werkzeug**: File upload handling

### Browser Compatibility
- Chrome/Chromium (recommended)
- Firefox
- Safari
- Edge

### Mobile Support
- Responsive design for all screen sizes
- Touch-friendly interface
- Mobile-optimized remote trigger

## Troubleshooting

### Common Issues

1. **Camera not working**:
   - Ensure camera permissions are granted
   - Check if another application is using the camera
   - Try refreshing the page

2. **Voice recognition not working**:
   - Check microphone permissions
   - Ensure stable internet connection (uses Google API)
   - Speak clearly and avoid background noise

3. **File uploads failing**:
   - Check file size (max 16MB)
   - Ensure file type is supported
   - Check available disk space

4. **Timer not executing**:
   - Keep the browser tab open
   - Ensure system clock is accurate
   - Check for browser sleep mode

### Performance Tips

- Close unnecessary browser tabs for better performance
- Ensure good lighting for camera-based triggers
- Use a stable internet connection for voice recognition
- Keep the application updated for best compatibility

## Security Notes

- The application runs on localhost by default
- File uploads are restricted to specific types
- Camera and microphone access require user permission
- No data is stored permanently (except uploaded files)

## Development

### Adding New Triggers

1. Add the trigger option to `index.html`
2. Create a new route in `app.py`
3. Add the trigger logic
4. Create a corresponding template
5. Update the selection logic in `store_selection()`

### Customizing Detection

- Modify OpenCV parameters in `gen_frames()` and `gen_frames_proximity()`
- Adjust speech recognition settings in the `listen()` route
- Customize QR code generation in `qr_trigger()`

## License

This project is open source and available under the MIT License.

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review the console for error messages
3. Ensure all dependencies are properly installed
4. Verify system requirements are met

## Live Demo

[Launch Activity App on Render](https://lunch-activity.onrender.com)
