# Webcam-Based Detection

1. Color Detection
   
Detects a specific color (default: yellow) from the webcam feed.

Draws a bounding box around detected regions.

Useful for object tracking based on color.
```bash
python color_detection.py
```

2. Face Anonymizer
   
Uses MediaPipe Face Detection to identify faces.

Applies a blur effect to anonymize detected faces in real-time.
```bash
python face_anonymizer.py
```

3. Text Detection
   
Supports two OCR backends:

Tesseract

EasyOCR

Reads and prints text from the webcam feed.

```bash
python text_detection.py
Options: 'Pytesseract' or 'EasyOCR'
```

4. Emotion Detection

Two-class classifier (happy vs. sad) trained on facial landmarks.

Workflow:

Resize & clean dataset images.

Extract facial landmarks using MediaPipe.

Train a RandomForestClassifier.

Predict emotions in real-time from webcam input.

Real-time Detection:

```bash
python emotion_detection.py
```

