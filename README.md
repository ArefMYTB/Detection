# **Detection**

A computer vision repository with modular components for detecting and analyzing various visual elements through webcam or image input. This project uses OpenCV, MediaPipe, EasyOCR, Tesseract, and machine learning models to provide real-time or static image-based detection for:

Colors
Faces (with anonymization)
Text
Emotions
Weather conditions

# Requirements
Install the following libraries before running the modules:
```bash
pip install opencv-python numpy pillow mediapipe easyocr pytesseract scikit-learn img2vec-pytorch

Note:
Tesseract must be installed separately.
img2vec-pytorch uses pretrained models and may require PyTorch and torchvision.


# Datasets
Place your datasets under:
Dataset/emotion/ (subfolders: e.g. happy/, sad/)
Dataset/weather/ (images prefixed with category name)
