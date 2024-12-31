import streamlit as st
import cv2
import numpy as np
from src.models.emotion_detector import EmotionDetector
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Emotion Detection App",
    page_icon="😊",
    layout="wide"
)

# Title
st.title("Real-time Emotion Detection")
st.write("Upload an image or use your webcam to detect emotions!")

# Initialize emotion detector
@st.cache_resource
def load_detector():
    return EmotionDetector()

detector = load_detector()

# Sidebar
st.sidebar.title("Options")
detection_mode = st.sidebar.radio("Choose Detection Mode", ["Upload Image", "Webcam"])

if detection_mode == "Upload Image":
    # Image upload
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Convert uploaded file to image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Process image
        processed_image = detector.process_image(image)
        
        # Display results
        st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), caption="Processed Image")

else:
    # Webcam
    st.write("Webcam Mode")
    camera_image = st.camera_input("Take a photo")
    
    if camera_image is not None:
        # Convert the image from bytes to numpy array
        bytes_data = camera_image.getvalue()
        image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        
        # Process image
        processed_image = detector.process_image(image)
        
        # Display processed image
        st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), caption="Processed Image")

# Add information about the project
st.markdown("""
## About
This emotion detection system can recognize 7 different emotions:
- 😮 Surprised
- 😨 Fear
- 🤢 Disgust
- 😊 Happy
- 😢 Sad
- 😠 Angry
- 😐 Neutral

## How it works
The system uses a deep learning model (ResNet18) trained on the RAF-DB dataset to detect emotions in real-time.
""")

# GitHub link
st.sidebar.markdown("---")
st.sidebar.markdown("Created by [Your Name](https://github.com/yourusername)") 