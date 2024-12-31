import streamlit as st
import cv2
import numpy as np
from src.models.emotion_detector import EmotionDetector
from pathlib import Path
import tempfile

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
    st.warning("Note: The webcam feature may not work when deployed on Streamlit Cloud due to security restrictions. For the best experience, please run this app locally using `streamlit run app.py`")
    run = st.checkbox("Start Webcam")
    FRAME_WINDOW = st.image([])
    
    if run:
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Could not access webcam. Please check your webcam permissions and try again.")
                st.info("If you're using this app on Streamlit Cloud, the webcam feature is not supported. Please run locally instead.")
            else:
                while run:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to access webcam")
                        break
                        
                    # Process frame
                    processed_frame = detector.process_image(frame)
                    
                    # Convert BGR to RGB
                    rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    
                    # Display frame
                    FRAME_WINDOW.image(rgb_frame)
                
                cap.release()
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.info("If you're using this app on Streamlit Cloud, try running it locally instead.")

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
st.sidebar.markdown("Created by [Faruq Bayram](https://github.com/omerfaruqb)") 