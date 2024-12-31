import streamlit as st
import cv2
import numpy as np
from src.models.emotion_detector import EmotionDetector
from pathlib import Path
import platform
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
detection_mode = st.sidebar.radio("Choose Detection Mode", ["Upload Image", "Take Photo", "Webcam"])

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

elif detection_mode == "Take Photo":
    st.write("Take Photo Mode")
    
    # Check if running on cloud
    is_cloud = platform.system().lower() == "linux" and "streamlit.io" in str(st.runtime.get_instance())
    
    # Initialize camera placeholders
    preview_placeholder = st.empty()
    captured_placeholder = st.empty()
    
    col1, col2 = st.columns(2)
    
    if is_cloud:
        st.warning("⚠️ Direct webcam access is not available in cloud environment.")
        st.info("Please provide a stream URL (e.g., IP camera or video stream)")
        stream_url = st.text_input("Stream URL")
        
        with col1:
            preview = st.button("Preview Stream")
        with col2:
            take_photo = st.button("Capture Photo")
        
        if preview and stream_url:
            try:
                import requests
                from PIL import Image
                from io import BytesIO
                
                response = requests.get(stream_url)
                if response.status_code == 200:
                    # Convert stream data to image
                    image = Image.open(BytesIO(response.content))
                    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    # Convert BGR to RGB for preview
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Display preview
                    preview_placeholder.image(rgb_frame, caption="Stream Preview")
                    st.session_state['preview_frame'] = frame
                else:
                    st.error("❌ Could not access the stream URL")
            except Exception as e:
                st.error(f"❌ Error accessing stream: {str(e)}")
        
        if take_photo and stream_url:
            if 'preview_frame' in st.session_state:
                frame = st.session_state['preview_frame']
                # Process frame
                processed_frame = detector.process_image(frame)
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                # Display captured photo
                captured_placeholder.image(rgb_frame, caption="Captured Photo with Emotion Detection")
            else:
                st.warning("Please preview the stream first before capturing")
    else:
        with col1:
            preview = st.button("Preview Camera")
        with col2:
            take_photo = st.button("Take Photo")
        
        if preview:
            try:
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    st.error("❌ Could not access webcam. Please check your permissions.")
                else:
                    ret, frame = cap.read()
                    if ret:
                        # Convert BGR to RGB for preview
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        # Display preview
                        preview_placeholder.image(rgb_frame, caption="Camera Preview")
                        st.session_state['preview_frame'] = frame
                    else:
                        st.error("❌ Failed to get preview from camera")
                    cap.release()
            except Exception as e:
                st.error(f"❌ Error accessing camera: {str(e)}")
        
        if take_photo:
            if 'preview_frame' in st.session_state:
                frame = st.session_state['preview_frame']
                # Process frame
                processed_frame = detector.process_image(frame)
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                # Display captured photo
                captured_placeholder.image(rgb_frame, caption="Captured Photo with Emotion Detection")
            else:
                st.warning("Please preview the camera first before taking a photo")

else:
    # Webcam
    st.write("Webcam Mode")
    
    # Check if running on cloud
    is_cloud = platform.system().lower() == "linux" and "streamlit.io" in str(st.runtime.get_instance())
    
    if is_cloud:
        st.warning("⚠️ Webcam access might be limited when running on cloud platforms. For best experience, please run the app locally or use the 'Upload Image' option.")
    
    run = st.checkbox("Start Webcam")
    FRAME_WINDOW = st.image([])
    
    if run:
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("❌ Could not access webcam. Please check your permissions or try using 'Upload Image' mode instead.")
                run = False
            else:
                while run:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("❌ Failed to get frame from webcam")
                        break
                    
                    # Process frame
                    try:
                        processed_frame = detector.process_image(frame)
                        # Convert BGR to RGB
                        rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                        # Display frame
                        FRAME_WINDOW.image(rgb_frame)
                    except Exception as e:
                        st.error(f"❌ Error processing frame: {str(e)}")
                        break
                
                cap.release()
        except Exception as e:
            st.error(f"❌ Error initializing webcam: {str(e)}")
            st.info("💡 If you're running this app on a cloud platform, try using the 'Upload Image' mode instead.")

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