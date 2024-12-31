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
    
    # Initialize session state for frame storage
    if 'preview_frame' not in st.session_state:
        st.session_state['preview_frame'] = None
    if 'captured_frame' not in st.session_state:
        st.session_state['captured_frame'] = None
    
    # Initialize camera placeholders
    preview_placeholder = st.empty()
    captured_placeholder = st.empty()
    
    # Check webcam availability first
    try:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow on Windows
        if not cap.isOpened():
            st.error("❌ Could not access webcam. Please ensure:")
            st.write("1. Your webcam is properly connected")
            st.write("2. No other application is using the webcam")
            st.write("3. You have granted browser permission to access the webcam")
            st.write("4. Your antivirus or security software is not blocking webcam access")
        else:
            cap.release()  # Release immediately after checking
            
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                preview = st.button("Preview Camera")
            with col2:
                take_photo = st.button("Take Photo")
            with col3:
                if st.button("Clear"):
                    st.session_state['preview_frame'] = None
                    st.session_state['captured_frame'] = None
                    preview_placeholder.empty()
                    captured_placeholder.empty()
            
            # Show preview of camera
            if preview:
                try:
                    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow on Windows
                    ret, frame = cap.read()
                    if ret:
                        # Store the original frame
                        st.session_state['preview_frame'] = frame.copy()
                        # Convert BGR to RGB for display
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        # Display preview
                        preview_placeholder.image(rgb_frame, caption="Camera Preview")
                    else:
                        st.error("❌ Failed to get preview from camera. Please try again.")
                    cap.release()
                except Exception as e:
                    st.error(f"❌ Error accessing camera: {str(e)}")
                    st.info("💡 Try refreshing the page or restarting your browser")
            
            # Take and process photo
            if take_photo:
                if st.session_state['preview_frame'] is not None:
                    try:
                        # Get the stored frame
                        frame = st.session_state['preview_frame'].copy()
                        # Process frame for emotion detection
                        processed_frame = detector.process_image(frame)
                        # Store the processed frame
                        st.session_state['captured_frame'] = processed_frame.copy()
                        # Convert BGR to RGB for display
                        rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                        # Display captured and processed photo
                        captured_placeholder.image(rgb_frame, caption="Captured Photo with Emotion Detection")
                    except Exception as e:
                        st.error(f"❌ Error processing photo: {str(e)}")
                else:
                    st.warning("Please preview the camera first before taking a photo")
    except Exception as e:
        st.error(f"❌ Error initializing camera: {str(e)}")
        st.info("💡 If you're using Windows, make sure your privacy settings allow camera access")

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