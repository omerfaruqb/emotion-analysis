import cv2
import numpy as np
from src.models.emotion_detector import EmotionDetector
from src.utils.logger import logger

def main():
    """Main function for real-time emotion detection."""
    try:
        # Initialize the emotion detector
        detector = EmotionDetector()
        logger.info("Starting webcam capture...")
        
        # Initialize video capture
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Could not open the camera")
        
        while True:
            # Read frame from camera
            success, frame = cap.read()
            if not success:
                logger.error("Failed to read frame from camera")
                break
            
            # Process frame
            try:
                processed_frame = detector.process_image(frame)
                cv2.imshow("Emotion Detection", processed_frame)
                
            except Exception as e:
                logger.error(f"Error processing frame: {str(e)}")
                cv2.imshow("Emotion Detection", frame)
            
            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("User requested exit")
                break
        
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
    
    finally:
        # Clean up
        logger.info("Cleaning up resources...")
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()