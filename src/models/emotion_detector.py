import cv2
import numpy as np
import onnx
import onnxruntime as rt
import yaml
from pathlib import Path
from ..utils.logger import logger

class EmotionDetector:
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the emotion detector with configuration.
        
        Args:
            config_path (str): Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.model = self._load_model()
        self.emotion_dict = {
            int(k): tuple(v) for k, v in self.config['emotion_labels'].items()
        }
        self.class_weights = np.array(self.config['data']['class_weights'])
        
        # Initialize face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        logger.info("Emotion detector initialized successfully")
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            raise
    
    def _load_model(self):
        """Load and initialize ONNX model."""
        try:
            model_path = self.config['model']['weights_path']
            model = onnx.load(model_path)
            onnx.checker.check_model(model)
            self.session = rt.InferenceSession(model_path)
            self.input_name = self.session.get_inputs()[0].name
            logger.info(f"Model loaded successfully from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def preprocess_image(self, face: np.ndarray) -> np.ndarray:
        """Preprocess face image for model input.
        
        Args:
            face (np.ndarray): Input face image
            
        Returns:
            np.ndarray: Preprocessed image
        """
        try:
            input_size = tuple(self.config['model']['input_size'])
            mean = np.array(self.config['data']['mean'])
            std = np.array(self.config['data']['std'])
            
            resized = cv2.resize(face, input_size)
            transformed = ((resized / 255.0) - mean) / std
            transformed = transformed.astype(np.float32)
            
            reshaped = np.transpose(transformed, (2, 0, 1))
            reshaped = np.expand_dims(reshaped, axis=0)
            
            return reshaped
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            raise
    
    def predict_emotion(self, face: np.ndarray) -> tuple:
        """Predict emotion from face image.
        
        Args:
            face (np.ndarray): Input face image
            
        Returns:
            tuple: (emotion_label, color, probability)
        """
        try:
            preprocessed = self.preprocess_image(face)
            outputs = self.session.run(None, {self.input_name: preprocessed})
            
            # Apply softmax and weight adjustment
            probs = np.exp(outputs[0]) / np.sum(np.exp(outputs[0]))
            weighted_probs = probs * self.class_weights
            weighted_probs /= np.sum(weighted_probs)
            
            emotion_idx = np.argmax(weighted_probs)
            emotion_label, color = self.emotion_dict[emotion_idx]
            probability = float(np.max(weighted_probs) * 100)
            
            logger.debug(f"Predicted emotion: {emotion_label} with {probability:.2f}% confidence")
            return emotion_label, color, probability
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            raise
    
    def detect_faces(self, image: np.ndarray) -> list:
        """Detect faces in the image.
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            list: List of face rectangles (x, y, w, h)
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=self.config['inference']['face_detection_scale_factor'],
                minNeighbors=self.config['inference']['face_detection_min_neighbors']
            )
            return faces
        except Exception as e:
            logger.error(f"Error in face detection: {str(e)}")
            return []
    
    def process_image(self, image: np.ndarray) -> np.ndarray:
        """Process image and draw emotion predictions.
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            np.ndarray: Processed image with predictions
        """
        try:
            faces = self.detect_faces(image)
            
            for (x, y, w, h) in faces:
                # Extract face ROI
                face_roi = image[y:y+h, x:x+w]
                
                # Predict emotion
                emotion, color, prob = self.predict_emotion(face_roi)
                
                # Draw rectangle around face
                cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
                
                # Draw emotion label
                label = f"{emotion}: {prob:.1f}%"
                cv2.putText(
                    image, label,
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, color, 2
                )
            
            return image
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return image

if __name__ == "__main__":
    detector = EmotionDetector()
    test_image = cv2.imread("data/test/happy.jpg")
    if test_image is not None:
        emotion, color, prob = detector.predict_emotion(test_image)
        print(f"Emotion: {emotion} - Probability: {prob:.2f}%") 