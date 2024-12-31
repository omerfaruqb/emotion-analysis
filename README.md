# Real-time Emotion Analysis

A real-time facial emotion detection system using PyTorch and OpenCV. The system can detect and classify seven different emotions: Surprise, Fear, Disgust, Happy, Sad, Angry, and Neutral.

## Features

- Real-time emotion detection from webcam feed
- Support for 7 different emotions
- Configurable parameters via YAML config
- Logging system for debugging and monitoring
- Pre-trained ResNet18 model converted to ONNX for efficient inference

## Project Structure

```
emotion-analysis/
├── config/              # Configuration files
├── data/               # Dataset and processed data
├── models/             # Trained models and weights
├── notebooks/          # Jupyter notebooks for experiments
├── src/               # Source code
│   ├── models/        # Model implementations
│   ├── utils/         # Utility functions
│   └── data/          # Data processing scripts
└── tests/             # Unit tests
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/emotion-analysis.git
cd emotion-analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main application:
```bash
python main.py
```

Press 'q' to quit the application.

## Configuration

The system can be configured via `config/config.yaml`. Key parameters include:
- Model architecture and weights
- Input image size
- Preprocessing parameters
- Emotion labels and colors
- Inference settings

## Model Training

The emotion detection model is based on ResNet18 architecture, trained on the [RAF-DB dataset](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset).

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
