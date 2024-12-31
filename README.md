# emotion-analysis
A real-time emotion analysis according to faces
The data is taken from a [Kaggle Dataset](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset) dataset.

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
