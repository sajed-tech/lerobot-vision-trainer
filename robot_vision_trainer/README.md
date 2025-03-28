# Robot Vision Trainer

A unified platform that enables end-to-end imitation learning for robotics. This application streamlines the process of:

1. Visual data input
2. Object and action detection
3. Dataset selection and policy training
4. Policy evaluation
5. Benchmark analysis against baseline methods

## Features

- **Image Upload**: Easily upload images through a web interface
- **Intelligent Object Detection**: Uses modern vision language models to identify objects and relevant actions
- **Automatic Dataset Selection**: Intelligently maps detected objects and actions to appropriate training datasets
- **Policy Training**: Trains imitation learning policies using the LERobot framework
- **Policy Evaluation**: Test trained policies in simulated environments
- **Benchmark Analysis**: Compare your trained policy against baseline algorithms

## Setup

### Prerequisites

- Python 3.9+
- API keys for OpenRouter or OpenAI (for object detection)
- Hugging Face token for accessing LERobot datasets

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/sajed-tech/robot-vision-trainer.git
   cd robot-vision-trainer
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Configure environment variables:
   - Create a `.env` file based on the provided example
   - Add your API keys for OpenRouter/OpenAI and Hugging Face

### Configuration

Edit the `.env` file with your specific settings:

```
# API keys
OPENAI_API_KEY=your_openai_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here
HF_TOKEN=your_huggingface_token_here

# Model configuration
VISION_MODEL=gemma-3-27b-vision

# Other settings
USE_WANDB=false
WANDB_API_KEY=your_wandb_key_here

# Flask configuration
PORT=5000
HOST=0.0.0.0
```

## Usage

1. Start the application:
   ```
   python run.py --port 5000
   ```

2. Open your browser and navigate to:
   ```
   http://localhost:5000
   ```

3. Follow the step-by-step process:
   - **Step 1**: Upload an image
   - **Step 2**: Detect objects and define actions
   - **Step 3**: Train a policy using selected datasets
   - **Step 4**: Evaluate and benchmark your trained policy

## Technical Details

### Dataset Selection Logic

The system intelligently selects appropriate datasets based on detected objects and actions:

- Maps common objects (cups, bowls, etc.) to relevant datasets
- Maps actions (pick up, pour, etc.) to datasets containing those actions
- Assigns priority scores to datasets based on relevance
- Selects the highest-scoring datasets for training

### Policy Training Architecture

- Uses convolutional neural networks for processing visual input
- Handles multiple input formats (images, state vectors)
- Implements standard reinforcement learning methods through the LERobot framework

### Evaluation Metrics

The system evaluates policies using:
- Success rate (percentage of successful task completions)
- Average reward
- Average completion time
- Collision rate

### Benchmarking

The benchmark feature compares trained policies against:
- Random policy baseline
- Rule-based policy baseline
- State-of-the-art benchmark algorithms

## APIs

The service provides several API endpoints:

- `/upload` - Upload an image
- `/detect` - Detect objects in an image
- `/train` - Train a policy based on objects and actions
- `/models` - List available trained models
- `/evaluate` - Evaluate a trained model
- `/run-benchmark` - Run benchmark analysis on a trained model

## Project Structure

```
automated_imitation_learning/
├── app.py                  # Main Flask application
├── object_detection.py     # Object detection module
├── robot_trainer.py        # Robot policy training module
├── datasets_config.py      # Dataset configuration
├── simple_benchmark.py     # Benchmark evaluation
├── run.py                  # Application runner
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables
├── static/                 # Static files
│   └── uploads/            # Uploaded images
│   └── benchmark_charts/   # Generated benchmark charts
├── templates/              # HTML templates
│   └── index.html          # Main UI template
├── models/                 # Trained models
└── data/                   # Application data storage
```

## License

MIT

## Credits

This project uses:
- LERobot for imitation learning
- OpenAI/OpenRouter APIs for vision language models
- PyTorch for neural network implementation 