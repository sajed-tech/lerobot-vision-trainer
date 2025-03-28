# Automated Imitation Learning Pipeline

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

An end-to-end platform for robot imitation learning - from visual perception to policy deployment.

## Overview

This pipeline simplifies the process of training robot policies from visual data. In just a few steps, you can go from images of objects to trained policies that can be deployed on robotic systems:

1. **Upload images** of objects and environments
2. **Detect objects and actions** using advanced vision-language models
3. **Train policies** using datasets that match the detected objects and actions
4. **Evaluate performance** in simulation
5. **Benchmark** against baseline methods

## Quick Start

1. Clone the repository
2. Install the requirements:
   ```
   cd robot_vision_trainer
   pip install -r requirements.txt
   ```
3. Start the application:
   ```
   python run.py --port 5000
   ```
4. Access the web interface:
   ```
   http://localhost:5000
   ```

## Documentation

For detailed documentation, see the [README](robot_vision_trainer/README.md) in the robot_vision_trainer directory.

## License

MIT

## Credits

This project uses:
- LERobot for imitation learning
- OpenAI/OpenRouter APIs for vision language models
- PyTorch for neural network implementation 