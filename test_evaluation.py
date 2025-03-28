import os
import sys
import json
import signal
import time

# Change directory to the project root before imports
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Setup signal handler for timeout
def timeout_handler(signum, frame):
    print("\n\nTIMEOUT: Evaluation process took too long to complete!")
    print("This indicates the process is hanging somewhere in the evaluation.")
    sys.exit(1)

# Set a timeout of 60 seconds
signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(60)

# Now import from the correct location
print("Importing RobotTrainer...")
from robot_vision_trainer.robot_trainer import RobotTrainer

# Set environment variables
os.environ["MUJOCO_PATH"] = "/Users/sajed/mujoco/mujoco237"
os.environ["MUJOCO_PLUGIN_PATH"] = "/Users/sajed/mujoco/mujoco237/bin"

print("Setting up trainer...")
# Set up a dummy token (not actually used for evaluation)
trainer = RobotTrainer(hf_token="dummy_token", output_dir="models", skip_login=True)

# Test with an existing model
model_path = "models/robot_policy_2025_03_24_17_57_24.pt"

print(f"Testing evaluation with model: {model_path}")

# Add debug print statements around key steps
print("Step 1: Starting evaluation process...")
try:
    print("Step 2: Calling evaluate_policy method...")
    results = trainer.evaluate_policy(model_path=model_path, num_episodes=1)  # Use just 1 episode for faster testing
    
    print("Step 3: Evaluation completed!")
    # Print the results
    print("\nEvaluation Results:")
    print(json.dumps(results, indent=2))
    
except Exception as e:
    print(f"ERROR during evaluation: {e}")
    import traceback
    print(traceback.format_exc())

# Cancel the alarm if we get here
signal.alarm(0)
print("Test completed successfully") 