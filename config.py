import os

# Make sure this points to the actual project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))  # If config.py is in root

# Now correct the model path
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'trained_model.pkl')

# Model configuration
INPUT_SIZE = 784
HIDDEN_SIZE = 128
OUTPUT_SIZE = 10
LEARNING_RATE = 0.01
EPOCHS = 150
