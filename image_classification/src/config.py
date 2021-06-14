import os
import json

SM_TRAINING_ENV = json.loads(os.environ['SM_TRAINING_ENV'])

MODEL_DIR = SM_TRAINING_ENV['model_dir']
OUTPUT_DIR = SM_TRAINING_ENV['output_data_dir']
INPUT_DIR = SM_TRAINING_ENV['channel_input_dirs']

TEST_DIR = INPUT_DIR['test']
VALIDATION_DIR = INPUT_DIR['validation']
TRAIN_DIR = INPUT_DIR['train']
