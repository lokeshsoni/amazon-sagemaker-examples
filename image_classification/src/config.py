import os

SM_TRAINING_ENV = json.loads(os.environ['SM_TRAINING_ENV'])

MODEL_DIR = SM_TRAINING_ENV['model_dir']
OUTPUT_DIR = SM_TRAINING_ENV['output_data_dir']
