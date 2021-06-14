from config import *

with open(f'{MODEL_DIR}/file_model.txt', 'w') as fh:
    fh.write('Sample Text added')
    
with open(f'{OUTPUT_DIR}/file_output_data.txt', 'w') as fh:
    fh.write('Sample Text added')


# def parse_args():
#     parser = ArgumentParser()

#     parser.add_argument('--model_dir', type=str)
#     parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
#     parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
#     parser.add_argument('--hosts', type=list, default=json.loads(os.environ.get('SM_HOSTS')))
#     parser.add_argument('--current-host', type=str, default=os.environ.get('SM_CURRENT_HOST'))

#     return parser.parse_known_args()


# def load_training_data(base_dir):
#     print('loading training data')
#     print(os.listdir(base_dir))


# def load_validation_data(base_dir):
#     print('loading validation data')
#     print(os.listdir(base_dir))
    
    
# if __name__ == "__main__":
#     args, _ = parse_args()
    
#     print(args)

#     load_training_data(args.train)
#     load_validation_data(args.train)
