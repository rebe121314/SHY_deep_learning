# data_processing.py

import pandas as pd
import os
import pickle

def save_processed_data(sample_name, image_name, data, processed_data_folder):
    sample_folder = os.path.join(processed_data_folder, sample_name)
    if not os.path.exists(sample_folder):
        os.makedirs(sample_folder)
    file_path = os.path.join(sample_folder, f'{image_name}.pkl')
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def load_processed_data(sample_name, image_name, processed_data_folder):
    file_path = os.path.join(processed_data_folder, sample_name, f'{image_name}.pkl')
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    return None
