import os
from operator import index
import math
import pandas as pd
import numpy as np
import librosa
import librosa.display
import warnings

folder_path = "./mfcc_npy/"
sample_rate = 22050
duration = 30  # mesaured in seconds
samples_per_track = sample_rate * duration

def extract_mfcc_into_npy(n_mfcc, hop_length, num_segments, dataset_type, art):
    alldata = read_dataset(dataset_type, art)
    num_samples_per_segment = int(samples_per_track / num_segments)
    expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length)
    
    for meter, group in alldata.groupby(['meter']):
        for index, song in group.iterrows():
            path = os.path.relpath("./dataverse_files/" + song["filename"])
            signal, sr = librosa.load(path, sr=sample_rate)
            name = os.path.basename(song["filename"]).replace(".wav","")
            for s in range(num_segments):
                start_sample = num_samples_per_segment * s
                finish_sample = start_sample + num_samples_per_segment

                mfcc = librosa.feature.mfcc(y=signal[start_sample:finish_sample], n_mfcc=n_mfcc, sr=sr, hop_length=hop_length)
                mfcc = mfcc.T

                if len(mfcc) == expected_num_mfcc_vectors_per_segment:
                    target_dir = os.path.join(folder_path, dataset_type, str(meter[0]))
                    os.makedirs(target_dir, exist_ok=True)
                    npy_filename = f"{name}-{s}.npy"
                    np.save(os.path.join(target_dir, npy_filename), mfcc)

def read_dataset(dataset_type, art=False):
    if art!=False and dataset_type == "train":
        dataset_type=dataset_type+'_art'
    file_name = f"./dataverse_files/data_{dataset_type}_4_classes.csv"
    print(file_name)
    return pd.read_csv(file_name)

def main():
    n_mfcc = 13
    hop_length = 512
    num_segments = 10
    art = False
    # For each dataset type
    for dataset_type in ['train', 'val', 'test']:
        print(f"Processing {dataset_type}")
        extract_mfcc_into_npy(n_mfcc, hop_length, num_segments, dataset_type, art)

if __name__ == "__main__":
    main()