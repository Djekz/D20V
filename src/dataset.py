# 

import os
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset

class D20V(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.file_list = os.listdir(data_dir)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        filename = os.path.join(self.data_dir, self.file_list[idx])
        # Example: Load audio file and its label
        audio_data, label = self.load_audio_and_label(filename)

        if self.transform:
            audio_data = self.transform(audio_data)

        return audio_data, label

    def load_audio_and_label(self, filename):
        # Example: Load audio using librosa and return data along with label
        # Here, you may need to implement your own logic to load audio and labels
        # Return audio data and label
        return audio_data, label
# dataset.py

import os
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset

class D20V(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.file_list = os.listdir(data_dir)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        filename = os.path.join(self.data_dir, self.file_list[idx])
        # Example: Load audio file and its label
        audio_data, label = self.load_audio_and_label(filename)

        if self.transform:
            audio_data = self.transform(audio_data)

        return audio_data, label

    def load_audio_and_label(self, filename):
        # Example: Load audio using librosa and return data along with label
        # Here, you may need to implement your own logic to load audio and labels
        # Return audio data and label
        return audio_data, label
