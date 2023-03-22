#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 12:53:27 2023

@author: schama
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader


import pydicom as dicom



class DICOMDataset(Dataset):
    def __init__(self, dicom_dir, csv_file, transform=None):
        self.dicom_dir = dicom_dir
        self.labels = pd.read_csv(csv_file)
        self.imgs_names = os.listdir(dicom_dir)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Load DICOM image
        file_name = self.labels.iloc[idx, 0]
        file_path = os.path.join(self.dicom_dir, file_name)
        image_data = dicom.dcmread(file_path)
        img_array = np.array(image_data.pixel_array, dtype=np.float32)[np.newaxis]
        
        image_data = np.expand_dims(image_data, axis=0)
        image = torch.from_numpy(img_array)
        
        # Load labels
        label = self.labels.iloc[idx, 1]
        
        if self.transform:
            image = self.transform(image)
        
        
        return image, label

def plot_batch(batch_imgs, batch_labels):
    batch_size = len(batch_imgs)
    fig, axs = plt.subplots(1, batch_size, figsize=(10,5))
    for i in range(batch_size):
        axs[i].imshow(batch_imgs[i,0,:,:])
        axs[i].set_title(f"Label {batch_labels[i]}")
    plt.show()
