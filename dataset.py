import torch
import pandas as pd
import numpy as np
from skimage.io import imread   
import cv2 
from PIL import Image
from torch.utils.data import Dataset

class iMetDataset(Dataset):
    """iMet dataset."""

    def __init__(self, csv_file, label_file, img_path, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with train_data.
            label_file (string): Path to the csv file with labels.
            img_path (string): Directory with all the images.
            root_dir (string): root Directory.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = pd.read_csv(root_dir + csv_file)
        self.labels = pd.read_csv(root_dir + label_file)
        self.transform = transform
        self.img_path = root_dir + img_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        
        img_id = self.data.iloc[index,:]["id"]
        labels = self.data.iloc[index,:]["attribute_ids"].split(" ")
        
        image = imread(self.img_path + img_id + ".png")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)

        labels = list(map(int, labels))
        one_hot = np.zeros (1103)
        for label in labels: one_hot[label] = 1
            
        sample = {'image': image, 'img_id': img_id, "labels":torch.from_numpy(one_hot)}
        
        return sample
