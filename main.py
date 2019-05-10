import pandas as pd 
import numpy as np    
import matplotlib.pyplot as plt   
import torch 
import torchvision 
from torchvision import transforms
from sklearn.model_selection import train_test_split
import argparse
from dataset import iMetDataset
from transform import RandomSizedCrop
from model import mymodel

def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--model', default='resnet50')
    arg('--batch-size', type=int, default=64)
    arg('--lr', type=float, default=1e-4)
    arg('--patience', type=int, default=4)
    arg('--epochs', type=int, default=100)
    arg('--fold', type=int, default=0)
    args = parser.parse_args()


    transformed_dataset = iMetDataset(csv_file='train.csv', 
                                  label_file="labels.csv", 
                                  img_path="train_unzip/", 
                                  root_dir='../input/',
                                  transform=transforms.Compose([
                                      RandomSizedCrop((256)),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(
                                          [0.485, 0.456, 0.406], 
                                          [0.229, 0.224, 0.225])
                                  ]))

    for i in range(len(transformed_dataset)):
    
        sample = transformed_dataset[i]
        print(i, sample['image'].size())
        print(i, sample['labels'])
        if i == 3:
            break

    model = mymodel()    

    

if __name__ == '__main__':
    main()
