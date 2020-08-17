import os
import torch
from torchvision import datasets, transforms
import numpy as np
import random

def process(data_path='./Data', batch_size = 100, apply_transform=False):

    if apply_transform == True:
        transform = transforms.Compose([transforms.Grayscale(),
                                        transforms.Resize((150, 150)),
                                        transforms.RandomAffine(degrees=(0,90), translate=(0.2,0.2)),
                                        transforms.ToTensor()])
                                        
    else:
        transform = transforms.Compose([transforms.Grayscale(),
                                        transforms.Resize((150, 150)),
                                        transforms.ToTensor()])

    data = datasets.ImageFolder(data_path, transform = transform)
    data_loader = torch.utils.data.DataLoader(data, batch_size = batch_size, shuffle=True)

    x_train = []
    for x, labels in data_loader:
        x_train.append(x.numpy())
    x_train = np.asarray(x_train)
    
    print('Data Processed!')
    print('Shape of the Data: ' + str(x_train.shape))
    
    return x_train
    



