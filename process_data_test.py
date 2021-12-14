from collections import Counter
import os

import torch
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import xmltodict
from utils.config import batch_size, train_dataset_size

path_image = "original dataset/images/"
path_annotations = "original dataset/annotations/"
options = {"with_mask": 0, "without_mask": 1, "mask_weared_incorrect": 2}


def splitting_dataset_training_test():
    my_transform = transforms.Compose([transforms.Resize((226, 226)),
                                       transforms.RandomRotation(degrees=(0, 180)),
                                       transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
                                       transforms.ColorJitter(brightness=.5, hue=.3),
                                       transforms.RandomAdjustSharpness(sharpness_factor=2),
                                       transforms.RandomAutocontrast(),
                                       transforms.RandomEqualize(),
                                       transforms.ToTensor()])
    mydataset = datasets.ImageFolder("Processed Dataset", transform=my_transform)
    train_size = int(len(mydataset) * train_dataset_size)
    test_size = len(mydataset) - train_size
    print('Length of dataset is', len(mydataset), '\nLength of training set is :', train_size,
          '\nLength of test set is :', test_size)
    trainset, testset = torch.utils.data.random_split(mydataset, [train_size, test_size])

    train_dataloader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=True, num_workers=4)

    return train_dataloader, test_dataloader


if __name__ == '__main__':

    splitting_dataset_training_test()
