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


def extractions_images_and_annotatons():
    img_names = []
    xml_names = []
    for dirname, _, filenames in os.walk('original dataset'):
        for filename in filenames:
            if os.path.join(dirname, filename)[-3:] != "xml":
                img_names.append(filename)
            else:
                xml_names.append(filename)
    listing = []
    for i in img_names[:]:
        with open(path_annotations + i[:-4] + ".xml") as fd:
            doc = xmltodict.parse(fd.read())
        temp = doc["annotation"]["object"]
        if type(temp) == list:
            for i in range(len(temp)):
                listing.append(temp[i]["name"])
        else:
            listing.append(temp["name"])

    Items = Counter(listing).keys()
    values = Counter(listing).values()
    print(Items, '\n', values)
    return values, Items, img_names


def dataset_creation(image_list):
    my_transform = transforms.Compose([transforms.Resize((226, 226)),
                                       transforms.ToTensor()])
    image_tensor = []
    label_tensor = []
    for i, j in enumerate(image_list):
        with open(path_annotations + j[:-4] + ".xml") as fd:
            doc = xmltodict.parse(fd.read())
        if type(doc["annotation"]["object"]) != list:
            temp = doc["annotation"]["object"]
            x, y, w, h = list(map(int, temp["bndbox"].values()))
            label = options[temp["name"]]
            image = transforms.functional.crop(Image.open(path_image + j).convert("RGB"), y, x, h - y, w - x)
            image_tensor.append(my_transform(image))
            label_tensor.append(torch.tensor(label))
        else:
            temp = doc["annotation"]["object"]
            for k in range(len(temp)):
                x, y, w, h = list(map(int, temp[k]["bndbox"].values()))
                label = options[temp[k]["name"]]
                image = transforms.functional.crop(Image.open(path_image + j).convert("RGB"), y, x, h - y, w - x)
                image_tensor.append(my_transform(image))
                label_tensor.append(torch.tensor(label))

    final_dataset = [[k, l] for k, l in zip(image_tensor, label_tensor)]
    return tuple(final_dataset)


def splitting_dataset_training_test(mydataset):
    train_size = int(len(mydataset) * train_dataset_size)
    test_size = len(mydataset) - train_size
    print('Length of dataset is', len(mydataset), '\nLength of training set is :', train_size,
          '\nLength of test set is :', test_size)
    trainset, testset = torch.utils.data.random_split(mydataset, [train_size, test_size])
    train_dataloader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=True, num_workers=4)

    return train_dataloader, test_dataloader


if __name__ == '__main__':
    values, Items, img_names = extractions_images_and_annotatons()

    dataset = dataset_creation(img_names)
    splitting_dataset_training_test(dataset)
