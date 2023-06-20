import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
import random


class SymbolsDataset(Dataset):
    """ Custom dataset for image pairs from a CSV file."""

    def __init__(self, csv_file, transform=None):
        """
        Arguments:
        - csv_file (str): Path to the CSV file containing image data and labels.
        - transform (callable, optional): Optional transformations to be applied to the images.
        """

        self.data = pd.read_csv(csv_file, header=None)
        self.filtered_dataset = self.filter_dataset()
        self.transform = transform

    def filter_dataset(self):
        # Count the number of instances per class
        class_counts = self.data.iloc[:, -1].value_counts()

        # Filter out classes with less than 7 instances
        valid_classes = class_counts[class_counts >= 100].index

        # Filter the dataset indices based on the valid classes
        filtered_dataset = [
            idx for idx, label in enumerate(self.data.iloc[:, -1]) if label in valid_classes
        ]
        # Limit each class to have a maximum of 50 instances
        # filtered_dataset = []
        # for label in valid_classes:
        #     instances = self.data[self.data.iloc[:, -1] == label]
        #     sampled_instances = instances.sample(n=min(class_counts[label], 50))
        #     filtered_dataset.extend(sampled_instances.index)

        return filtered_dataset

    def __len__(self):
        return len(self.filtered_dataset)

    def __getitem__(self, idx):
        # First image
        img0_idx = random.choice(self.filtered_dataset)
        class0 = self.data.iloc[img0_idx, -1]

        should_get_same_class = random.randint(0, 1)    # 0 for yes and 1 for no
        if should_get_same_class:
            while True:
                # Look until the same class image is found
                img1_idx = random.choice(self.filtered_dataset)
                class1 = self.data.iloc[img1_idx, -1]

                if class0 == class1:
                    break
        else:
            while True:
                # Look until a different class image is found
                img1_idx = random.choice(self.filtered_dataset)
                class1 = self.data.iloc[img1_idx, -1]

                if class0 != class1:
                    break

        # Images
        img0 = np.reshape(self.data.iloc[img0_idx, :-1].values, (100, 100)).astype(np.uint8)
        img1 = np.reshape(self.data.iloc[img1_idx, :-1].values, (100, 100)).astype(np.uint8)

        img0 = Image.fromarray(img0, mode='L')
        img1 = Image.fromarray(img1, mode='L')

        # Label is set to 0 if images are of same class, 1 if images have different class
        label = int(class0 != class1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, label, class0, class1


class TransformDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img0, img1, label, class0, class1 = self.subset[idx]
        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, label, class0, class1

