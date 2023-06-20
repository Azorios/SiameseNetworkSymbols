# pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# pip3 install pandas seaborn

import torch
import torchvision
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset import SymbolsDataset, TransformDataset
from help_functions import imshow, resample, distribution, class_counts
from network import SiameseNetwork
from contrastive_loss import ContrastiveLoss
from training import training
from testing import testing
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch import optim
import random


if __name__ == '__main__':
    # used device for computing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device, 'will be used.')

    rotation_angles = [(90, 90), (180, 180), (270, 270)]

    transforms = {
        'general': transforms.Compose([
            transforms.Resize((100, 100)),
            transforms.ToTensor()
        ]),
        'train': transforms.Compose([
            transforms.RandomApply([
                transforms.RandomRotation(random.choice(rotation_angles))
            ])
        ])
    }

    # get dataset
    dataset = SymbolsDataset(csv_file='./data/symbols_pixel.csv', transform=transforms['general'])

    # get class labels of filtered dataset and show distribution
    class_labels = dataset.data.iloc[:, -1].values[dataset.filtered_dataset]
    distribution(class_labels)

    # split dataset into train, val and test set stratify = class_labels
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
    #train_dataset, val_dataset = train_test_split(train_dataset, test_size=0.2, random_state=42)

    # undersample classes with more than 100 instances and oversample classes with less than 100  to reach 100 per class
    resampled_train_dataset = resample(train_dataset)

    # rotate images in train dataset
    #train_dataset = TransformDataset(resampled_train_dataset, transform=transforms['train'])
    #test_dataset = TransformDataset(test_dataset)

    # load datasets
    train_loader = DataLoader(resampled_train_dataset, shuffle=True, num_workers=0, batch_size=64)
    #val_loader = DataLoader(val_dataset, shuffle=True, num_workers=0, batch_size=64)
    test_loader = DataLoader(test_dataset, shuffle=True, num_workers=0, batch_size=1)

    print("Train Dataset: ")
    class_counts(train_loader)
    print("Test Dataset: ")
    class_counts(test_loader)

    # extract one batch for visualisation
    vis_loader = DataLoader(train_dataset, shuffle=True, num_workers=0, batch_size=8)
    batch = next(iter(vis_loader))

    # Example batch is a list containing 2x8 images, indexes 0 and 1 and the label.
    # label = 1 means not the same person, label = 0 means the same person
    concatenated = torch.cat((batch[0], batch[1]), dim=0)

    # show example batch and print labels accordingly
    labels = batch[2].squeeze().tolist()
    imshow(torchvision.utils.make_grid(concatenated), labels)
    print(labels)

    # model
    model = SiameseNetwork().to(device)
    try:
        model.load_state_dict(torch.load("output/model.pth"))
    except FileNotFoundError:
        pass

    # loss function and optimizer
    loss_fn = ContrastiveLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    #scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True)

    # train and save model
    #_ = training(train_loader, val_loader, device, optimizer, model, loss_fn, scheduler)
    #_ = training(train_loader, device, optimizer, model, loss_fn)

    # testing
    _ = testing(test_loader, model, device)
