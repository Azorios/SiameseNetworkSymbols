# pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# pip3 install pandas seaborn

import torch
import torchvision
import torchvision.datasets as datasets
import pandas as pd
import numpy as np
from dataset import SymbolsDataset
from help_functions import imshow, distribution, class_counts
from network import SiameseNetwork
from contrastive_loss import ContrastiveLoss
from training import training
from testing import testing
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch import optim


if __name__ == '__main__':
    # used device for computing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device, 'will be used.')

    transforms = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor()
    ])

    # get dataset
    dataset = SymbolsDataset(csv_file='./data/symbols_pixel.csv', transform=transforms)

    # get class labels of filtered dataset and show distribution
    class_labels = dataset.data.iloc[:, -1].values[dataset.filtered_dataset]
    #distribution(class_labels)

    # split dataset into train, val and test setstratify = class_labels
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42, stratify=class_labels)
    train_dataset, val_dataset = train_test_split(train_dataset, test_size=0.1, random_state=42)

    # load datasets
    train_loader = DataLoader(train_dataset, shuffle=True, num_workers=0, batch_size=32)
    val_loader = DataLoader(val_dataset, shuffle=True, num_workers=0, batch_size=32)
    test_loader = DataLoader(test_dataset, shuffle=True, num_workers=0, batch_size=1)

    class_counts(train_loader, val_loader, test_loader)

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
        model.load_state_dict(torch.load("./output/model.pth"))
    except FileNotFoundError:
        pass

    # loss function and optimizer
    loss_fn = ContrastiveLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    # train and save model
    _ = training(train_loader, val_loader, device, optimizer, model, loss_fn)

    # testing
    _ = testing(test_loader, model, device)

    # get dataset from local folder
    #data = pd.read_csv()
    #print(data.head())
    #print(data.columns)
    #print(data.shape)

    # # generate range of ints for suffixes with length exactly half of num_cols
    # # if num_cols is even, truncate concatenated list later to get to original list length
    # num_cols = data.shape[1]
    # new_cols = ['p_' + str(i) for i in (range(1, num_cols))]
    # new_cols.append('label')
    #
    # # ensure length of the new columns is equal to length of data's columns
    # data.columns = new_cols[:num_cols]
    # print(data.columns)
    #
    # # extract all labels
    # labels = data['label']
    # images = data.iloc[:, 0:num_cols - 1]
    # print(type(labels))
    # print(type(images))
    # print(images.head())
    #
    # imshow(images, num_rows=data.shape[0])
    #
    # # remove symbols with less than 7 instances
    # data = data[~data['label'].isin(['Ultrasonic Flow Meter', 'Barred Tee', 'Temporary Strainer',
    #                                  'Control Valve Angle Choke', 'Line Blindspacer', 'Vessel',
    #                                  'Valve Gate Through Conduit', 'Deluge', 'Control Valve'])]
    # # check class distribution
    # print(data['label'].value_counts())
    # distribution(labels=data['label'])
    # print(f'There are {len(data.label.unique())} Unique Symbol in the dataset')  # check number of labels
    #
    # dataset = data.values
    #
    # x = dataset[:, :-1]
    # y = dataset[:, -1]


# TODO
# upsample die mit wenigen instanzen durch rotation
# validation aus training entfernen? loss ändert sich nach undersampling nicht mehr stark, evtl stratify
# undersampling only on train
