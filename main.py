# pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# pip3 install pandas seaborn

import torch
import torchvision.datasets as datasets
import torchvision
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset import SymbolsDataset, TransformDataset, SiameseNetworkDataset
from utils import resample, distribution, class_counts, imshow, process_data
from network import SiameseNetwork
from contrastive_loss import ContrastiveLoss, contrastive_loss_with_margin
from training import training
from testing import testing
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch import optim
import random
import tensorflow as tf
import numpy as np


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

    # get dataset from local folder
    train_dataset = datasets.ImageFolder("./dataset/train/")
    test_dataset = datasets.ImageFolder("./dataset/test/")


    # initialise dataset for training and testing
    siamese_dataset_train = SiameseNetworkDataset(imageFolderDataset=train_dataset, transform=transforms['general'])
    siamese_dataset_test = SiameseNetworkDataset(imageFolderDataset=test_dataset, transform=transforms['general'])

    # dataloaders for visualisation, training and testing
    vis_loader = DataLoader(siamese_dataset_train, shuffle=True, num_workers=0, batch_size=8)
    train_loader = DataLoader(siamese_dataset_train, shuffle=True, num_workers=0, batch_size=64)
    test_loader = DataLoader(siamese_dataset_test, shuffle=True, num_workers=0, batch_size=1)

    # extract one batch for visualisation
    batch = next(iter(vis_loader))

    # Example batch is a list containing 2x8 images, indexes 0 and 1 and the label.
    # label = 1 means not the same person, label = 0 means the same person
    concatenated = torch.cat((batch[0], batch[1]), 0)

    # show example batch and print labels accordingly
    imshow(torchvision.utils.make_grid(concatenated))
    print(batch[2].detach().cpu().numpy().reshape(-1))

    # model
    model = SiameseNetwork().to(device)
    try:
        model.load_state_dict(torch.load("output/model.pth"))
    except FileNotFoundError:
        pass

    # loss function and optimizer
    loss_fn = ContrastiveLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    # train and save model
    #_ = training(train_loader, device, optimizer, model, loss_fn)

    # testing
    _ = testing(test_loader, model, device)

    # # get dataset
    # dataset = SymbolsDataset(csv_file='./data/symbols_pixel.csv', transform=transforms['general'])
    #
    # # get class labels of filtered dataset and show distribution
    # class_labels = dataset.data.iloc[:, -1].values[dataset.filtered_dataset]
    # distribution(class_labels)
    #
    # # x = image pairs, y = labels
    # x = [dataset[i][0:2] for i in range(len(dataset))]
    # y = [dataset[i][2] for i in range(len(dataset))]
    #
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
    #
    # x_train = x_train.reshape(x_train.shape[0], 100, 100, 1).astype('float32')
    # x_test = x_test.reshape(x_test.shape[0], 100, 100, 1).astype('float32')
    #
    # print(x_train.shape)

    # split dataset into train, val and test set stratify = class_labels
    #train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)


    # X_train_pairs = [train_dataset[i][0:2] for i in range(len(train_dataset))]
    # Y_train = [train_dataset[i][2] for i in range(len(train_dataset))]
    #
    # X_test_pairs = [test_dataset[i][0:2] for i in range(len(test_dataset))]
    # Y_test = [test_dataset[i][2] for i in range(len(test_dataset))]
    #
    #
    # #print(Y_test)
    #
    # print(X_train_pairs)
    # left_input = tf.keras.layers.Input((100, 100, 1))
    # right_input = tf.keras.layers.Input((100, 100, 1))
    #
    # # Create a base network
    # model = create_subnetwork()
    #
    # # Generate the encodings (feature vectors) for the two images
    # encoded_l = model(left_input)
    # encoded_r = model(right_input)
    #
    # # Add a customized layer to compute the absolute difference between the encodings
    # L1_layer = tf.keras.layers.Lambda(lambda tensors: tf.math.abs(tensors[0] - tensors[1]))
    # L1_distance = L1_layer([encoded_l, encoded_r])
    #
    # # Add a dense layer with a sigmoid unit to generate the similarity score
    # prediction = tf.keras.layers.Dense(1, activation='sigmoid')(L1_distance)
    #
    # # Connect the inputs with the outputs
    # siamese_net = tf.keras.models.Model(inputs=[left_input, right_input], outputs=prediction)
    #
    # siamese_net.compile(optimizer=tf.keras.optimizers.Adam(lr=0.00006), loss=contrastive_loss_with_margin(margin=1))
    # results = siamese_net.fit([train_dataset, train_dataset[:, 1]], train_dataset[:, 2], epochs=10)
    #
    # history_dict = results.history
    # print(history_dict.keys())


    # train_dataset, val_dataset = train_test_split(train_dataset, test_size=0.2, random_state=42)

    # undersample classes with more than 100 instances and oversample classes with less than 100  to reach 100 per class
    #resampled_train_dataset = resample(train_dataset)

    # rotate images in train dataset
    # train_dataset = TransformDataset(resampled_train_dataset, transform=transforms['train'])
    # test_dataset = TransformDataset(test_dataset)

    # load datasets
    # train_loader = DataLoader(train_dataset, shuffle=True, num_workers=0, batch_size=64)
    # # val_loader = DataLoader(val_dataset, shuffle=True, num_workers=0, batch_size=64)
    # test_loader = DataLoader(test_dataset, shuffle=True, num_workers=0, batch_size=1)
    #
    # print("Train Dataset: ")
    # class_counts(train_loader)
    # print("Test Dataset: ")
    # class_counts(test_loader)
    #
    # # extract one batch for visualisation
    # vis_loader = DataLoader(train_dataset, shuffle=True, num_workers=0, batch_size=8)
    # batch = next(iter(vis_loader))
    #
    # # Example batch is a list containing 2x8 images, indexes 0 and 1 and the label.
    # # label = 1 means not the same person, label = 0 means the same person
    # concatenated = torch.cat((batch[0], batch[1]), dim=0)
    #
    # # show example batch and print labels accordingly
    # labels = batch[2].squeeze().tolist()
    # imshow(torchvision.utils.make_grid(concatenated), labels)
    # print(labels)
    #
    # # model
    # model = SiameseNetwork().to(device)
    # try:
    #     model.load_state_dict(torch.load("output/model.pth"))
    # except FileNotFoundError:
    #     pass
    #
    # # loss function and optimizer
    # loss_fn = ContrastiveLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.0005)
    # #scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True)
    #
    # # train and save model
    # #_ = training(train_loader, val_loader, device, optimizer, model, loss_fn, scheduler)
    # _ = training(train_loader, device, optimizer, model, loss_fn)
    #
    # #testing
    # _ = testing(test_loader, model, device)
