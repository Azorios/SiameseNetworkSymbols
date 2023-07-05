import torch
import torch.nn.functional as F
import torchvision.utils
import matplotlib.pyplot as plt
from utils import imshow


def testing(test_loader, model, device):
    # test one image
    dataiter = iter(test_loader)
    x0, _, _ = next(dataiter)

    model.eval()

    for i in range(10):
        # Iterate over 10 images and test them with the first image (x0)
        _, x1, label2 = next(dataiter)

        # Concatenate the two images together
        concatenated = torch.cat((x0, x1), 0)

        output1, output2 = model(x0.to(device), x1.to(device))
        euclidean_distance = F.pairwise_distance(output1, output2)
        imshow(torchvision.utils.make_grid(concatenated), f'Dissimilarity: {euclidean_distance.item():.2f}')

    return None
def testing2(test_loader, model, device):
    # test one image
    dataiter = iter(test_loader)
    x0, _, _, class0, _ = next(dataiter)

    model.eval()

    for i in range(20):
        # Iterate over 10 images and test them with the first image (x0)
        _, x1, label, _, class1 = next(dataiter)

        # Concatenate the two images together
        concatenated = torch.cat((x0, x1), 0)

        output1, output2 = model(x0.to(device), x1.to(device))
        euclidean_distance = F.pairwise_distance(output1, output2)
        imshow(torchvision.utils.make_grid(concatenated),
               f'Class1: {class0[0]}\nClass2: {class1[0]}',
               f'Dissimilarity: {euclidean_distance.item():.2f}', i)

    return None
