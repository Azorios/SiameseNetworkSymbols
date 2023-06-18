import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import defaultdict


def distribution(class_labels):
    dist = pd.Series(class_labels).value_counts().reset_index()

    # Rename columns
    dist.columns = ['Class Label', 'Count']

    # Display the class counts
    print(dist)
    dist.plot(kind='bar')
    plt.show()


def class_counts(train_loader, val_loader, test_loader):

    instances = defaultdict(int)
    for batch in train_loader:
        class_labels = batch[3]
        for label in class_labels:
            instances[label] += 1

    # Print the class counts
    for label, count in instances.items():
        print(f"Class Label: {label}, Count: {count}")


def imshow(img, title, text=None):
    npimg = img.numpy()
    plt.axis("off")
    plt.title(title)
    if text:
        plt.text(75, 8, text, style='italic', fontweight='bold',
                 bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})

    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def show_plot(train_iteration, val_iteration, train_loss, val_loss, best_loss):
    plt.figure(figsize=(10, 5))

    # Plot train loss
    plt.subplot(1, 2, 1)
    plt.plot(train_iteration, train_loss)
    plt.xlabel('Iteration')
    plt.ylabel('Train Loss')
    plt.title('Train Loss')

    # Plot validation loss
    plt.subplot(1, 2, 2)
    plt.plot(val_iteration, val_loss)
    plt.xlabel('Iteration')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss')

    plt.tight_layout()

    plt.savefig(f"./output/loss_{best_loss}.png")
    plt.show()


def save_best_loss(best_loss):
    with open("./output/best_loss.txt", "w") as file:
        file.write(str(best_loss.item()))


def load_best_loss():
    try:
        with open("./output/best_loss.txt", "r") as file:
            best_loss = float(file.read())
    except FileNotFoundError:
        best_loss = float('inf')

    print(f"Best Validation Loss: {best_loss}")
    return best_loss
