import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import defaultdict
from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import csv
from itertools import combinations


def imshow(img, text=None):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic', fontweight='bold',
                 bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})

    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()



def process_data(data):
    # Group images by class
    class_to_images = defaultdict(list)
    for image in data:
        pixels = image[:-1]
        class_label = image[-1]
        class_to_images[class_label].append(pixels)

    # Filter out classes with fewer than 7 instances
    class_to_images = {class_label: images for class_label, images in class_to_images.items() if len(images) >= 7}

    # Generate image pairs and labels
    pairs = []
    labels = []
    for class_label, images in class_to_images.items():
        # Pairs of images within the same class (label 0)
        same_class_pairs = list(combinations(images, 2))
        pairs.extend(same_class_pairs)
        labels.extend([0]*len(same_class_pairs))

        # Pairs of images from different classes (label 1)
        for other_class_label, other_images in class_to_images.items():
            if class_label != other_class_label:
                diff_class_pairs = list(combinations(images + other_images, 2))
                pairs.extend(diff_class_pairs)
                labels.extend([1]*len(diff_class_pairs))

    return pairs, labels


def distribution(class_labels):
    dist = pd.Series(class_labels).value_counts().reset_index()

    # Rename columns
    dist.columns = ['Class Label', 'Count']

    # Display the class counts
    print(dist)
    dist.plot(kind='bar')
    plt.show()


def class_counts(data_loader):

    instances = defaultdict(int)
    for batch in data_loader:
        class_labels = batch[3]
        for label in class_labels:
            instances[label] += 1

    # Print the class counts
    for label, count in instances.items():
        print(f"Class Label: {label}, Count: {count}")


# def imshow(img, title, text=None, i=None):
#     npimg = img.numpy()
#     plt.axis("off")
#     plt.title(title)
#     if text:
#         plt.text(75, 8, text, style='italic', fontweight='bold',
#                  bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})
#
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.savefig(f'output/dissimilarity{i}.png')
#     plt.show()

def show_plot(iteration, loss):
    plt.plot(iteration, loss)
    plt.savefig(f"./output/loss_{loss[-1]}.png")
    plt.show()


# def show_plot(train_iteration, val_iteration, train_loss, val_loss, best_loss):
#     plt.figure(figsize=(10, 5))
#
#     # Plot train loss
#     plt.subplot(1, 2, 1)
#     plt.plot(train_iteration, train_loss)
#     plt.xlabel('Iteration')
#     plt.ylabel('Train Loss')
#     plt.title('Train Loss')
#
#     # Plot validation loss
#     plt.subplot(1, 2, 2)
#     plt.plot(val_iteration, val_loss)
#     plt.xlabel('Iteration')
#     plt.ylabel('Validation Loss')
#     plt.title('Validation Loss')
#
#     plt.tight_layout()
#
#     plt.savefig(f"./output5/loss_{best_loss}.png")
#     plt.show()


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


def show_plot_simple(iteration, loss):
    plt.plot(iteration, loss)
    plt.savefig(f"./output/loss_{loss[-1]}.png")
    plt.show()


def resample(train_dataset):
    # Get class labels of the training dataset
    train_class_labels = [train_dataset[i][3] for i in range(len(train_dataset))]

    # Convert train_dataset to a DataFrame
    train_df = pd.DataFrame(train_dataset, columns=['img0', 'img1', 'label', 'class0', 'class1'])

    # Convert train_class_labels to a DataFrame
    train_labels_df = pd.DataFrame(train_class_labels, columns=['class_label'])

    # Calculate the class counts
    class_counts = train_labels_df['class_label'].value_counts()

    # Define the resampling ratios for classes with at least 100 instances
    undersampling_ratios = {
        class_label: 150 for class_label, count in class_counts.items() if count >= 150
    }

    upsampling_ratios = {
        class_label: 150 for class_label, count in class_counts.items() if count < 150
    }

    # Define the pipeline
    pipeline = make_pipeline(
        RandomUnderSampler(sampling_strategy=undersampling_ratios),  # randomly remove samples without replacement
        RandomOverSampler(sampling_strategy=upsampling_ratios)  # randomly duplicate sample
    )

    # Apply the pipeline to the training dataset and labels
    resampled_train_df, resampled_train_labels_df = pipeline.fit_resample(train_df, train_labels_df)

    # Convert resampled_train_df back to list of tuples
    resampled_train_dataset = resampled_train_df.to_records(index=False).tolist()

    # Convert resampled_train_labels_df back to a list
    resampled_train_class_labels = resampled_train_labels_df['class_label'].tolist()

    # Check the class distribution after resampling
    distribution(resampled_train_class_labels)

    return resampled_train_dataset
