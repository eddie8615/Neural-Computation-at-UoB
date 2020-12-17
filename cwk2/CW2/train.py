from torch.utils.data import DataLoader
from .utils.datautils import TrainDataset, TestDataset
from matplotlib import pyplot as plt

data_path = './data/train'
num_workers = 4
batch_size = 4
train_set = TrainDataset(data_path)
training_data_loader = DataLoader(dataset=train_set, num_workers=num_workers, batch_size=batch_size, shuffle=True)

# Fetch images and labels.
for iteration, sample in enumerate(training_data_loader):
    img, mask = sample

    show_image_mask(img[0, ...].squeeze(), mask[0, ...].squeeze())  # visualise all data in training set
    plt.pause(1)

    # Write your FORWARD below
    # Note: Input image to your model and ouput the predicted mask and Your predicted mask should have 4 channels

    # Then write your BACKWARD & OPTIMIZE below
    # Note: Compute Loss and Optimize


def show_image_mask(img, mask, cmap='gray'): # visualisation
    fig = plt.figure(figsize=(5,5))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap=cmap)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap=cmap)
    plt.axis('off')