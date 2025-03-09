from turtle import down
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torchvision
from torchvision.transforms import ToTensor
import numpy as np
from IPython.display import Image, display

classes = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

training_data = torchvision.datasets.CIFAR10(
    root='./Lab4/Data/dataset',
    train=True,
    transform=ToTensor()
)

test_data = torchvision.datasets.CIFAR10(
    root='./Lab4/Data/dataset',
    train=False,
    transform=ToTensor()
)

batch_size = 64

# Create data loaders
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

for image, labels in train_dataloader:
    print(labels.shape)
    print(image.shape)
    break

def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Get some random training images
dataiter = iter(train_dataloader)
images, labels = next(dataiter)

# Show images
imshow(torchvision.utils.make_grid(images))

# Print labels
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))