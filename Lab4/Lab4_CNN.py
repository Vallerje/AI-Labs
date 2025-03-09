from turtle import down
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torchvision
from torchvision.transforms import ToTensor
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


classes = [
    "plane", 
    "car", 
    "bird", 
    "cat", 
    "deer", 
    "dog", 
    "frog", 
    "horse", 
    "ship", 
    "truck"]

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

""" for image, labels in train_dataloader:
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
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size))) """

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

net = Net()
print(net)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(20):     #each forward+backward = 1 epoch
    running_loss = 0.0
    for i, data in enumerate(train_dataloader, 0):
        #get the input, data is a list [inputs, labels]
        inputs, labels = data

        #forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #print statisitcs
        running_loss += loss.item() #item() converts loss value to a standard Python number
    print(f"Epoch {epoch}", f" number of images:{i}", "loss: ", running_loss)


correct = 0
total = 0

with torch.no_grad():
    for data in test_dataloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy of the network on the 10000 test images: {100 * correct / total}%")

torch.save(net.state_dict(), "./Lab4/Models/cifar_net.pth")