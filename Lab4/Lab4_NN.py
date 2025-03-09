import json
from turtle import color
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
import torch.nn.functional as AF
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
#from torch.utils.data import DataLoader, random_split

import torchaudio
from torchaudio import transforms as T
import torchaudio.functional as AF

from torch.nn.utils.rnn import pad_sequence

import numpy as np
import matplotlib.pyplot as plt
from torchmetrics import Accuracy
from tqdm import tqdm

from IPython.display import Audio

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class M5(nn.Module):
    def __init__(self, n_input=1, n_output=35, stride=16, n_channel=32):
        super().__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return x.squeeze(1)


class AudioDataset(Dataset):
    def __init__(self, root_dir, json_path, subset="training"):

        # Load the JSON file
        with open(json_path, "r") as f:
            data = json.load(f)

        # Filter all files by category: training/testing
        self.files = [item for item in data["files"] if item["category"] == subset]
        self.root_dir = root_dir

        # Create label to index mapping
        unique_labels = sorted(set(item["label"]["label"] for item in self.files))
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_data = self.files[idx]

        # Load the audio file
        waveform, sample_rate = torchaudio.load(f"{self.root_dir}/{file_data['path']}")

        # Get specific label
        label = self.label_to_idx[file_data["label"]["label"]]
        return waveform, sample_rate, label


root_dir = "./Lab4/Data/audio_data"
json_path = "Lab4/Data/audio_data/info.labels"

num_cls = 4
epochs = 50
batch_size = 32

device = "cuda" if torch.cuda.is_available() else "cpu"

train_set = AudioDataset(root_dir, json_path, subset="training")
test_set = AudioDataset(root_dir, json_path, subset="testing")

labels = set([d[2] for d in train_set])
label2num = {label: num for num, label in enumerate(labels)}


def collate_fn(batch):
    data = [b[0][0] for b in batch]
    data = pad_sequence(data, batch_first=True)
    data = AF.resample(data, 16000, 8000).unsqueeze(1)
    labels = torch.LongTensor([label2num[b[2]] for b in batch])
    return data, labels


def train_one_epoch(model, train_loader, sloss_fn, optimizer, epoch=None):
    model.train()
    loss_train = AverageMeter()
    acc_train = Accuracy(task="multiclass", num_classes=num_cls).to(device)
    with tqdm(train_loader, unit="batch") as tepoch:
        for inputs, targets in tepoch:
            if epoch is not None:
                tepoch.set_description(f"Epoch {epoch}")
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)

            loss = loss_fn(outputs, targets)

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            loss_train.update(loss.item())
            acc_train(outputs, targets.int())
            tepoch.set_postfix(loss=loss_train.avg, accuracy=100.0 * acc_train.compute().item())
    return model, loss_train.avg, acc_train.compute().item()


def validation(model, test_loader, loss_fn):
    model.eval()
    with torch.no_grad():
        loss_valid = AverageMeter()
        acc_valid = Accuracy(task="multiclass", num_classes=num_cls).to(device)
        for i, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            loss_valid.update(loss.item())
            acc_valid(outputs, targets.int())
    return loss_valid.avg, acc_valid.compute().item()


train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

print(f"Number of training samples: {len(train_set)}")
print(f"Number of testing samples: {len(test_set)}")
print(f"Available labels: {list(train_set.label_to_idx.keys())}")


model = M5(n_input=1, n_output=num_cls).to(device)
loss_fn = nn.CrossEntropyLoss()

""" x_batch, y_batch = next(iter(train_loader))
outputs = model(x_batch.to(device))
loss = loss_fn(outputs, y_batch.to(device))
print(loss)


_, mini_train_dataset = random_split(train_set, (len(train_set) - 500, 500))
mini_train_loader = DataLoader(mini_train_dataset, 20, collate_fn=collate_fn)


optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

num_epochs = 100
for epoch in range(num_epochs):
     model, _, _ = train_one_epoch(model, mini_train_loader, loss_fn, optimizer, epoch)

num_epochs = 1
for lr in [0.01, 0.001, 0.0001]:
     print(f"LR={lr}")
     model = M5(n_input=1, n_output=NUM_OF_CLASSES).to(device)
     model = torch.load("model.pt")
     optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=1e-4, momentum=0.9)
     for epoch in range(num_epochs):
         model, _, _ = train_one_epoch(model, train_loader, loss_fn, optimizer, epoch)
     print()

num_epochs = 5

for lr in [0.05, 0.04, 0.03, 0.02, 0.01, 0.009, 0.008, 0.007, 0.006, 0.005]:
     for wd in [1e-4, 1e-5, 0.0]:
         model = M5(n_input=1, n_output=NUM_OF_CLASSES).to(device)
         optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
         print(f"LR={lr}, WD={wd}")

         for epoch in range(num_epochs):
             model, loss, _ = train_one_epoch(model, train_loader, loss_fn, optimizer, epoch)
         print()

lr = 0.05 and wd = 0.0 is the best settings 
 """
lr = 0.05
wd = 0.0
optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)

loss_train_hist = []
loss_valid_hist = []

acc_train_hist = []
acc_valid_hist = []

best_loss_valid = torch.inf
epoch_counter = 0

for epoch in range(epochs):

    # Train
    model, loss_train, acc_train = train_one_epoch(model, train_loader, loss_fn, optimizer, epoch)

    # Validation
    loss_valid, acc_valid = validation(model, test_loader, loss_fn)

    loss_train_hist.append(loss_train)
    loss_valid_hist.append(loss_valid)

    acc_train_hist.append(acc_train)
    acc_valid_hist.append(acc_valid)

    if loss_valid < best_loss_valid:
        torch.save(model, f"Lab4/Model/model.pt")
        best_loss_valid = loss_valid
        print("Model Saved!")

    print(f"Valid: Loss = {loss_valid:.4}, Acc = {acc_valid:.4}")
    print()

    epoch_counter += 1


plt.figure(figsize=(13, 6))

plt.subplot(1, 2, 1)
plt.plot(range(epoch_counter), loss_train_hist, color='red', label="Train")
plt.plot(range(epoch_counter), loss_valid_hist, color='blue', label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.title("Training and Validation Loss")


plt.subplot(1, 2, 2)
plt.plot(range(epoch_counter), acc_train_hist, color='red', label="Train")
plt.plot(range(epoch_counter), acc_valid_hist, color='blue', label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend()
plt.title("Training and Validation Accuracy")


plt.tight_layout()
plt.savefig("Lab4/Figure/training_curves.png")
plt.show()