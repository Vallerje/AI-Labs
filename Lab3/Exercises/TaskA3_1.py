import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, mean_squared_error
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

# build custom module for logistic regression
class LinearRegression(torch.nn.Module):    
    # build the constructor
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)

    # make predictions
    def forward(self, x):
        x = torch.sigmoid(self.linear(x))
        return x
    

def load_mnist_data():
    train_dataset = datasets.MNIST(
        root="./Lab3/Data", train=True, transform=transforms.ToTensor(), download=True
    )
    test_dataset = datasets.MNIST(
        root="./Lab3/Data", train=False, transform=transforms.ToTensor()
    )
    
    train_data = train_dataset.data[:10000].view(-1, 28 * 28).numpy()
    train_labels = train_dataset.targets[:10000].numpy()
    test_data = test_dataset.data.view(-1, 28 * 28).numpy()
    test_labels = test_dataset.targets.numpy()

    return train_data, train_labels, test_data, test_labels
 
def visualize_mse(mse_lr, mse_svm, mse_rf):
    epochs = range(1, len(mse_lr) + 1)

    plt.plot(epochs, mse_lr, label='LR', color='blue')
    plt.plot(epochs, [mse_svm] * len(mse_lr), label='SVM', color='red')
    plt.plot(epochs, [mse_rf] * len(mse_lr), label='RF', color='green')

    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend(loc="upper right")
    plt.title("MSE vs Epoch")
    plt.show()

class TaskA3_I:

    def linear_regression():
        train_dataset = datasets.MNIST(
            root="./Lab3/Data", train=True, transform=transforms.ToTensor(), download=True
        )
        test_dataset = datasets.MNIST(
            root="./Lab3/Data", train=False, transform=transforms.ToTensor()
        )
        
        epochs = 10
        batch_size = 32
        input_size = 28 * 28
        output_size = 10
        learning_rate = 0.001

        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True) 
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
        
        model = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.Sigmoid()
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(torch.cuda.is_available())

        # Initialize the model
        model = LinearRegression(input_size, output_size)

        #defining the optimizer and loss function
        optimizer = optim.Adam(model.parameters(), learning_rate)
        loss_fn = nn.MSELoss()

        mse_list = []

        for epoch in range(epochs):
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                # get output from the model, given the inputs
                outputs = model(images.view(-1, 28 * 28))

                labels_one_hot = nn.functional.one_hot(labels, num_classes=10).float()
                # get loss for the predicted output
                loss = loss_fn(outputs, labels_one_hot)
                loss.backward()
                optimizer.step()
            mse_list.append(loss.item())
            correct = 0
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images.view(-1, 28 * 28))
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum()
            accuracy = 100 * (correct.item()) / len(test_dataset)
            print("Epoch: {}. Loss: {}. Accuracy: {}".format(epoch, loss.item(), accuracy))
        return mse_list

    def svm_classification():
        train_data, train_labels, test_data, test_labels = load_mnist_data()

        clf = svm.SVC(kernel='linear')
        clf.fit(train_data, train_labels)

        prediction = clf.predict(test_data)
        mse = mean_squared_error(test_labels, prediction)

        print(f"Support Vector Accuracy (SVM): {accuracy_score(test_labels, prediction) * 100:.2f}%")

        return mse

    def random_forest_classification():
        train_data, train_labels, test_data, test_labels = load_mnist_data()

        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(train_data, train_labels)

        prediction = clf.predict(test_data)
        mse = mean_squared_error(test_labels, prediction)

        print(f"Random Forest Accuracy: {accuracy_score(test_labels, prediction) * 100:.2f}%")

        return mse

if __name__ == "__main__":

    visualize_mse(
        TaskA3_I.linear_regression(), 
        TaskA3_I.svm_classification(), 
        TaskA3_I.random_forest_classification() 
    )
