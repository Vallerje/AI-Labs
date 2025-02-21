from statistics import linear_regression
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
import seaborn as sns
from sklearn import svm

class LinearRegression(torch.nn.Module):    
    # build the constructor
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)

    # make predictions
    def forward(self, x):
        x = torch.sigmoid(self.linear(x))
        return x

def load_weather_data():

    df = pd.read_csv('Lab3/Data/seattle-weather.csv')
    df = df.drop(columns=["date"])

    features = df.drop(columns=["weather"])
    labels = df["weather"].replace({"drizzle": 0, "rain": 1, "sun": 2, "snow": 3, "fog": 4})
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    standardizer = StandardScaler()

    X_train = standardizer.fit_transform(X_train)
    X_test = standardizer.transform(X_test)
    
    return X_train, y_train, X_test, y_test

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

class TaskA3_II:

    def linear_regression():
        X_train, y_train, X_test, y_test = load_weather_data()

        epochs = 10000
        input_dim = X_train.shape[1]
        output_dim = 5
        learning_rate = 0.01

        model = LinearRegression(input_dim, output_dim)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        train_inputs = torch.tensor(X_train, dtype=torch.float32)
        train_labels = torch.tensor(y_train.values, dtype=torch.long)
        test_inputs = torch.tensor(X_test, dtype=torch.float32)
        test_labels = torch.tensor(y_test.values, dtype=torch.long)

        mse_list = []
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            predictions = model(train_inputs)
            one_hot_labels = nn.functional.one_hot(train_labels, num_classes=5).float()
            loss = criterion(predictions, one_hot_labels)
            loss.backward()
            optimizer.step()
            mse_list.append(loss.item())

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

            model.eval()
            with torch.no_grad():
                test_predictions = model(test_inputs)
                _, predicted_classes = torch.max(test_predictions.data, 1)
                conf_matrix = confusion_matrix(test_labels, predicted_classes)
                accuracy = (predicted_classes == test_labels).sum().item() / len(y_test)
                print(f"Epoch {epoch + 1}/{epochs}, Accuracy: {accuracy * 100:.2f}%")

        plt.figure(figsize=(12, 8))
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt="d",
            cmap="Reds",
            xticklabels=["drizzle", "rain", "sun", "snow", "fog"],
            yticklabels=["drizzle", "rain", "sun", "snow", "fog"],
        )
        plt.xlabel("Prediction")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix for Linear Regression")
        plt.show()

        return mse_list
    
    def svm_classification():
        X_train, y_train, X_test, y_test = load_weather_data()

        clf = svm.SVC(kernel='linear')
        clf.fit(X_train, y_train)

        predictions = clf.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        conf_matrix = confusion_matrix(y_test, predictions)

        print(f"SVM Accuracy: {accuracy * 100:.2f}%")


    def random_forest_classification():
        X_train, y_train, X_test, y_test = load_weather_data()

        clf = RandomForestClassifier(max_depth=10, n_estimators=100)
        clf.fit(X_train, y_train)

        predictions = clf.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        conf_matrix = confusion_matrix(y_test, predictions)

        print(f"Random Forest Accuracy: {accuracy * 100:.2f}%")



if __name__ == "__main__":

    visualize_mse(
        TaskA3_II.linear_regression(), 
        TaskA3_II.svm_classification(), 
        TaskA3_II.random_forest_classification() 
    )