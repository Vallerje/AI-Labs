import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import seaborn as sns
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

class TaskA3_4:
    def k_means():
        # Load and preprocess the data
        data = pd.read_csv("Lab3/Data/penguins.csv", usecols=["species", "bill_length_mm", "bill_depth_mm"])
        data.dropna(inplace=True)
        features = data[["bill_length_mm", "bill_depth_mm"]]

        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)

        # Determine the optimal number of clusters using the elbow method
        inertia_values = []
        for n_clusters in range(1, 11):
            kmeans = KMeans(n_clusters=n_clusters)
            kmeans.fit(scaled_features)
            inertia_values.append(kmeans.inertia_)

        plt.plot(range(1, 11), inertia_values, marker='o')
        plt.title("Elbow Method for Optimal Clusters")
        plt.xlabel("Number of Clusters")
        plt.ylabel("Inertia")
        plt.show()

        # Perform K-Means clustering with the optimal number of clusters
        optimal_kmeans = KMeans(n_clusters=3, init="k-means++", n_init=20, random_state=42)
        data["cluster"] = optimal_kmeans.fit_predict(scaled_features)

        # Map clusters to species names
        cluster_to_species_map = {0: "Adelie", 2: "Chinstrap", 1: "Gentoo"}
        data["species_cluster"] = data["cluster"].map(cluster_to_species_map)

        # Visualize the clusters
        colors = ["blue", "orange", "green"]
        for species, color in zip(cluster_to_species_map.values(), colors):
            subset = data[data["species_cluster"] == species]
            plt.scatter(subset["bill_length_mm"], subset["bill_depth_mm"], label=species, color=color)

        plt.xlabel("Bill Length (mm)")
        plt.ylabel("Bill Depth (mm)")
        plt.title("K-Means Clustering of Penguins")
        plt.legend(loc="upper right")
        plt.show()

        # Visualize the clusters with centroids
        centroids = scaler.inverse_transform(optimal_kmeans.cluster_centers_)
        for species, color in zip(cluster_to_species_map.values(), colors):
            subset = data[data["species_cluster"] == species]
            plt.scatter(subset["bill_length_mm"], subset["bill_depth_mm"], label=species, color=color)
        plt.scatter(centroids[:, 0], centroids[:, 1], c="black", marker="X", label="Centroids")
        plt.xlabel("Bill Length (mm)")
        plt.ylabel("Bill Depth (mm)")
        plt.title("K-Means Clustering of Penguins with Centroids")
        plt.legend(loc="upper right")
        plt.show()

        # Evaluate the clustering accuracy
        correct_predictions = sum(data["species"] == data["species_cluster"])
        accuracy = correct_predictions / len(data)
        print(f"K-Means Clustering Accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
       TaskA3_4.k_means()