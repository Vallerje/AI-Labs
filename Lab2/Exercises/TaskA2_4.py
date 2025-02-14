import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def TaskA2_4_I():
    df = pd.read_csv('Lab2/Data/aw_fb_data.csv')

    df['log'] = np.log(df["calories"])
    df['log'].hist()

    plt.show()

#def TaskA2_4_II():


#TaskA2_4_III
def pad_and_plot(ax, datasets, max_length, labels, colors, title, ylabel):
    padded_datasets = [
        np.pad(data, (0, max_length - len(data)), mode='constant', constant_values=np.nan) for data in datasets
    ]
    ax.stackplot(np.arange(max_length), padded_datasets, labels=labels, colors=colors)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper right")
    ax.grid(True)


def TaskA2_4_III():
    # Load the data
    df = pd.read_csv("Lab2/Data/aw_fb_data.csv")
    df_participants = pd.read_csv("Lab2/Data/aw_fb_data_participants.csv")

    df_participants["id"] = np.arange(1, len(df_participants) + 1)
    unique_columns = ["age", "height", "weight", "gender"]
    df_combined = df.merge(df_participants[unique_columns + ["id"]], on=unique_columns, how="left")

    ids = [6, 28, 36]
    participants = [df_combined[df_combined["id"] == pid] for pid in ids]

    # Determine the maximum length of data among participants
    max_length = max(len(p) for p in participants)
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 7))

    # Plot steps
    pad_and_plot(
        axes[0],
        [p["steps"] for p in participants],
        max_length,
        labels=[f"Participant #{pid}" for pid in ids],
        colors=["blue", "red", "green"],
        title="Steps of selected participants",
        ylabel="Steps",
    )

    # Plot heart rate
    pad_and_plot(
        axes[1],
        [p["hear_rate"] for p in participants],
        max_length,
        labels=[f"Participant #{pid}" for pid in ids],
        colors=["blue", "red", "green"],
        title="Heart rate of selected participants",
        ylabel="Heart rate",
    )

    # Plot calories
    pad_and_plot(
        axes[2],
        [p["calories"] for p in participants],
        max_length,
        labels=[f"Participant #{pid}" for pid in ids],
        colors=["blue", "red", "green"],
        title="Calories of selected participants",
        ylabel="Calories",
    )

    plt.tight_layout()
    plt.show()

#TaskA2_4_IV
def normalize_column(dataframe, col_name):
    col_min = dataframe[col_name].min()
    col_max = dataframe[col_name].max()
    dataframe[col_name] = (dataframe[col_name] - col_min) / (col_max - col_min)
    return dataframe

def standardize_column(dataframe, col_name):
    col_mean = dataframe[col_name].mean()
    col_std = dataframe[col_name].std()
    dataframe[f"{col_name}_standardized"] = (dataframe[col_name] - col_mean) / col_std
    return dataframe

def TaskA2_4_IV():
    df = pd.read_csv("Lab2/Data/aw_fb_data.csv")

    df = normalize_column(df, "age")
    df = normalize_column(df, "height")
    df = normalize_column(df, "weight")

    df = standardize_column(df, "steps")
    df = standardize_column(df, "hear_rate")

    df.to_csv("Lab2/Data/aw_fb_data_normalized_standardized.csv", index=False)


#TaskA2_4_V
def TaskA2_4_V():
    df = pd.read_csv("Lab2/Data/aw_fb_data.csv")

    train_data, temp_data = train_test_split(df, test_size=0.3, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    print(f"Training set size: {len(train_data)}")
    print(f"Validation set size: {len(val_data)}")
    print(f"Test set size: {len(test_data)}")

if __name__ == "__main__":
    TaskA2_4_I()

    #TaskA2_4_III()
    #TaskA2_4_IV()
    #TaskA2_4_V()
pass