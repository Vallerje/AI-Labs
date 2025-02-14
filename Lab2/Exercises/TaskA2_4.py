import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt


def TaskA2_4_I():
    df = pd.read_csv('Lab2/Data/aw_fb_data.csv')

    df['log'] = np.log(df["calories"])
    df['log'].hist()

    plt.show()

#def TaskA2_4_II():


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



if __name__ == "__main__":
    #TaskA2_4_I()
    TaskA2_4_III()
pass