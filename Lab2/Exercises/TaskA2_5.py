import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def TaskA2_5():
    df = pd.read_csv("Lab2/Data/Climate2016.csv")

    df["windveloX"] = df["windvelo (m/s)"] * np.cos(np.deg2rad(df["winddeg (deg)"]))
    df["windveloY"] = df["windvelo (m/s)"] * np.sin(np.deg2rad(df["winddeg (deg)"]))

    df["norm_windveloX"] = (df["windveloX"] - df["windveloX"].min()) / (df["windveloX"].max() - df["windveloX"].min())
    df["norm_windveloY"] = (df["windveloY"] - df["windveloY"].min()) / (df["windveloY"].max() - df["windveloY"].min())

    df.to_csv("Lab2/Data/Climate2016_normalized.csv", index=False)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist2d(df["winddeg (deg)"], df["windvelo (m/s)"], bins=50, vmax=400)
    plt.colorbar(label="Frequency")
    plt.xlabel("Wind Direction [deg]")
    plt.ylabel("Wind Velocity [m/s]")

    plt.subplot(1, 2, 2)
    plt.hist2d(df["norm_windveloX"], df["norm_windveloY"], bins=50, vmax=400)
    plt.colorbar(label="Frequency")
    plt.xlabel("Normalized Wind X")
    plt.ylabel("Normalized Wind Y")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    TaskA2_5()

pass