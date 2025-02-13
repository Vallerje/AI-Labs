import pandas as pd
import matplotlib.pyplot as plt


def TaskA2_2_II():
    #Read CSV file
    df = pd.read_csv('Lab2/Data/acceleration_data.csv')

    #Plot data from CSV file
    plt.figure(figsize=(10, 5))
    plt.plot(df["Time"], df["Ax"], color="yellow", label="Ax")
    plt.plot(df["Time"], df["Ay"], color="orange", label="Ay")
    plt.plot(df["Time"], df["Az"], color="purple", label="Az")
    plt.plot(df["Time"], df["Gx"], color="red", label="Gx")
    plt.plot(df["Time"], df["Gy"], color="blue", label="Gy")
    plt.plot(df["Time"], df["Gz"], color="green", label="Gz")

    plt.xlabel("Time(seconds)")
    plt.ylabel("Acceleration (m.sq/s.sq)")
    plt.title("Acceleration and Gyroscope over time")
    plt.legend(loc="upper right")

    plt.grid(True)

    plt.show()


def TaskA2_2_III():
    #Read CSV file
    df = pd.read_csv('Lab2/Data/acceleration_data.csv')

    # Remove rows where the acceleration values are close to 0
    df = df[(df["Ax"].abs() > 0.03) & (df["Ay"].abs() > 0.02) & (df["Az"].abs() > 1)]

    # Plot the data
    plt.figure(figsize=(10, 5))
    plt.plot(df["Time"], df["Ax"], color="yellow", label="Ax")
    plt.plot(df["Time"], df["Ay"], color="orange", label="Ay")
    plt.plot(df["Time"], df["Az"], color="purple", label="Az")
    plt.plot(df["Time"], df["Gx"], color="red", label="Gx")
    plt.plot(df["Time"], df["Gy"], color="blue", label="Gy")
    plt.plot(df["Time"], df["Gz"], color="green", label="Gz")

    plt.xlabel("Time(seconds)")
    plt.ylabel("Acceleration (m.sq/s.sq)")
    plt.legend(loc="upper right")
    plt.title("Acceleration and Gyroscope over time")

    plt.grid(True)

    plt.show()


if __name__ == "__main__":

     #TaskA2_2_II()
     TaskA2_2_III()

pass
