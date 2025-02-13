import pandas as pd    
import matplotlib.pyplot as plt


def TaskA2_3_I():

    df = pd.read_csv("Lab2/Data/IOT-temp.csv")

    df["noted_date"] = pd.to_datetime(df["noted_date"], dayfirst=True)

    df = df[
        (df["noted_date"] >= pd.to_datetime("2018-12-02"))
        & (df["noted_date"] <= pd.to_datetime("2018-12-08"))
    ]

    plt.figure(figsize=(10, 5))
    plt.plot(
        df[df["out/in"] == "Out"]["noted_date"], #X-axis
        df[df["out/in"] == "Out"]["temp"], #Y-axis
        color="red",
        label="Outdoor temperature",
    )

    plt.plot(
        df[df["out/in"] == "In"]["noted_date"], #X-axis
        df[df["out/in"] == "In"]["temp"], #Y-axis
        color="blue",
        label="Indoor temperature",
    )

    plt.xlabel("Date")
    plt.ylabel("Temperature")
    plt.title("Indoor and outdoor temperature")
    plt.legend(loc="upper right")

    plt.grid(True)

    plt.show()

def TaskA2_3_II(): 

    df = pd.read_csv("Lab2/Data/IOT-temp.csv")

    df["out/in"] = df["out/in"].apply(lambda x: 1 if x == "Out" else 0)

    df["noted_date"] = pd.to_datetime(df["noted_date"], dayfirst=True)
    df["date"] = df["noted_date"].dt.date
    df["time"] = df["noted_date"].dt.time

    df = df[df["date"] == pd.to_datetime("2018-12-08").date()]
    df = df.drop("noted_date", axis=1)

    df.to_csv("Lab2/Data/IOT-temp-modified.csv", index=False)


if __name__ == "__main__":

     TaskA2_3_I()
     #TaskA2_3_II()

pass