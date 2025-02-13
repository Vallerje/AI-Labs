import pandas as pd
import matplotlib.pyplot as plt

#Read CSV file
df = pd.read_csv('Lab2/Data/acceleration_data')

#Plot data from CSV file
plt.figure(figsize=(10, 5))
plt.plot(df["Timesteps"], df["Temperature"], color="orange", label= "Temperature")

plt.xlabel("Time (seconds)")
plt.ylabel("Temperature (degrees Celsius)")
plt.title("Temperature over time")
plt.legend(loc="upper right")

plt.grid(True)

plt.show()