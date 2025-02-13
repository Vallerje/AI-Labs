import serial
import time
import csv
import os


os.environ["TK_SILENCE_DEPRECATION"] = "1"

serial_port = '/dev/cu.usbmodem1101'
baud_rate = 115200
csv_file_temperature = "Lab2/Data/raw_temperature_data.csv"
csv_file_acceleration = "Lab2/Data/raw_acceleration_data.csv"

#Helper Functions

def setup_serial_connection():
    # Connect to the Arduino device
    ser = serial.Serial(serial_port, baudrate=115200, timeout=1)
    ser.flushInput()
    return ser

def process_line(raw_line):
    #Split data line into values
    try:
        return raw_line.split(',')
    except:
        print("Error: couldn't split that line")
        return None

def save_to_csv(data):
    #Save our data to the CSV file
    with open(csv_file_acceleration, 'a', newline='') as f:  #Either csv_file_temperature or csv_file_acceleration
        writer = csv.writer(f)
        writer.writerow(data)


#Main
ser = setup_serial_connection()

print("Starting to read data...")
try:
    while True:
        # Read from serial
        raw_bytes = ser.readline()
        clean_line = raw_bytes.decode('utf-8').strip()
        
        if clean_line:  # Only process if we got real data
            print(f"Raw: {clean_line}")
            
            # Split into values
            values = process_line(clean_line)
            
        if values:  # Only save if splitting worked
                print(f"Values: {values}")
                save_to_csv(values)
                

except KeyboardInterrupt:
    print("\nStopped by user")
except Exception as e:
    print(f"Unexpected error occured: {e}")