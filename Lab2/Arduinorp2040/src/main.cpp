#include <Arduino.h>
#include <Arduino_LSM6DSOX.h>

#define Temperature 0
#define Acc_Gyr 1

unsigned baud_rate = 115200;

#if Temperature
  float temperature;

  void setup() {
    Serial.begin(baud_rate);
    while (!Serial); 

    if (!IMU.begin()) {
        Serial.println("Failed to initialize IMU!");
        while (1);
    }

    Serial.println("Entry, Temperature(C)");
  }
  
  void loop() {
    
    if (IMU.temperatureAvailable()) {
      temperature = 0;
      
      IMU.readTemperatureFloat(temperature);
      
      Serial.print(millis()); // Timestamp in ms
      Serial.print(", ");
      Serial.println(temperature); // Temperature in celsius
    }
    
    delay(1000); // Log new value every second
  }
#endif

#if Acc_Gyr
#include <TimeLib.h> 

  time_t startTime;
  unsigned long startMillis;

  void setup() {
      Serial.begin(baud_rate);
      while (!Serial); 

      if (!IMU.begin()) {
          Serial.println("Failed to initialize IMU!");
          while (1);
      }
      Serial.println("Date,Time,Ax,Ay,Az,Gx,Gy,Gz");
      startMillis = millis();
      startTime = now(); // Capturing the start time 
  }

  void loop() {
      float ax, ay, az;
      float gx, gy, gz;
      if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
          IMU.readAcceleration(ax, ay, az);
          IMU.readGyroscope(gx, gy, gz);

          unsigned long currentMillis = millis();
          time_t currentTime = startTime + (currentMillis - startMillis) / 1000;
          char timeBuffer[9];
          
          snprintf(timeBuffer, sizeof(timeBuffer), "%02d:%02d:%02d", hour(currentTime), minute(currentTime), second(currentTime));

          Serial.print("2025-02-11"); // Print the date and time
          Serial.print(",");
          Serial.print(timeBuffer);
          Serial.print(",");
          Serial.print(ax); // Acceleration in X
          Serial.print(",");
          Serial.print(ay); // Acceleration in Y
          Serial.print(",");
          Serial.print(az); // Acceleration in Z
          Serial.print(",");
          Serial.print(gx); // Gyroscope in X
          Serial.print(",");
          Serial.print(gy); // Gyroscope in Y
          Serial.print(",");
          Serial.println(gz); // Gyroscope in Z
      }
      delay(500); // Log new values every 0.5 second
  }

#endif