#include <Servo.h>
#include <EEPROM.h>
#include <ArduinoJson.h>

Servo myServo1;
Servo myServo2;
int servoPin1 = 6;
int servoPin2 = 10;

int servo1_angle;
int servo2_angle;
int unlock_angle1 = 180;
int unlock_angle2 = 180;
int lockDelay = 7000;  // Default lock delay in milliseconds
int unlockDelay = 2000;  // Default unlock delay in milliseconds

bool configReceived = false;

void setup() {
  Serial.begin(9600);
  Serial.println("Arduino is ready");

  // Initialize servos
  myServo1.attach(servoPin1);
  myServo2.attach(servoPin2);
  Serial.println("Servos initialized");

  // Read last saved servo positions from EEPROM
  servo1_angle = EEPROM.read(0);
  servo2_angle = EEPROM.read(1);

  // Set servos to last known positions
  myServo1.write(servo1_angle);
  myServo2.write(servo2_angle);
  Serial.print("Set servo angles to last known positions: Servo1: ");
  Serial.print(servo1_angle);
  Serial.print(", Servo2: ");
  Serial.println(servo2_angle);

  // Wait for the configuration data from the Python script
  while (Serial.available() == 0) {}
  String configData = Serial.readString();
  Serial.print("Received configData: ");
  Serial.println(configData);
  deserializeConfig(configData);
  Serial.println("Configuration data received and parsed");

  configReceived = true;
}

void loop() {
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    
    Serial.print("Received command: ");
    Serial.println(command);
    
    if (command.startsWith("A")) {
      // Extract angles from the command string
      int commaIndex = command.indexOf(',');
      if (commaIndex != -1) {
        servo1_angle = command.substring(1, commaIndex).toInt();
        servo2_angle = command.substring(commaIndex + 1).toInt();

        // Move servos to specified angles
        myServo1.write(servo1_angle);
        myServo2.write(servo2_angle);
        Serial.println("OK");  // Respond with OK to indicate success
      } else {
        Serial.println("Invalid command format");
      }
    } else if (command == "O" && configReceived) {
      Serial.println("Executing unlock command");

      Serial.print("Unlock angles: ");
      Serial.print("Servo1: "); Serial.print(unlock_angle1);
      Serial.print(", Servo2: "); Serial.println(unlock_angle2);
      Serial.print("Delays: UnlockDelay: "); Serial.print(unlockDelay);
      Serial.print(", LockDelay: "); Serial.println(lockDelay);

      // Move servos to unlock position
      myServo1.write(unlock_angle1);
      myServo2.write(unlock_angle2);
      delay(lockDelay);  // Use unlock delay from configuration

      // Move servos to lock position
      myServo1.write(servo1_angle);  // Use locking position from configuration
      myServo2.write(servo2_angle);
      delay(unlockDelay);  // Use lock delay from configuration

      // Save the current positions to EEPROM
      EEPROM.write(0, servo1_angle);
      EEPROM.write(1, servo2_angle);
      Serial.println("Current servo positions saved to EEPROM");
    } else {
      Serial.print("Command received before configuration was applied or unknown command: ");
      Serial.println(command);
    }
  }
  delay(50);  
}

void resetServos() {
  myServo1.write(servo1_angle);  // Use locking position from configuration
  myServo2.write(servo2_angle);
  Serial.println("Reset servo angles to locking position");
}

void deserializeConfig(String json) {
  StaticJsonDocument<400> doc;  // Increase the size to 400 bytes
  DeserializationError error = deserializeJson(doc, json);
  
  if (error) {
    Serial.print("deserializeJson() failed: ");
    Serial.println(error.c_str());
    return;
  }

  servo1_angle = doc["servo1_angle"];
  servo2_angle = doc["servo2_angle"];
  unlock_angle1 = doc["unlock_angle1"];
  unlock_angle2 = doc["unlock_angle2"];
  lockDelay = doc["lock_delay"].as<int>() * 1000;  // Convert to milliseconds if needed
  unlockDelay = doc["unlock_delay"].as<int>() * 1000;  // Convert to milliseconds if needed

  Serial.println("Configuration data applied:");
  Serial.print("servo1_angle: "); Serial.println(servo1_angle);
  Serial.print("servo2_angle: "); Serial.println(servo2_angle);
  Serial.print("unlock_angle1: "); Serial.println(unlock_angle1);
  Serial.print("unlock_angle2: "); Serial.println(unlock_angle2);
  Serial.print("lock_delay: "); Serial.println(lockDelay);
  Serial.print("unlock_delay: "); Serial.println(unlockDelay);

  // Save the new positions to EEPROM
  EEPROM.write(0, servo1_angle);
  EEPROM.write(1, servo2_angle);
  Serial.println("New servo positions saved to EEPROM");
  
  // Explicitly reset servos after applying configuration
  resetServos();
}
