/*
 * Drowsiness Detection Alert System
 * 
 * This sketch receives signals from the Python drowsiness detection program
 * and triggers appropriate alerts (buzzer, LEDs) based on the detected state.
 * 
 * Signal codes:
 * '0' - Normal state (no alerts)
 * '1' - Drowsiness detected
 * '2' - Yawning detected
 * '3' - Head down detected
 * '4' - Eyes closed detected
 */

// Pin definitions
const int BUZZER_PIN = 9;     // Buzzer connected to digital pin 9
const int RED_LED_PIN = 6;    // Red LED for alerts
const int YELLOW_LED_PIN = 5; // Yellow LED for warnings
const int GREEN_LED_PIN = 4;  // Green LED for normal state

// Variables
char incomingByte = '0';      // For incoming serial data
unsigned long lastAlertTime = 0;
const unsigned long BLINK_INTERVAL = 300; // LED blink interval in ms
bool ledState = false;

void setup() {
  // Initialize serial communication
  Serial.begin(9600);
  
  // Initialize output pins
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(RED_LED_PIN, OUTPUT);
  pinMode(YELLOW_LED_PIN, OUTPUT);
  pinMode(GREEN_LED_PIN, OUTPUT);
  
  // Initial state - green LED on
  digitalWrite(GREEN_LED_PIN, HIGH);
  digitalWrite(YELLOW_LED_PIN, LOW);
  digitalWrite(RED_LED_PIN, LOW);
  
  // Startup sequence to confirm hardware is working
  startupSequence();
  
  Serial.println("Drowsiness Alert System Ready");
}

void loop() {
  // Check if data is available to read
  if (Serial.available() > 0) {
    // Read the incoming byte
    incomingByte = Serial.read();
    
    // Process the command
    processCommand(incomingByte);
  }
  
  // Handle alert patterns based on current state
  updateAlerts();
}

void processCommand(char command) {
  // Reset all outputs
  digitalWrite(GREEN_LED_PIN, LOW);
  digitalWrite(YELLOW_LED_PIN, LOW);
  digitalWrite(RED_LED_PIN, LOW);
  noTone(BUZZER_PIN);
  
  // Process based on command
  switch (command) {
    case '0': // Normal state
      Serial.println("Normal state");
      digitalWrite(GREEN_LED_PIN, HIGH);
      break;
      
    case '1': // Drowsiness detected - highest priority
      Serial.println("ALERT: Drowsiness detected!");
      digitalWrite(RED_LED_PIN, HIGH);
      // Buzzer will be handled in updateAlerts()
      break;
      
    case '2': // Yawning detected
      Serial.println("Warning: Yawning detected");
      digitalWrite(YELLOW_LED_PIN, HIGH);
      break;
      
    case '3': // Head down detected
      Serial.println("ALERT: Head down detected!");
      digitalWrite(RED_LED_PIN, HIGH);
      // Buzzer will be handled in updateAlerts()
      break;
      
    case '4': // Eyes closed detected
      Serial.println("ALERT: Eyes closed detected!");
      digitalWrite(RED_LED_PIN, HIGH);
      // Buzzer will be handled in updateAlerts()
      break;
      
    default:
      // Unknown command, default to normal state
      digitalWrite(GREEN_LED_PIN, HIGH);
      break;
  }
}

void updateAlerts() {
  // Handle blinking and sound patterns for alerts
  if (incomingByte == '1' || incomingByte == '3' || incomingByte == '4') {
    // Critical alerts - blinking red LED and beeping
    unsigned long currentTime = millis();
    
    if (currentTime - lastAlertTime >= BLINK_INTERVAL) {
      lastAlertTime = currentTime;
      ledState = !ledState;
      
      digitalWrite(RED_LED_PIN, ledState);
      
      if (ledState) {
        // Different tones for different alerts
        if (incomingByte == '1') {
          tone(BUZZER_PIN, 2000); // High pitch for drowsiness
        } else if (incomingByte == '3') {
          tone(BUZZER_PIN, 1500); // Medium pitch for head down
        } else if (incomingByte == '4') {
          tone(BUZZER_PIN, 1800); // Different pitch for eyes closed
        }
      } else {
        noTone(BUZZER_PIN);
      }
    }
  } 
  else if (incomingByte == '2') {
    // Warning - yellow LED blinks slower, no sound
    unsigned long currentTime = millis();
    
    if (currentTime - lastAlertTime >= BLINK_INTERVAL * 2) {
      lastAlertTime = currentTime;
      ledState = !ledState;
      
      digitalWrite(YELLOW_LED_PIN, ledState);
    }
  }
  // Normal state ('0') has steady green LED, no action needed here
}

void startupSequence() {
  // Play a startup sequence to verify hardware
  
  // Blink all LEDs
  for (int i = 0; i < 2; i++) {
    digitalWrite(RED_LED_PIN, HIGH);
    digitalWrite(YELLOW_LED_PIN, HIGH);
    digitalWrite(GREEN_LED_PIN, HIGH);
    delay(200);
    digitalWrite(RED_LED_PIN, LOW);
    digitalWrite(YELLOW_LED_PIN, LOW);
    digitalWrite(GREEN_LED_PIN, LOW);
    delay(200);
  }
  
  // Play ascending tones
  tone(BUZZER_PIN, 1000);
  delay(100);
  tone(BUZZER_PIN, 1500);
  delay(100);
  tone(BUZZER_PIN, 2000);
  delay(100);
  noTone(BUZZER_PIN);
  
  // Turn on green LED to indicate ready state
  digitalWrite(GREEN_LED_PIN, HIGH);
} 