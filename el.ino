#include <WiFi.h>
#include <HTTPClient.h>
#include <DHT.h>
#include <ArduinoJson.h>

// WiFi credentials
const char* ssid = "Avi's Nord 5G";
const char* password = "Avi@9322564784";

// Flask server URLs
const String dataURL = "http://10.249.165.189:5000/esp_data";
const String commandURL = "http://10.249.165.189:5000/esp_command";

// Pin configuration (adjust if needed)
#define DHTPIN 15            // GPIO15
#define DHTTYPE DHT11
#define SOIL_MOISTURE_PIN 34 // GPIO34 (ADC1)
#define RAIN_PIN 35          // GPIO35 (ADC1)
#define RELAY_PIN 14         // GPIO14 for Relay (LOW = ON)

DHT dht(DHTPIN, DHTTYPE);

void setup() {
  Serial.begin(115200);
  dht.begin();

  pinMode(RELAY_PIN, OUTPUT);
  digitalWrite(RELAY_PIN, HIGH);  // Initial: Pump OFF (HIGH = OFF)

  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\n✅ Connected to WiFi!");
}

void loop() {
  // Sensor Readings
  float temp = dht.readTemperature();
  float hum = dht.readHumidity();
  int rawSoil = analogRead(SOIL_MOISTURE_PIN); // 0–4095
  int rawRain = analogRead(RAIN_PIN);          // 0–4095

  int soil = map(rawSoil, 4095, 0, 0, 100);     // Dry = 0%, Wet = 100%
  int rain = map(rawRain, 4095, 0, 0, 100);     // No rain = 0%, Heavy rain = 100%

  if (isnan(temp) || isnan(hum)) {
    Serial.println("⚠️ DHT11 failed to read.");
    delay(1000);
    return;
  }

  Serial.println("---- Sensor Readings ----");
  Serial.printf("🌡️ Temp     : %.1f °C\n", temp);
  Serial.printf("💧 Humidity : %.1f %%\n", hum);
  Serial.printf("🌱 Soil     : %d %%\n", soil);
  Serial.printf("☔ Rain     : %d %%\n", rain);

  if (WiFi.status() == WL_CONNECTED) {
    WiFiClient client;
    HTTPClient http;

    // 📤 Send sensor data
    http.begin(client, dataURL);
    http.addHeader("Content-Type", "application/json");

    String payload = "{\"temperature\":" + String(temp) +
                     ",\"humidity\":" + String(hum) +
                     ",\"soil_moisture\":" + String(soil) +
                     ",\"rain\":" + String(rain) + "}";

    int httpCode = http.POST(payload);
    if (httpCode > 0) {
      Serial.print("✅ POST → ");
      Serial.println(http.getString());
    } else {
      Serial.print("❌ POST failed: ");
      Serial.println(httpCode);
    }
    http.end();
    delay(200);

    // 📥 Get pump command from server
    http.begin(client, commandURL);
    int getCode = http.GET();
    if (getCode > 0) {
      String response = http.getString();
      Serial.print("🟢 Server Response: ");
      Serial.println(response);

      DynamicJsonDocument doc(256);
      DeserializationError err = deserializeJson(doc, response);

      if (!err) {
        String pump = doc["pump"];
        if (pump == "ON") {
          digitalWrite(RELAY_PIN, HIGH);  // Relay ON
          Serial.println("✅ Pump ON (LOW)");
        } else {
          digitalWrite(RELAY_PIN, LOW); // Relay OFF
          Serial.println("⛔ Pump OFF (HIGH)");
        }
      } else {
        Serial.print("❌ JSON Error: ");
        Serial.println(err.c_str());
      }
    } else {
      Serial.print("❌ GET failed: ");
      Serial.println(getCode);
    }
    http.end();
  } else {
    Serial.println("❌ WiFi not connected");
  }

  delay(3000);  // Loop every 3 seconds
}
