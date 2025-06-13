# KeyWatch
Keystroke Behaviour Monitoring &amp; Anomaly Detection Dashboard

This project simulates a local keylogging system that captures and analyzes typing patterns in real time. 
It detects anomalies using machine learning (Isolation Forest) based on `dwell time` and `flight time`.

## Features
- 🔍 Real-time anomaly detection (based on typing behavior)
- 📊 Interactive dashboard with:
  - Timeline Graph (key presses over time)
  - Dwell Time Anomaly Graph (normal vs. anomaly)
  - Filterable Event Logs Table
- 🎛 Sensitivity control (Low, Medium, High)
- 📅 Date range filtering
- 🔐 100% offline, local-only data capture
- 🧠 Smart retraining with threshold and hash-check

## ML Pipeline
1. **Data Collection:** Dwell and flight times logged using `pynput`
2. **Preprocessing:** Timestamp parsing, normalization using `MinMaxScaler`
3. **Model:** Isolation Forest (sklearn) with adjustable contamination rate
4. **Retraining:** Automatic every 10 mins or on sensitivity change
5. **Prediction:** Re-evaluated every time dashboard updates

## Logging Strategy
- Keystrokes stored in `logs/keystrokes.csv`
- App events/errors logged in `logs/debug.log`
- Robust error handling and duplicate-training checks implemented

## 💻 System Requirements

### Minimum
- Dual-core CPU (Intel i3 or equivalent)
- 4 GB RAM
- Python 3.9 or higher
- Disk: 500 MB free space
- OS: Windows 10, macOS, or Linux

### Recommended (For Best Demo Performance)
- Quad-core CPU (Intel i5 or better)
- 8 GB RAM
- SSD Storage
- Python 3.11+
- Chrome browser for dashboard

## How to Run

1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

2. Start the keylogger (captures typing data):
   ```bash
   python keylogger.py
   ```

3. In a new terminal, launch the dashboard:
   ```bash
   python dashboard.py
   ```

4. Open in browser: http://127.0.0.1:8050

## Disclaimer
This project is for educational purposes only. It does **not** store sensitive data and runs **entirely offline**.

## Author
Samarth Desai — University of Hertfordshire | BSc (Hons) Cyber Security & Networking (Final Year)
