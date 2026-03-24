---
title: Smart Ambulance AI edge Monitor
emoji: 🚑
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: "4.26.0"
app_file: app.py
pinned: false
---
# 🚑 Smart Ambulance AI Edge Monitor

This project is an intelligent **Time-Series Anomaly Detection** and **Decision Support System** built for paramedic environments. It analyzes streaming patient vitals (Heart rate, SpO2, Blood Pressure) in real-time, intelligently filters out ambulance motion artifacts, and uses a hybrid Machine Learning & Clinical Rules pipeline to proactively detect patient deterioration over 5-minute rolling windows.

---

## 🚀 Project Overview & Core Features

The project encompasses the entire machine learning lifecycle, from data generation to deployment, and provides the following core capabilities:

- **Synthetic Data Generation**: Simulates 30-minute patient vital streams (HR, SpO2, BP) including normal transport, distress scenarios, and motion-induced artifacts.
- **Artifact Detection**: Implements signal processing techniques (rolling median, Z-score outlier removal) to filter out motion artifacts and sensor noise before anomaly detection.
- **Feature Engineering**: Uses sliding windows to extract statistical features, trend/slope features, and cross-signal correlations (e.g., HR-SpO2 correlation).
- **Anomaly Detection**: Utilizes an Isolation Forest model to detect early deterioration signals and calculate anomaly probabilities.
- **Risk Scoring System**: A hybrid triage scoring system (0-100 scale) combining physiological rule breaches (HR, SpO2, BP) with the ML anomaly score, along with a confidence metric based on signal quality.
- **Evaluation Metrics**: Generates precision, recall, false alert rates, and alert latency metrics, along with confusion matrices and risk score trends.
- **Failure Analysis**: Analyzes edge cases such as motion artifacts mistaken for anomalies, missing true deteriorations, and sensor dropouts.
- **API Service**: A FastAPI-powered REST service exposing a `/predict` endpoint for real-time inference on vital streams.
- **Deployment**: A stateful, interactive Gradio web dashboard, natively deployable to Hugging Face Spaces.

---

## 📂 Project Architecture

```text
smart-ambulance-ai/
│
├── data/                  # Synthetic dataset generation & storage
├── src/                   # Core logic
│   ├── data_generation/   # Synthetic vitals generator
│   ├── preprocessing/     # Artifact filtering & signal cleaning
│   ├── features/          # Windowed and slope feature engineering
│   ├── models/            # Isolation Forest anomaly_model & risk_scoring
│   ├── evaluation/        # Precision/Recall metrics & failure analysis
│   └── api/               # FastAPI endpoints
├── app.py                 # The Stateful Gradio Web Interface Dashboard
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

---

## 💻 Local Testing & Execution

This project uses `uv` for lightning-fast dependency management.

1. **Install dependencies**:
```bash
uv pip install -r requirements.txt
```

2. **Boot the Dashboard**:
```bash
python app.py
```
*Navigate to `http://127.0.0.1:7860` in your web browser!*

3. **Run the API Service Locally**:
```bash
uvicorn src.api.main:app --reload
```
*Access the API documentation at `http://127.0.0.1:8000/docs`*

---

## 📡 API Usage Example

You can send a `POST` request to the `/predict` endpoint to get real-time anomaly flags and risk scores:

**Request:**
```json
{
  "heart_rate": 115,
  "spo2": 92,
  "bp_sys": 140,
  "bp_dia": 90,
  "motion_signal": 0.2
}
```

**Response:**
```json
{
  "anomaly_flag": true,
  "risk_score": 85,
  "confidence": 0.92
}
```

---

## 🤗 Deploying to Hugging Face Spaces

Deploying to Hugging Face is incredibly easy because `app.py` is written specifically for it!

1. Go to **[Hugging Face Spaces](https://huggingface.co/spaces)** and log in.
2. Click **Create new Space**.
3. **Space Name**: `smart-ambulance-ai` (or anything you like).
4. **License**: Choose `MIT` (Optional).
5. **Select the Space SDK**: Click "**Gradio**".
6. **Space Hardware**: Leave it on the Free CPU basic tier (our ML model is highly optimized).
7. **Create Space**.

**Pushing the code to the Space:**
You can upload the files directly into the Hugging Face "Files and versions" tab on the website, OR use Git directly from your terminal:

```bash
# Add your new Hugging Face Space as a remote destination
git remote add hf https://huggingface.co/spaces/YourUsername/smart-ambulance-ai

# Push your code directly to the Hugging Face server!
git push hf main
```

Hugging Face will automatically read the `requirements.txt`, install everything inside a Docker container natively, and boot `app.py`. Within ~2 minutes, your dashboard will be live online 24/7!
