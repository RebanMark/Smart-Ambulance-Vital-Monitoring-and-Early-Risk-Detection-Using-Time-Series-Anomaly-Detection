# Smart Ambulance AI – Project Plan

## Gray Mobility AI/ML Engineer Assignment

Domain: Smart Ambulance | Time-Series ML | Decision Support

This project builds an intelligent system that analyzes patient vitals streamed in real time from an ambulance environment and detects early medical risk signals.

The system will simulate physiological data, remove motion artifacts, detect anomalies, generate risk scores, and expose the system through an API service.

Final deployment will be provided as a demo application hosted on Hugging Face Spaces.

---

# AI Agent Instructions

## Expert AI Pair Programmer

You are my expert AI pair programmer.

You have the judgment, skill, and context awareness of a top senior software engineer at a leading tech company.

You always think critically about requirements, proactively identify ambiguities, and flag anything unclear.

You are obsessed with:

- code quality
- maintainability
- reproducibility
- real-world reliability

When possible, explain reasoning briefly but avoid unnecessary verbosity.

If you detect missing context or unclear requirements, ask precise questions before coding.

You operate as a true collaborator, not just an assistant.

---

# Project Scope and Intent

This project simulates the **Smart Ambulance monitoring system**.

Patient vitals stream every second and the system must detect:

- patient deterioration
- abnormal physiological patterns
- sensor artifacts caused by ambulance motion

The system should detect **early warning signals**, not just threshold breaches.

The final output will include:

- anomaly detection
- risk scoring
- confidence score
- API inference service

---

# Technology Stack

Language

Python 3.10+

Environment Manager

Use **uv** for dependency management and virtual environments.

Core Libraries

Time Series

- numpy
- pandas
- scipy

Machine Learning

- scikit-learn
- pyod

Visualization

- matplotlib
- seaborn

API

- fastapi
- uvicorn

Interface

- gradio

Deployment

Deploy demo on **Hugging Face Spaces**

---

# Environment Setup (Using uv)

Install uv

pip install uv


Initialize project


uv init smart-ambulance-ai
cd smart-ambulance-ai


Create virtual environment


uv venv


Activate environment

Mac/Linux


source .venv/bin/activate


Windows


.venv\Scripts\activate


Install dependencies


uv add numpy pandas scipy scikit-learn pyod matplotlib seaborn fastapi uvicorn gradio


Export requirements file


uv pip freeze > requirements.txt


---

# Project Architecture


smart-ambulance-ai
│
├── data
│ └── synthetic_vitals.csv
│
├── src
│ ├── data_generation
│ │ └── synthetic_vitals.py
│ │
│ ├── preprocessing
│ │ └── artifact_detection.py
│ │
│ ├── features
│ │ └── feature_engineering.py
│ │
│ ├── models
│ │ ├── anomaly_model.py
│ │ └── risk_scoring.py
│ │
│ ├── evaluation
│ │ └── metrics.py
│ │
│ └── api
│ └── main.py
│
├── notebooks
│ └── exploration.ipynb
│
├── train.py
├── inference.py
├── requirements.txt
├── README.md
└── report.md


---

# Phase 1 — Synthetic Data Generation

Generate realistic ambulance vital signals.

Dataset length

30 minutes per patient

Sampling rate

1 second

Total patients

10–20

Signals

- Heart Rate (HR)
- SpO2
- Blood Pressure (Systolic)
- Blood Pressure (Diastolic)
- Motion/Vibration signal

Simulated scenarios

Normal transport


HR: 70–90
SpO2: 96–100
BP: stable


Distress scenario


HR spikes
SpO2 drops
BP instability


Motion artifacts


SpO2 false drop
HR spikes due to bumps
missing segments
sensor noise


Dataset format


timestamp
patient_id
heart_rate
spo2
bp_sys
bp_dia
motion_signal
event_label


Save dataset


data/synthetic_vitals.csv


---

# Phase 2 — Artifact Detection

Implement explicit artifact handling before anomaly detection.

Artifacts include

- motion-induced SpO2 drops
- HR spikes caused by vehicle bumps
- missing sensor values

Techniques

Rolling median filtering

Z-score outlier removal

Motion correlation filtering

Visualizations required

- raw signals
- cleaned signals

---

# Phase 3 — Feature Engineering

Use sliding windows.

Window size


20 seconds


Features

Statistical features


mean
std
min
max


Trend features


slope
rate of change


Cross-signal features


HR-SpO2 correlation
motion influence


---

# Phase 4 — Anomaly Detection

Goal

Detect early deterioration signals.

Preferred model


Isolation Forest


Alternative models


Local Outlier Factor
Autoencoder


Model output


anomaly_probability


---

# Phase 5 — Risk Scoring System

Design a triage score combining multiple signals.

Example scoring logic


risk_score =
HR_abnormality +
SpO2_drop +
BP_instability +
model_anomaly_score


Normalize


0–100 scale


Confidence score

Based on signal quality and motion level.

---

# Phase 6 — Evaluation Metrics

Report

- Precision
- Recall
- False alert rate
- Alert latency

Visualizations

- Confusion matrix
- Alert timeline
- Risk score trend

---

# Phase 7 — Failure Analysis

Analyze at least 3 failure cases.

Examples

Motion artifact mistaken as anomaly

True deterioration missed

Sensor dropout causing false alerts

Explain

- why it happened
- how to fix it

---

# Phase 8 — API Service

Build a REST API using FastAPI.

Endpoint


POST /predict


Input


{
"heart_rate": value,
"spo2": value,
"bp_sys": value,
"bp_dia": value,
"motion_signal": value
}


Output


{
"anomaly_flag": true,
"risk_score": 72,
"confidence": 0.85
}


Run server


uvicorn src.api.main:app --reload


---

# Phase 9 — Deployment

Create a simple interface using Gradio.

Inputs

- HR
- SpO2
- BP
- Motion

Outputs

- risk score
- anomaly flag
- confidence

Deploy demo on **Hugging Face Spaces**.

---

# Report Requirements

Write a short report (max 2 pages) explaining

1. Most dangerous failure mode of the system
2. How to reduce false alerts without missing deterioration
3. What should never be fully automated in medical AI systems

---

# Bonus (Optional)

Choose one

Explainability


SHAP


Drift Detection


detect distribution shifts


Green Corridor ETA


predict ambulance arrival time


Dockerization


docker container for API


---

# Development Timeline

Day 1


Data generation


Day 2


Artifact detection


Day 3


Feature engineering


Day 4


Anomaly model


Day 5


Risk scoring + evaluation


Day 6


API service


Day 7


Deployment + report


---

# Expected Deliverables

GitHub repository containing

- modular code
- reproducible environment
- training script
- inference script
- API service
- README
- short report