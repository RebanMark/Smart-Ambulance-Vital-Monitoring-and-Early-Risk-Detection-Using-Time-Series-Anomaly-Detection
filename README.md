# 🚑 Smart Ambulance AI Edge Monitor

This project is an intelligent **Time-Series Anomaly Detection** and **Decision Support System** built for paramedic environments. It analyzes streaming patient vitals (Heart rate, SpO2, Blood Pressure) in real-time, intelligently filters out ambulance motion artifacts, and uses a hybrid Machine Learning & Clinical Rules pipeline to proactively detect patient deterioration over 5-minute rolling windows.

---

## 🚀 Project Architecture
- `data/`: Synthetic dataset generation
- `src/`: Core logic
  - `preprocessing/`: Artifact filtering
  - `features/`: Windowed and Slope engineering
  - `models/`: Isolation Forest `anomaly_model` & `risk_scoring.py` 
  - `evaluation/`: Precision/Recall `metrics.py` and `failure_analysis.py`
- `app.py`: The Stateful **Gradio Web Interface** Dashboard (Hugging Face ready).

---

## 💻 Local Testing & Execution
Once cloned, testing the dashboard locally is simple.

1. Install dependencies (Using `uv`):
```bash
uv pip install -r requirements.txt
```

2. Boot the Dashboard:
```bash
python app.py
```
*Navigate to `http://127.0.0.1:7860` in your web browser!*

---

## 📦 Pushing to GitHub
To publish this completed project to your GitHub profile natively, open your terminal in this directory and run the following commands sequentially:

```bash
# 1. Initialize your project folder as a git repository
git init

# 2. Add all your beautifully formatted code and models
git add .

# 3. Create your first save point
git commit -m "Initial commit: Complete Smart Ambulance AI system with Phase 1-9 including elegant Gradio app."

# 4. Link it to your GitHub account (REPLACE the URL below with your actual blank Github repo URL!)
git remote add origin https://github.com/YourUsername/smart-ambulance-ai.git

# 5. Push it up! (Change 'main' to 'master' if your git branches differently)
git branch -M main
git push -u origin main
```

---

## 🤗 Deploying to Hugging Face Spaces (Complete Guide)

Deploying to Hugging Face is the final step (Phase 9) to host the dashboard permanently online. It's incredibly easy because we wrote `app.py` specifically for it!

1. Go to **[Hugging Face Spaces](https://huggingface.co/spaces)** and log in.
2. Click **Create new Space**.
3. **Space Name**: `smart-ambulance-ai` (or anything you like).
4. **License**: Choose `MIT` (Optional).
5. **Select the Space SDK**: Click "**Gradio**".
6. **Space Hardware**: Leave it on the Free CPU basic tier (our ML model is highly optimized).
7. **Create Space**.

**Pushing the code to the Space:**
Now you just need to upload these files to your Space. You can do this by dragging and dropping the files directly into the Hugging Face "Files and versions" tab on the website, OR use Git directly from your terminal:

```bash
# Add your new Hugging Face Space as a remote destination
git remote add hf https://huggingface.co/spaces/YourUsername/smart-ambulance-ai

# Push your code directly to the Hugging Face server!
git push hf main
```

Hugging Face will automatically read the `requirements.txt`, install everything inside a Docker container natively, and boot `app.py`. Within ~2 minutes, your dashboard will be live online 24/7!
