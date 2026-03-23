import gradio as gr
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

from src.preprocessing.artifact_detection import ArtifactRemover
from src.features.feature_engineering import FeatureEngineer
from src.models.anomaly_model import VitalAnomalyDetector
from src.models.risk_scoring import RiskScorer

# Load singletons
cleaner = ArtifactRemover()
engineer = FeatureEngineer(window_size="20s")
try:
    detector = VitalAnomalyDetector.load_model("models/vital_anomaly_model.pkl")
    print("Model loaded successfully.")
except Exception as e:
    detector = None
    print(f"Warning: Model not found. Did you train it? Error: {e}")
scorer = RiskScorer()

CUSTOM_CSS = """
body { background-color: #0f172a; font-family: 'Inter', sans-serif; }
.glass-panel {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border-radius: 15px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    padding: 20px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    color: white;
}
.score-circle {
    width: 150px;
    height: 150px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 3rem;
    font-weight: bold;
    margin: 0 auto;
    transition: all 0.5s ease;
    box-shadow: 0 0 20px rgba(0,0,0,0.5);
}
.score-green { background: linear-gradient(135deg, #10b981, #059669); text-shadow: 0 0 10px #fff; box-shadow: 0 0 30px #10b981; }
.score-yellow { background: linear-gradient(135deg, #fbbf24, #d97706); text-shadow: 0 0 10px #fff; box-shadow: 0 0 30px #fbbf24; }
.score-red { background: linear-gradient(135deg, #ef4444, #b91c1c); text-shadow: 0 0 10px #fff; box-shadow: 0 0 30px #ef4444; animation: pulse 1.5s infinite; }

@keyframes pulse {
    0% { transform: scale(1); box-shadow: 0 0 20px #ef4444; }
    50% { transform: scale(1.05); box-shadow: 0 0 40px #ef4444; }
    100% { transform: scale(1); box-shadow: 0 0 20px #ef4444; }
}
.alert-box {
    text-align: center; margin-top: 20px; padding: 15px; border-radius: 10px; font-size: 1.5rem; font-weight: bold; letter-spacing: 2px;
}
.alert-safe { background: rgba(16, 185, 129, 0.2); color: #10b981; border: 1px solid #10b981; }
.alert-danger { background: rgba(239, 68, 68, 0.2); color: #ef4444; border: 1px solid #ef4444; animation: blink 1s infinite; }
@keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.6; } }
"""


def generate_dashboard_html(risk, alert):
    color_class = "score-green"
    if risk >= 65 or alert:
        color_class = "score-red"
    elif risk >= 40:
        color_class = "score-yellow"

    alert_class = "alert-danger" if alert else "alert-safe"
    alert_text = "CRITICAL DETERIORATION DETECTED" if alert else "PATIENT STABLE"

    html = f"""
    <div class="glass-panel">
        <h2 style="text-align:center; color:#94a3b8; font-size: 1.2rem; text-transform:uppercase; letter-spacing: 2px; margin-bottom:20px;">Live Triage Risk Score</h2>
        <div class="score-circle {color_class}">
            {int(risk)}
        </div>
        <div class="alert-box {alert_class}">
            {alert_text}
        </div>
    </div>
    """
    return html


def pad_history(history):
    # We need backwards padding so slopes can calculate immediately without waiting 5 mins.
    # Take the first entry, and duplicate it backwards 310 times, assuming an interval of -1s.
    if not history:
        return []
    base = history[0]
    padded = []
    base_time = datetime.now() - timedelta(seconds=len(history) + 310)
    for i in range(310):
        t = base_time + timedelta(seconds=i)
        entry = base.copy()
        entry["timestamp"] = t
        padded.append(entry)

    # Now append actual history
    for i, h in enumerate(history):
        h_copy = h.copy()
        h_copy["timestamp"] = base_time + timedelta(seconds=310 + i)
        padded.append(h_copy)
    return padded


def predict_stream(hr, spo2, bpsys, bpdia, motion, history):
    if detector is None:
        return (
            generate_dashboard_html(0, False),
            0.0,
            history,
            "Model Error: Run training first.",
        )

    # Standardize input
    new_entry = {
        "patient_id": "LIVE_01",
        "heart_rate": float(hr),
        "spo2": float(spo2),
        "bp_sys": float(bpsys),
        "bp_dia": float(bpdia),
        "motion_signal": float(motion),
    }

    # Append to history
    history.append(new_entry)
    # Keep only last 10 minutes (600 seconds) max
    if len(history) > 600:
        history = history[-600:]

    # Build context dataframe (pad if < 310 rows so features work instantly)
    full_context = pad_history(history)
    df = pd.DataFrame(full_context)

    try:
        # Run Pipeline
        clean_df = cleaner.fit_transform(df)
        feat_df = engineer.fit_transform(clean_df)

        # We only care about the very last row for live dashboard output
        if len(feat_df) == 0:
            return generate_dashboard_html(0, False), 0.0, history, "Warming up..."

        # Predict ML
        results = detector.predict(feat_df)
        feat_df["anomaly_prob"] = results["anomaly_probabilities"]

        # Score Risk
        final_df = scorer.score_data(feat_df)
        last_row = final_df.iloc[-1]

        risk = last_row["risk_score"]
        alert = bool(last_row["alert_flag"])
        conf = last_row["confidence_score"]
        ml_prob = last_row["anomaly_prob"]

        html_out = generate_dashboard_html(risk, alert)

        debug_text = f"Analyzed {len(feat_df)} active windows. Last ML Anomaly Prob: {ml_prob:.2f}. Last Risk: {risk:.1f}. Alert Logic: {alert}"

        return html_out, conf, history, debug_text

    except Exception as e:
        import traceback

        return (
            generate_dashboard_html(0, False),
            0.0,
            history,
            f"Pipeline Error: {str(e)}\n{traceback.format_exc()}",
        )


# --- Gradio UI ---
theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="blue",
).set(
    body_background_fill="#0f172a",
    body_text_color="white",
    block_background_fill="#1e293b",
    block_border_width="1px",
    block_border_color="rgba(255,255,255,0.1)",
)

with gr.Blocks(
    theme=theme, css=CUSTOM_CSS, title="Smart Ambulance Live Monitor"
) as demo:
    history_state = gr.State([])

    gr.HTML(
        "<h1 style='text-align:center; margin-bottom:0; color:white;'>🚑 Smart Ambulance AI Edge Monitor</h1><p style='text-align:center; color:#94a3b8; margin-top:5px;'>Predictive Deterioration & Early Warning System</p>"
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Live Patient Vitals Stream")

            with gr.Group():
                hr_slider = gr.Slider(
                    minimum=40,
                    maximum=180,
                    value=75,
                    step=1,
                    label="Heart Rate (bpm)",
                    interactive=True,
                )
                spo2_slider = gr.Slider(
                    minimum=70,
                    maximum=100,
                    value=98,
                    step=1,
                    label="SpO2 (%)",
                    interactive=True,
                )
                bpsys_slider = gr.Slider(
                    minimum=60,
                    maximum=220,
                    value=120,
                    step=1,
                    label="Systolic BP (mmHg)",
                    interactive=True,
                )
                bpdia_slider = gr.Slider(
                    minimum=40,
                    maximum=140,
                    value=80,
                    step=1,
                    label="Diastolic BP (mmHg)",
                    interactive=True,
                )
                motion_slider = gr.Slider(
                    minimum=0.0,
                    maximum=5.0,
                    value=1.0,
                    step=0.1,
                    label="Motion/Vibration Signal (G-force)",
                    interactive=True,
                )

            with gr.Row():
                stream_btn = gr.Button(
                    "🔵 Stream Reading (1 Hz)", variant="primary", size="lg"
                )
                reset_btn = gr.Button(
                    "Reset Patient Session", variant="secondary", size="lg"
                )

        with gr.Column(scale=1):
            dashboard_html = gr.HTML(generate_dashboard_html(0, False))
            conf_display = gr.Number(
                label="System Signal Confidence (0-1.0)", value=1.0, precision=2
            )
            debug_out = gr.Textbox(
                label="System Log",
                placeholder="Waiting for stream...",
                interactive=False,
            )

    # Logic
    stream_btn.click(
        fn=predict_stream,
        inputs=[
            hr_slider,
            spo2_slider,
            bpsys_slider,
            bpdia_slider,
            motion_slider,
            history_state,
        ],
        outputs=[dashboard_html, conf_display, history_state, debug_out],
    )

    def reset_session():
        return [], generate_dashboard_html(0, False), 1.0, "Session Reset."

    reset_btn.click(
        fn=reset_session,
        inputs=[],
        outputs=[history_state, dashboard_html, conf_display, debug_out],
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
