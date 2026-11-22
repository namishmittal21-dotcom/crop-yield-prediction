"""
AI-Based Crop Yield Prediction System - Gradio app (local version)

Run:  python app.py   (or double-click run_app.bat / ./run_app.sh)
The online version of the same app is streamlit_app.py.
"""
import os

import gradio as gr
import pandas as pd

import crop_model as cm

B = cm.load_or_train()
STATES = list(B["state_profiles"].index)
SLIDER_ORDER = ["N", "P", "K", "pH", "Annual_Rainfall", "Avg_Temp_C", "Max_Temp_C", "Humidity_pct", "Solar_Radiation_MJ"]


def state_defaults(state):
    d = cm.state_defaults(B, state)
    return [d[k] for k in SLIDER_ORDER]


def predict(crop, season, state, year, area, fert, pest, N, P, K, pH, rain, temp, tmax, hum, solar):
    try:
        conditions = dict(zip(SLIDER_ORDER, [N, P, K, pH, rain, temp, tmax, hum, solar]))
        r = cm.predict(B, crop, season, state, year, area, fert, pest, conditions)
    except (TypeError, ValueError) as err:
        return f"⚠️ {err if 'Area' in str(err) else 'Please enter a valid area.'}"
    unit = r["unit"]
    if r["history"]:
        diff = (r["estimate"] - r["history"]) / r["history"] * 100
        hist_line = (f"| 📚 Typical real yield for {crop} in {state} (1997–2020 median) | {r['history']:,.2f} {unit}/ha "
                     f"(this prediction is {abs(diff):.0f}% {'above' if diff >= 0 else 'below'} it) |")
    else:
        hist_line = f"| 📚 Typical real yield for {crop} in {state} | no records in the dataset |"
    note = " *(beyond 2020 the model extrapolates from past trends)*" if r["year"] > 2020 else ""
    return f"""
### 🌾 Estimated yield: **{r['estimate']:,.2f} {unit} per hectare**
| | |
|---|---|
| 📦 Estimated production on {r['area']:,.1f} ha | **{r['production']:,.1f} {unit}** |
| 🎯 80% range (10th–90th percentile of the forest's trees) | {r['low']:,.2f} – {r['high']:,.2f} {unit}/ha |
{hist_line}
| 🤖 Model | {r['model_name']} · year {r['year']}{note} |
"""


with gr.Blocks(title="AI Crop Yield Prediction") as demo:
    gr.Markdown(
        "# 🌾 AI-Based Crop Yield Prediction System\n"
        "Trained on **19,689 real crop records** from Indian states (1997–2020), enriched with **soil nutrients (N, P, K, pH)** "
        f"and **historical weather** (Open-Meteo API). Model: **{B['model_name']}**."
    )
    with gr.Tab("🔮 Predict"):
        d = cm.state_defaults(B, "Punjab")
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🌱 Crop and farm")
                crop = gr.Dropdown(B["crops"], value="Rice", label="Crop")
                season = gr.Dropdown(B["seasons"], value="Kharif", label="Season")
                state = gr.Dropdown(STATES, value="Punjab", label="State (fills in typical soil and weather)")
                year = gr.Slider(1997, 2030, value=2020, step=1, label="Crop year")
                area = gr.Number(value=1000, label="Area (hectares)", minimum=0.1)
                fert = gr.Slider(50, 300, value=B["fert_median"], step=1, label="Fertilizer used (kg per hectare)")
                pest = gr.Slider(0.05, 0.8, value=B["pest_median"], step=0.01, label="Pesticide used (kg per hectare)")
            with gr.Column():
                gr.Markdown("### 🧪 Soil nutrients")
                N = gr.Slider(30, 200, value=d["N"], step=1, label="Nitrogen (N)")
                P = gr.Slider(5, 100, value=d["P"], step=1, label="Phosphorus (P)")
                K = gr.Slider(10, 100, value=d["K"], step=1, label="Potassium (K)")
                pH = gr.Slider(4.0, 9.0, value=d["pH"], step=0.1, label="Soil pH")
                gr.Markdown("### ☀️ Weather (yearly averages)")
                rain = gr.Slider(200, 5000, value=d["Annual_Rainfall"], step=10, label="Annual rainfall (mm)")
                temp = gr.Slider(0, 35, value=d["Avg_Temp_C"], step=0.1, label="Average temperature (°C)")
                tmax = gr.Slider(5, 42, value=d["Max_Temp_C"], step=0.1, label="Average daily maximum temperature (°C)")
                hum = gr.Slider(20, 100, value=d["Humidity_pct"], step=0.5, label="Relative humidity (%)")
                solar = gr.Slider(10, 25, value=d["Solar_Radiation_MJ"], step=0.1, label="Solar radiation (MJ/m² per day)")
        btn = gr.Button("🔮 Predict Yield", variant="primary", size="lg")
        out = gr.Markdown()
        inputs = [crop, season, state, year, area, fert, pest, N, P, K, pH, rain, temp, tmax, hum, solar]
        state.change(state_defaults, inputs=state, outputs=[N, P, K, pH, rain, temp, tmax, hum, solar])
        btn.click(predict, inputs, out)
        gr.Examples([list(e) for e in cm.EXAMPLES], inputs=[crop, season, state, year, area],
                    label="Try an example (then press Predict)")

    with gr.Tab("📊 Model comparison"):
        gr.Markdown("Five models trained on the same pipeline; ranked by 5-fold cross-validation R² (higher is better).")
        if cm.output_path("model_comparison.csv"):
            gr.Dataframe(pd.read_csv(cm.output_path("model_comparison.csv")).round(4), interactive=False)
        if cm.output_path("model_comparison.png"):
            gr.Image(cm.output_path("model_comparison.png"), show_label=False)
        if cm.output_path("feature_group_study.png"):
            gr.Markdown("### What each data source adds")
            gr.Image(cm.output_path("feature_group_study.png"), show_label=False)

    with gr.Tab("🧠 Feature importance"):
        gr.Markdown("Which inputs the Random Forest relies on most: built-in importance (left) and permutation importance "
                    "on the test set (right).")
        if cm.output_path("feature_importance.png"):
            gr.Image(cm.output_path("feature_importance.png"), show_label=False)
        if cm.output_path("feature_importance_zoom.png"):
            gr.Markdown("### Beyond crop type: soil, weather, season and farm inputs")
            gr.Image(cm.output_path("feature_importance_zoom.png"), show_label=False)
        with gr.Row():
            if cm.output_path("importance_by_group.png"):
                gr.Image(cm.output_path("importance_by_group.png"), show_label=False)
            if cm.output_path("feature_importance.csv"):
                gr.Dataframe(pd.read_csv(cm.output_path("feature_importance.csv")).round(4), interactive=False)

    with gr.Tab("🗂️ Data & method"):
        gr.Markdown(cm.DATA_SOURCES_MD)
        if cm.output_path("actual_vs_predicted.png"):
            gr.Image(cm.output_path("actual_vs_predicted.png"), show_label=False)

if __name__ == "__main__":
    demo.launch(inbrowser=True)
