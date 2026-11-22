"""
AI-Based Crop Yield Prediction System - Gradio app

Uses the model trained in Crops_Prediction.ipynb (models/crop_yield_model.joblib).
If the saved model is missing, it trains the Random Forest pipeline from the files in data/ first (about a minute).

Run:  python app.py   (or double-click "Run App.bat")
"""
import json
import os

import gradio as gr
import joblib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUTPUTS = os.path.join(HERE, "outputs")
MODEL_PATH = os.path.join(HERE, "models", "crop_yield_model.joblib")

CAT_COLS = ["Crop", "Season"]
LOG_COLS = ["Area"]
NUM_COLS = ["Crop_Year", "Annual_Rainfall", "Fertilizer_per_ha", "Pesticide_per_ha",
            "N", "P", "K", "pH", "Avg_Temp_C", "Max_Temp_C", "Humidity_pct", "Solar_Radiation_MJ"]
FEATURES = CAT_COLS + LOG_COLS + NUM_COLS


def load_merged():
    crop = pd.read_csv(os.path.join(DATA, "crop_yield.csv"))
    for col in ["Crop", "Season", "State"]:
        crop[col] = crop[col].str.strip()
    soil = pd.read_csv(os.path.join(DATA, "state_soil.csv"))
    weather = pd.read_csv(os.path.join(DATA, "state_weather.csv"))
    df = crop.drop_duplicates().merge(soil, on="State").merge(weather, on=["State", "Crop_Year"])
    df["Fertilizer_per_ha"] = df["Fertilizer"] / df["Area"]
    df["Pesticide_per_ha"] = df["Pesticide"] / df["Area"]
    return df


def train_and_save():
    """Same pipeline as the notebook (used only if the saved model is missing)."""
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

    df = load_merged()
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
        ("log", Pipeline([("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")), ("scale", StandardScaler())]), LOG_COLS),
        ("num", StandardScaler(), NUM_COLS),
    ])
    params, name = {"n_estimators": 150, "max_features": 0.5}, "Random Forest"
    best = os.path.join(OUTPUTS, "best_params.json")
    if os.path.exists(best):                      # use the tuned settings found in the notebook
        saved = json.load(open(best))
        params, name = saved["params"], saved["model_name"]
    model = Pipeline([("preprocess", pre),
                      ("model", RandomForestRegressor(**params, n_jobs=-1, random_state=42))])
    model.fit(df[FEATURES], np.log1p(df["Yield"]))
    bundle = {
        "model": model, "model_name": name, "features": FEATURES,
        "state_profiles": df.groupby("State")[["N", "P", "K", "pH", "Annual_Rainfall", "Avg_Temp_C", "Max_Temp_C",
                                               "Humidity_pct", "Solar_Radiation_MJ"]].median().round(2),
        "history": df.groupby(["Crop", "State"])["Yield"].median().round(3),
        "crops": sorted(df["Crop"].unique()), "seasons": sorted(df["Season"].unique()),
        "fert_median": round(df["Fertilizer_per_ha"].median(), 1), "pest_median": round(df["Pesticide_per_ha"].median(), 2),
    }
    import sklearn
    bundle["sklearn_version"] = sklearn.__version__
    try:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(bundle, MODEL_PATH, compress=3)
    except OSError as err:                            # read-only disk (some hosts): keep the model in memory
        print(f"Could not save the model ({err}); continuing without saving.")
    return bundle


def load_or_train():
    """Load the saved model; retrain if it is missing or was saved with a different scikit-learn version."""
    import sklearn
    if os.path.exists(MODEL_PATH):
        try:
            bundle = joblib.load(MODEL_PATH)
            if bundle.get("sklearn_version") in (None, sklearn.__version__):
                print(f"Loaded saved model: {bundle['model_name']}")
                return bundle
            print("Saved model was made with another scikit-learn version - retraining...")
        except Exception as err:                      # corrupt or incompatible file
            print(f"Could not load the saved model ({err}) - retraining...")
    else:
        print("No saved model found - training the Random Forest on the real data (about a minute)...")
    bundle = train_and_save()
    print("Model trained and saved to", MODEL_PATH)
    return bundle


B = load_or_train()

MODEL, MODEL_NAME, PROFILES, HISTORY = B["model"], B["model_name"], B["state_profiles"], B["history"]
STATES = list(PROFILES.index)


def read_output(name):
    path = os.path.join(OUTPUTS, name)
    return path if os.path.exists(path) else None


def predict_with_range(row):
    Xt = MODEL.named_steps["preprocess"].transform(row)
    trees = np.array([t.predict(Xt) for t in MODEL.named_steps["model"].estimators_])[:, 0]
    # middle (median) tree answer as the estimate; 10th-90th percentile of the trees as the range
    low, mid, high = np.percentile(trees, [10, 50, 90])
    return float(np.expm1(mid)), float(np.expm1(low)), float(np.expm1(high))


def state_defaults(state):
    p = PROFILES.loc[state]
    return [p["N"], p["P"], p["K"], p["pH"], p["Annual_Rainfall"], p["Avg_Temp_C"], p["Max_Temp_C"],
            p["Humidity_pct"], p["Solar_Radiation_MJ"]]


def predict(crop, season, state, year, area, fert, pest, N, P, K, pH, rain, temp, tmax, hum, solar):
    try:
        area = float(area)
    except (TypeError, ValueError):
        return "⚠️ Please enter a valid area."
    if area <= 0:
        return "⚠️ Area must be greater than 0."
    row = pd.DataFrame([{
        "Crop": crop, "Season": season, "Area": area, "Crop_Year": int(year), "Annual_Rainfall": float(rain),
        "Fertilizer_per_ha": float(fert), "Pesticide_per_ha": float(pest), "N": float(N), "P": float(P),
        "K": float(K), "pH": float(pH), "Avg_Temp_C": float(temp), "Max_Temp_C": float(tmax),
        "Humidity_pct": float(hum), "Solar_Radiation_MJ": float(solar),
    }])[FEATURES]
    est, low, high = predict_with_range(row)
    unit = "nuts" if crop == "Coconut" else "tonnes"
    hist = HISTORY.get((crop, state))
    if hist is not None and hist > 0:
        diff = (est - hist) / hist * 100
        hist_line = (f"| 📚 Typical real yield for {crop} in {state} (1997–2020 median) | {hist:,.2f} {unit}/ha "
                     f"(this prediction is {abs(diff):.0f}% {'above' if diff >= 0 else 'below'} it) |")
    else:
        hist_line = f"| 📚 Typical real yield for {crop} in {state} | no records in the dataset |"
    note = " *(beyond 2020 the model extrapolates from past trends)*" if int(year) > 2020 else ""
    return f"""
### 🌾 Estimated yield: **{est:,.2f} {unit} per hectare**
| | |
|---|---|
| 📦 Estimated production on {area:,.1f} ha | **{est * area:,.1f} {unit}** |
| 🎯 80% range (10th–90th percentile of the forest's trees) | {low:,.2f} – {high:,.2f} {unit}/ha |
{hist_line}
| 🤖 Model | {MODEL_NAME} · year {int(year)}{note} |
"""


with gr.Blocks(title="AI Crop Yield Prediction") as demo:
    gr.Markdown(
        "# 🌾 AI-Based Crop Yield Prediction System\n"
        "Trained on **19,689 real crop records** from Indian states (1997–2020), enriched with **soil nutrients (N, P, K, pH)** "
        f"and **historical weather** (Open-Meteo API). Model: **{MODEL_NAME}**."
    )
    with gr.Tab("🔮 Predict"):
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
                d = state_defaults("Punjab")
                gr.Markdown("### 🧪 Soil nutrients")
                N = gr.Slider(30, 200, value=d[0], step=1, label="Nitrogen (N)")
                P = gr.Slider(5, 100, value=d[1], step=1, label="Phosphorus (P)")
                K = gr.Slider(10, 100, value=d[2], step=1, label="Potassium (K)")
                pH = gr.Slider(4.0, 9.0, value=d[3], step=0.1, label="Soil pH")
                gr.Markdown("### ☀️ Weather (yearly averages)")
                rain = gr.Slider(200, 5000, value=d[4], step=10, label="Annual rainfall (mm)")
                temp = gr.Slider(0, 35, value=d[5], step=0.1, label="Average temperature (°C)")
                tmax = gr.Slider(5, 42, value=d[6], step=0.1, label="Average daily maximum temperature (°C)")
                hum = gr.Slider(20, 100, value=d[7], step=0.5, label="Relative humidity (%)")
                solar = gr.Slider(10, 25, value=d[8], step=0.1, label="Solar radiation (MJ/m² per day)")
        btn = gr.Button("🔮 Predict Yield", variant="primary", size="lg")
        out = gr.Markdown()
        inputs = [crop, season, state, year, area, fert, pest, N, P, K, pH, rain, temp, tmax, hum, solar]
        state.change(state_defaults, inputs=state, outputs=[N, P, K, pH, rain, temp, tmax, hum, solar])
        btn.click(predict, inputs, out)
        gr.Examples([["Rice", "Kharif", "Punjab", 2020, 1000], ["Wheat", "Rabi", "Uttar Pradesh", 2020, 1000],
                     ["Sugarcane", "Whole Year", "Maharashtra", 2020, 500], ["Cotton(lint)", "Kharif", "Gujarat", 2020, 800]],
                    inputs=[crop, season, state, year, area], label="Try an example (then press Predict)")

    with gr.Tab("📊 Model comparison"):
        gr.Markdown("Five models trained on the same pipeline; ranked by 5-fold cross-validation R² (higher is better).")
        if read_output("model_comparison.csv"):
            gr.Dataframe(pd.read_csv(read_output("model_comparison.csv")).round(4), interactive=False)
        if read_output("model_comparison.png"):
            gr.Image(read_output("model_comparison.png"), show_label=False)
        if read_output("feature_group_study.png"):
            gr.Markdown("### What each data source adds")
            gr.Image(read_output("feature_group_study.png"), show_label=False)

    with gr.Tab("🧠 Feature importance"):
        gr.Markdown("Which inputs the Random Forest relies on most: built-in importance (left) and permutation importance "
                    "on the test set (right).")
        if read_output("feature_importance.png"):
            gr.Image(read_output("feature_importance.png"), show_label=False)
        if read_output("feature_importance_zoom.png"):
            gr.Markdown("### Beyond crop type: soil, weather, season and farm inputs")
            gr.Image(read_output("feature_importance_zoom.png"), show_label=False)
        with gr.Row():
            if read_output("importance_by_group.png"):
                gr.Image(read_output("importance_by_group.png"), show_label=False)
            if read_output("feature_importance.csv"):
                gr.Dataframe(pd.read_csv(read_output("feature_importance.csv")).round(4), interactive=False)

    with gr.Tab("🗂️ Data & method"):
        gr.Markdown("""
| Data | Adds | Source |
|---|---|---|
| Crop yield records (19,689 rows, 1997–2020) | Crop, season, state, year, area, production, rainfall, fertilizer, pesticide, yield | Kaggle *Crop Yield in Indian States* (akshatgupta7); Mendeley Data DOI 10.17632/ncw2vbcgnk.2 |
| Soil profile per state | Nitrogen, phosphorus, potassium, pH | Kaggle *Crop Yield Data with Soil and Weather* (anshumish) |
| Weather per state and year | Avg and max temperature, humidity, solar radiation | Open-Meteo Historical Weather API (ERA5 reanalysis) |

**Method:** log(1 + yield) target · one-hot crop and season · scaled numeric inputs · 5 models compared with 5-fold CV ·
Random Forest tuned with RandomizedSearchCV · checked on unseen years 2017–2020.

**Limitations:** state-level averages rather than farm measurements; soil is one value per state; predictions after 2020 extrapolate past trends.
""")
        if read_output("actual_vs_predicted.png"):
            gr.Image(read_output("actual_vs_predicted.png"), show_label=False)

if __name__ == "__main__":
    on_hosting = bool(os.environ.get("SPACE_ID"))      # set automatically on Hugging Face Spaces
    demo.launch(inbrowser=not on_hosting)
