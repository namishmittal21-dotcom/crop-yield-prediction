"""
AI-Based Crop Yield Prediction System - Streamlit web app (the hosted version)

Run locally:  streamlit run streamlit_app.py
Hosted on Streamlit Community Cloud straight from the GitHub repo.
"""
import json

import pandas as pd
import streamlit as st

import crop_model as cm

GITHUB_URL = "https://github.com/namishmittal21-dotcom/crop-yield-prediction"

st.set_page_config(page_title="AI Crop Yield Prediction", page_icon="🌾", layout="wide")


@st.cache_resource(show_spinner="Preparing the model. The first visit after the app wakes up takes about a minute…")
def get_bundle():
    return cm.load_or_train()


B = get_bundle()
STATES = list(B["state_profiles"].index)

# key: (label, min, max, step)
SOIL = {"N": ("Nitrogen (N)", 30.0, 200.0, 1.0), "P": ("Phosphorus (P)", 5.0, 100.0, 1.0),
        "K": ("Potassium (K)", 10.0, 100.0, 1.0), "pH": ("Soil pH", 4.0, 9.0, 0.1)}
WEATHER = {"Annual_Rainfall": ("Annual rainfall (mm)", 200.0, 5000.0, 10.0),
           "Avg_Temp_C": ("Average temperature (°C)", 0.0, 35.0, 0.1),
           "Max_Temp_C": ("Average daily maximum temperature (°C)", 5.0, 42.0, 0.1),
           "Humidity_pct": ("Relative humidity (%)", 20.0, 100.0, 0.5),
           "Solar_Radiation_MJ": ("Solar radiation (MJ/m² per day)", 10.0, 25.0, 0.1)}
RANGES = {**SOIL, **WEATHER}


def fill_state_conditions():
    """Put the selected state's typical soil and weather into the sliders."""
    for key, value in cm.state_defaults(B, st.session_state["state"]).items():
        _, lo, hi, _ = RANGES[key]
        st.session_state[key] = round(min(max(value, lo), hi), 2)


def load_example(crop, season, state, year, area):
    st.session_state.update(crop=crop, season=season, state=state, year=year, area=float(area))
    fill_state_conditions()


if "state" not in st.session_state:            # first run of this browser session
    load_example(*cm.EXAMPLES[0])
    st.session_state.update(fert=float(B["fert_median"]), pest=float(B["pest_median"]))

try:
    with open(cm.OUTPUTS + "/summary.json") as f:
        SUMMARY = json.load(f)
except OSError:
    SUMMARY = {}

# ---------------------------------------------------------------- header
st.title("🌾 AI-Based Crop Yield Prediction System")
st.markdown("Predict crop yield for Indian states with a **Random Forest** trained on **real crop records**, "
            "**soil nutrients (N, P, K, pH)** and **24 years of weather data**.")
m = st.columns(4)
m[0].metric("Real crop records", "19,689")
m[1].metric("Crops · States", f"{len(B['crops'])} · {len(STATES)}")
m[2].metric("Test R²", f"{SUMMARY.get('final_test_r2', 0):.3f}" if SUMMARY else "–")
m[3].metric("R² on unseen years (2017–20)", f"{SUMMARY.get('future_r2', 0):.3f}" if SUMMARY else "–")

tab_predict, tab_models, tab_importance, tab_data = st.tabs(
    ["🔮 Predict", "📊 Model comparison", "🧠 Feature importance", "🗂️ Data & method"])

# ---------------------------------------------------------------- predict
with tab_predict:
    st.caption("Try an example, or pick your own inputs. The prediction updates as you change anything.")
    ex_cols = st.columns(len(cm.EXAMPLES))
    for col, ex in zip(ex_cols, cm.EXAMPLES):
        col.button(f"{ex[0]} · {ex[2]}", on_click=load_example, args=ex, width="stretch")

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("🌱 Crop and farm")
        st.selectbox("Crop", B["crops"], key="crop")
        st.selectbox("Season", B["seasons"], key="season")
        st.selectbox("State (fills in typical soil and weather)", STATES, key="state", on_change=fill_state_conditions)
        st.slider("Crop year", 1997, 2030, key="year")
        st.number_input("Area (hectares)", min_value=0.1, step=100.0, key="area")
        st.slider("Fertilizer used (kg per hectare)", 50.0, 300.0, step=1.0, key="fert")
        st.slider("Pesticide used (kg per hectare)", 0.05, 0.8, step=0.01, key="pest")
    with right:
        st.subheader("🧪 Soil nutrients")
        for key, (label, lo, hi, step) in SOIL.items():
            st.slider(label, lo, hi, step=step, key=key)
        st.subheader("☀️ Weather (yearly averages)")
        for key, (label, lo, hi, step) in WEATHER.items():
            st.slider(label, lo, hi, step=step, key=key)

    s = st.session_state
    r = cm.predict(B, s.crop, s.season, s.state, s.year, s.area, s.fert, s.pest, {k: s[k] for k in RANGES})
    unit = r["unit"]

    st.divider()
    st.subheader(f"Result: {s.crop} · {s.season} · {s.state} · {s.year}")
    c = st.columns(3)
    delta = None
    if r["history"]:
        diff = (r["estimate"] - r["history"]) / r["history"] * 100
        delta = f"{diff:+.0f}% vs real median ({r['history']:,.2f})"
    c[0].metric(f"Estimated yield ({unit} per hectare)", f"{r['estimate']:,.2f}", delta, delta_color="off")
    c[1].metric(f"Estimated production on {r['area']:,.1f} ha", f"{r['production']:,.1f} {unit}")
    c[2].metric("80% range (spread of the forest's trees)", f"{r['low']:,.2f} – {r['high']:,.2f}")
    notes = []
    if r["history"]:
        notes.append(f"Real median yield for {s.crop} in {s.state}, 1997–2020: **{r['history']:,.2f} {unit}/ha**.")
    else:
        notes.append(f"The dataset has no records of {s.crop} in {s.state}, so treat this estimate with extra care.")
    if r["year"] > 2020:
        notes.append("Years after 2020 are beyond the training data; the model extrapolates from past trends.")
    st.info(" ".join(notes) + f" Model: {r['model_name']}.")

# ---------------------------------------------------------------- models
with tab_models:
    st.markdown("Five models trained on the **same preprocessing pipeline**, ranked by 5-fold cross-validation R² "
                "(higher is better). Errors are on log(1 + yield), so crops of different scales are comparable.")
    if cm.output_path("model_comparison.csv"):
        st.dataframe(pd.read_csv(cm.output_path("model_comparison.csv")).round(4), hide_index=True,
                     width="stretch")
    if cm.output_path("model_comparison.png"):
        st.image(cm.output_path("model_comparison.png"), width="stretch")
    if cm.output_path("feature_group_study.png"):
        st.subheader("What each data source adds")
        st.image(cm.output_path("feature_group_study.png"), width="stretch")

# ---------------------------------------------------------------- importance
with tab_importance:
    st.markdown("Which inputs the Random Forest relies on most: built-in importance (left) and permutation "
                "importance on the test set (right).")
    if cm.output_path("feature_importance.png"):
        st.image(cm.output_path("feature_importance.png"), width="stretch")
    if cm.output_path("feature_importance_zoom.png"):
        st.subheader("Beyond crop type: soil, weather, season and farm inputs")
        st.image(cm.output_path("feature_importance_zoom.png"), width="stretch")
    a, b = st.columns([1, 1.3])
    if cm.output_path("importance_by_group.png"):
        a.image(cm.output_path("importance_by_group.png"), width="stretch")
    if cm.output_path("feature_importance.csv"):
        b.dataframe(pd.read_csv(cm.output_path("feature_importance.csv")).round(4), hide_index=True,
                    width="stretch")

# ---------------------------------------------------------------- data
with tab_data:
    st.markdown(cm.DATA_SOURCES_MD)
    if cm.output_path("actual_vs_predicted.png"):
        st.image(cm.output_path("actual_vs_predicted.png"), width="stretch")

st.divider()
st.caption(f"Code, data and full analysis notebook: [{GITHUB_URL.removeprefix('https://')}]({GITHUB_URL}) · "
           "Estimates are for learning and comparison, not agronomic advice.")
