"""
Shared model code for the Crop Yield Prediction apps (Gradio: app.py, Streamlit: streamlit_app.py).

Loads the saved Random Forest (models/crop_yield_model.joblib), or trains it from the files in data/
when it is missing or was saved with a different scikit-learn version (about a minute).
"""
import json
import os

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
PROFILE_COLS = ["N", "P", "K", "pH", "Annual_Rainfall", "Avg_Temp_C", "Max_Temp_C", "Humidity_pct", "Solar_Radiation_MJ"]

DATA_SOURCES_MD = """
| Data | Adds | Source |
|---|---|---|
| Crop yield records (19,689 rows, 1997–2020) | Crop, season, state, year, area, production, rainfall, fertilizer, pesticide, yield | Kaggle *Crop Yield in Indian States* (akshatgupta7); Mendeley Data DOI 10.17632/ncw2vbcgnk.2 |
| Soil profile per state | Nitrogen, phosphorus, potassium, pH | Kaggle *Crop Yield Data with Soil and Weather* (anshumish) |
| Weather per state and year | Avg and max temperature, humidity, solar radiation | Open-Meteo Historical Weather API (ERA5 reanalysis) |

**Method:** log(1 + yield) target · one-hot crop and season · scaled numeric inputs · 5 models compared with 5-fold CV ·
Random Forest tuned with RandomizedSearchCV · checked on unseen years 2017–2020.

**Limitations:** state-level averages rather than farm measurements; soil is one value per state; predictions after 2020 extrapolate past trends.
"""

EXAMPLES = [("Rice", "Kharif", "Punjab", 2020, 1000), ("Wheat", "Rabi", "Uttar Pradesh", 2020, 1000),
            ("Sugarcane", "Whole Year", "Maharashtra", 2020, 500), ("Cotton(lint)", "Kharif", "Gujarat", 2020, 800)]


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
    """Same pipeline and settings as the final model in the notebook."""
    import sklearn
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

    df = load_merged()
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
        ("log", Pipeline([("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                          ("scale", StandardScaler())]), LOG_COLS),
        ("num", StandardScaler(), NUM_COLS),
    ])
    params, name = {"n_estimators": 150, "max_features": 0.5}, "Random Forest"
    best = os.path.join(OUTPUTS, "best_params.json")
    if os.path.exists(best):                          # settings chosen in the notebook
        with open(best) as f:
            saved = json.load(f)
        params, name = saved["params"], saved["model_name"]
    model = Pipeline([("preprocess", pre),
                      ("model", RandomForestRegressor(**params, n_jobs=-1, random_state=42))])
    model.fit(df[FEATURES], np.log1p(df["Yield"]))
    bundle = {
        "model": model, "model_name": name, "features": FEATURES,
        "state_profiles": df.groupby("State")[PROFILE_COLS].median().round(2),
        "history": df.groupby(["Crop", "State"])["Yield"].median().round(3),
        "crops": sorted(df["Crop"].unique()), "seasons": sorted(df["Season"].unique()),
        "fert_median": round(float(df["Fertilizer_per_ha"].median()), 1),
        "pest_median": round(float(df["Pesticide_per_ha"].median()), 2),
        "sklearn_version": sklearn.__version__,
    }
    try:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(bundle, MODEL_PATH, compress=3)
    except OSError as err:                            # read-only disk on some hosts: keep it in memory
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
    print("Model ready.")
    return bundle


def output_path(name):
    path = os.path.join(OUTPUTS, name)
    return path if os.path.exists(path) else None


def state_defaults(bundle, state):
    """Typical soil and weather for a state (median over 1997-2020)."""
    return {k: float(v) for k, v in bundle["state_profiles"].loc[state].items()}


def predict(bundle, crop, season, state, year, area, fert_per_ha, pest_per_ha, conditions):
    """Predict yield per hectare with an 80% range.

    conditions: dict with N, P, K, pH, Annual_Rainfall, Avg_Temp_C, Max_Temp_C, Humidity_pct, Solar_Radiation_MJ.
    Returns a dict with the estimate, range, production, unit and the real historical median for comparison.
    """
    area = float(area)
    if area <= 0:
        raise ValueError("Area must be greater than 0.")
    row = pd.DataFrame([{"Crop": crop, "Season": season, "Area": area, "Crop_Year": int(year),
                         "Fertilizer_per_ha": float(fert_per_ha), "Pesticide_per_ha": float(pest_per_ha),
                         **{k: float(conditions[k]) for k in PROFILE_COLS}}])[FEATURES]
    model = bundle["model"]
    Xt = model.named_steps["preprocess"].transform(row)
    trees = np.array([t.predict(Xt) for t in model.named_steps["model"].estimators_])[:, 0]
    low, mid, high = np.expm1(np.percentile(trees, [10, 50, 90]))   # median tree = estimate, 10th-90th = range
    hist = bundle["history"].get((crop, state))
    return {"estimate": float(mid), "low": float(low), "high": float(high), "production": float(mid) * area,
            "unit": "nuts" if crop == "Coconut" else "tonnes", "area": area, "year": int(year),
            "history": float(hist) if hist is not None and hist > 0 else None, "model_name": bundle["model_name"]}
