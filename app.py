import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Fetal Health Prediction System",
    page_icon="🩺",
    layout="wide",
)

# -----------------------------
# Load model
# -----------------------------
MODEL_PATH = Path("model.pkl")

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "model.pkl was not found in the same folder as app.py. "
            "Save your trained model first with joblib.dump(best_rf, 'model.pkl')."
        )
    return joblib.load(MODEL_PATH)

model = load_model()

# -----------------------------
# Constants
# -----------------------------
CLASS_LABELS = {
    1.0: "Normal",
    2.0: "Suspect",
    3.0: "Pathological",
    1: "Normal",
    2: "Suspect",
    3: "Pathological",
}

FEATURE_ORDER = [
    "baseline value",
    "accelerations",
    "fetal_movement",
    "uterine_contractions",
    "light_decelerations",
    "severe_decelerations",
    "prolongued_decelerations",
    "abnormal_short_term_variability",
    "mean_value_of_short_term_variability",
    "percentage_of_time_with_abnormal_long_term_variability",
    "mean_value_of_long_term_variability",
    "histogram_width",
    "histogram_min",
    "histogram_max",
    "histogram_number_of_peaks",
    "histogram_number_of_zeroes",
    "histogram_mode",
    "histogram_mean",
    "histogram_median",
    "histogram_variance",
    "histogram_tendency",
]

DEFAULTS = {
    "baseline value": 132.0,
    "accelerations": 0.003,
    "fetal_movement": 0.0,
    "uterine_contractions": 0.004,
    "light_decelerations": 0.001,
    "severe_decelerations": 0.0,
    "prolongued_decelerations": 0.0,
    "abnormal_short_term_variability": 46.0,
    "mean_value_of_short_term_variability": 1.3,
    "percentage_of_time_with_abnormal_long_term_variability": 0.0,
    "mean_value_of_long_term_variability": 8.3,
    "histogram_width": 70.0,
    "histogram_min": 109.0,
    "histogram_max": 179.0,
    "histogram_number_of_peaks": 3.0,
    "histogram_number_of_zeroes": 0.0,
    "histogram_mode": 137.0,
    "histogram_mean": 135.0,
    "histogram_median": 138.0,
    "histogram_variance": 18.0,
    "histogram_tendency": 1.0,
}

# -----------------------------
# Helpers
# -----------------------------
def make_input_df(values: dict) -> pd.DataFrame:
    return pd.DataFrame([[values[col] for col in FEATURE_ORDER]], columns=FEATURE_ORDER)


def risk_box(label: str, confidence: float) -> None:
    if label == "Normal":
        st.success(f"Prediction: {label} | Confidence: {confidence:.2%}")
    elif label == "Suspect":
        st.warning(f"Prediction: {label} | Confidence: {confidence:.2%}")
    else:
        st.error(f"Prediction: {label} | Confidence: {confidence:.2%}")


# -----------------------------
# UI
# -----------------------------
st.title("🩺 Fetal Health Prediction System")
st.caption("Clinical decision-support prototype based on cardiotocography (CTG) measurements.")

st.info(
    "This app is for academic demonstration only. It should not be used as a medical diagnosis tool."
)

left, right = st.columns([1.2, 1])

with left:
    st.subheader("Enter CTG Measurements")

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            baseline_value = st.number_input("Baseline Value", min_value=0.0, value=DEFAULTS["baseline value"], step=1.0)
            accelerations = st.number_input("Accelerations", min_value=0.0, value=DEFAULTS["accelerations"], step=0.001, format="%.3f")
            fetal_movement = st.number_input("Fetal Movement", min_value=0.0, value=DEFAULTS["fetal_movement"], step=0.001, format="%.3f")
            uterine_contractions = st.number_input("Uterine Contractions", min_value=0.0, value=DEFAULTS["uterine_contractions"], step=0.001, format="%.3f")
            light_decelerations = st.number_input("Light Decelerations", min_value=0.0, value=DEFAULTS["light_decelerations"], step=0.001, format="%.3f")
            severe_decelerations = st.number_input("Severe Decelerations", min_value=0.0, value=DEFAULTS["severe_decelerations"], step=0.001, format="%.3f")
            prolongued_decelerations = st.number_input("Prolonged Decelerations", min_value=0.0, value=DEFAULTS["prolongued_decelerations"], step=0.001, format="%.3f")

        with c2:
            abnormal_stv = st.number_input("Abnormal Short-Term Variability", min_value=0.0, value=DEFAULTS["abnormal_short_term_variability"], step=1.0)
            mean_stv = st.number_input("Mean Value of Short-Term Variability", min_value=0.0, value=DEFAULTS["mean_value_of_short_term_variability"], step=0.1)
            abnormal_ltv_pct = st.number_input("% Time with Abnormal Long-Term Variability", min_value=0.0, value=DEFAULTS["percentage_of_time_with_abnormal_long_term_variability"], step=1.0)
            mean_ltv = st.number_input("Mean Value of Long-Term Variability", min_value=0.0, value=DEFAULTS["mean_value_of_long_term_variability"], step=0.1)
            histogram_width = st.number_input("Histogram Width", min_value=0.0, value=DEFAULTS["histogram_width"], step=1.0)
            histogram_min = st.number_input("Histogram Min", min_value=0.0, value=DEFAULTS["histogram_min"], step=1.0)
            histogram_max = st.number_input("Histogram Max", min_value=0.0, value=DEFAULTS["histogram_max"], step=1.0)

        with c3:
            histogram_peaks = st.number_input("Histogram Number of Peaks", min_value=0.0, value=DEFAULTS["histogram_number_of_peaks"], step=1.0)
            histogram_zeroes = st.number_input("Histogram Number of Zeroes", min_value=0.0, value=DEFAULTS["histogram_number_of_zeroes"], step=1.0)
            histogram_mode = st.number_input("Histogram Mode", min_value=0.0, value=DEFAULTS["histogram_mode"], step=1.0)
            histogram_mean = st.number_input("Histogram Mean", min_value=0.0, value=DEFAULTS["histogram_mean"], step=1.0)
            histogram_median = st.number_input("Histogram Median", min_value=0.0, value=DEFAULTS["histogram_median"], step=1.0)
            histogram_variance = st.number_input("Histogram Variance", min_value=0.0, value=DEFAULTS["histogram_variance"], step=1.0)
            histogram_tendency = st.number_input("Histogram Tendency", value=DEFAULTS["histogram_tendency"], step=1.0)

        submitted = st.form_submit_button("Predict Fetal Health")

    input_values = {
        "baseline value": baseline_value,
        "accelerations": accelerations,
        "fetal_movement": fetal_movement,
        "uterine_contractions": uterine_contractions,
        "light_decelerations": light_decelerations,
        "severe_decelerations": severe_decelerations,
        "prolongued_decelerations": prolongued_decelerations,
        "abnormal_short_term_variability": abnormal_stv,
        "mean_value_of_short_term_variability": mean_stv,
        "percentage_of_time_with_abnormal_long_term_variability": abnormal_ltv_pct,
        "mean_value_of_long_term_variability": mean_ltv,
        "histogram_width": histogram_width,
        "histogram_min": histogram_min,
        "histogram_max": histogram_max,
        "histogram_number_of_peaks": histogram_peaks,
        "histogram_number_of_zeroes": histogram_zeroes,
        "histogram_mode": histogram_mode,
        "histogram_mean": histogram_mean,
        "histogram_median": histogram_median,
        "histogram_variance": histogram_variance,
        "histogram_tendency": histogram_tendency,
    }

with right:
    st.subheader("Prediction Output")

    if submitted:
        input_df = make_input_df(input_values)
        prediction = model.predict(input_df)[0]
        label = CLASS_LABELS.get(prediction, str(prediction))

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            confidence = float(max(probabilities))
            prob_df = pd.DataFrame({
                "Class": ["Normal", "Suspect", "Pathological"],
                "Probability": probabilities,
            })
        else:
            confidence = 0.0
            prob_df = None

        risk_box(label, confidence)

        st.markdown("### Input Summary")
        st.dataframe(input_df.T.rename(columns={0: "Value"}), use_container_width=True)

        if prob_df is not None:
            st.markdown("### Class Probabilities")
            st.dataframe(prob_df, use_container_width=True)
            st.bar_chart(prob_df.set_index("Class"))
    else:
        st.write("Submit the form to generate a prediction.")
