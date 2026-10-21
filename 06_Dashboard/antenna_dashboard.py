from pathlib import Path
import re
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="ML Antenna Selection", page_icon="📡", layout="wide")

st.title("ML-Based Reconfigurable Planar Antenna")
st.caption("Prediction • Candidate ranking • Radiation evidence • Model limitations")

# Expected project layout:
# ML_Antenna_Project/
#   02_Processed_Data/
#   04_Models/
#   05_Graphs/
#   06_Dashboard/antenna_dashboard.py
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
DATA_DIR = PROJECT_ROOT / "02_Processed_Data"
MODEL_DIR = PROJECT_ROOT / "04_Models"
GRAPH_DIR = PROJECT_ROOT / "05_Graphs"

st.sidebar.header("Project files")
st.sidebar.write(f"Project root: `{PROJECT_ROOT}`")
st.sidebar.caption("Place this file in ML_Antenna_Project/06_Dashboard/.")

def load_csv(name):
    p = DATA_DIR / name
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception as e:
        st.error(f"Could not read {p.name}: {e}")
        return None

def switch_features(configuration):
    name = str(configuration).upper()
    match = re.search(r"S([1-5])", name)
    number = int(match.group(1)) if match else 0
    state = 1 if "_ON" in name else 0
    return number, state

def get_model(filename):
    p = MODEL_DIR / filename
    if not p.exists():
        return None
    try:
        return joblib.load(p)
    except Exception as e:
        st.error(f"Could not load {filename}: {e}")
        return None

tabs = st.tabs(["Overview", "Predict", "Rank candidates", "Radiation patterns", "Validation & limitations"])

with tabs[0]:
    st.subheader("Project scope")
    st.write(
        "This prototype uses existing antenna simulation/export data and saved machine-learning "
        "models to estimate S11, gain and VSWR, rank candidates under explicit matching limits, "
        "and inspect available radiation-pattern evidence. It is a decision-support tool; "
        "it does not replace electromagnetic simulation or prove a universal optimum."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Matching limit: S11", "≤ −10 dB")
    c2.metric("Matching limit: VSWR", "≤ 2")
    c3.metric("Ranking weights", "Gain 40% · S11 30% · VSWR 30%")
    st.markdown("**Current evidence-based lead**")
    st.info("S5_OFF at 17.910 GHz is the current leading candidate under the existing balanced score. "
            "Confirm gain units/reference and radiation objectives before claiming it is the best overall pattern.")
    st.markdown("**Project workflow**")
    st.code("Simulation CSVs → preprocessing → saved ML models → predictions → matching constraints → ranking → radiation evidence")

with tabs[1]:
    st.subheader("Predict electrical performance")
    st.write("Predictions below are from the saved models and should be treated cautiously, especially for unseen frequency regions.")
    configs = ["Conventional", "S1_OFF", "S1_ON", "S2_OFF", "S2_ON", "S3_OFF", "S3_ON",
               "S4_OFF", "S4_ON", "S5_OFF", "S5_ON", "Slot", "Split"]
    configuration = st.selectbox("Configuration", configs, index=configs.index("S5_OFF"))
    frequency = st.number_input("Frequency (GHz)", min_value=0.1, max_value=100.0, value=17.910, step=0.001, format="%.3f")
    number, state = switch_features(configuration)
    X = pd.DataFrame([{
        "frequency_GHz": frequency,
        "switch_number": number,
        "switch_state": state,
        "configuration": configuration
    }])
    model_names = {
        "Predicted S11 (dB)": "s11_config_aware_model.joblib",
        "Predicted gain": "gain_config_aware_model.joblib",
        "Predicted VSWR": "vswr_config_aware_model.joblib",
    }
    if st.button("Predict performance", type="primary"):
        models = {label: get_model(filename) for label, filename in model_names.items()}
        if any(m is None for m in models.values()):
            st.error("One or more model files are missing or could not be loaded. Check 04_Models.")
        else:
            try:
                vals = {label: float(model.predict(X)[0]) for label, model in models.items()}
                a, b, c = st.columns(3)
                a.metric("Predicted S11", f"{vals['Predicted S11 (dB)']:.3f} dB")
                b.metric("Predicted gain", f"{vals['Predicted gain']:.3f}")
                c.metric("Predicted VSWR", f"{vals['Predicted VSWR']:.3f}")
                passed = vals["Predicted S11 (dB)"] <= -10 and vals["Predicted VSWR"] <= 2
                st.success("Passes current matching constraints" if passed else "Does not pass current matching constraints")
                st.warning("These are model predictions, not new HFSS simulation results. Held-out 18–19 GHz S11 validation was weak.")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
    else:
        st.caption("Choose a configuration and frequency, then press Predict performance.")

with tabs[2]:
    st.subheader("Candidate rankings")
    ml = load_csv("ml_predicted_candidate_ranking.csv")
    evidence = load_csv("radiation_supported_evidence_ranking.csv")
    if ml is not None:
        st.markdown("**ML-predicted ranking**")
        st.caption("Predicted values; ranking score is relative to this candidate set.")
        st.dataframe(ml.head(50), use_container_width=True, hide_index=True)
        st.download_button("Download ML ranking CSV", ml.to_csv(index=False).encode("utf-8"), "ml_predicted_candidate_ranking.csv", "text/csv")
    else:
        st.warning("ML ranking file not found in 02_Processed_Data.")
    if evidence is not None:
        st.markdown("**Radiation-supported evidence ranking**")
        st.caption("Uses available performance/radiation data; does not prove best 3D radiation pattern.")
        st.dataframe(evidence.head(50), use_container_width=True, hide_index=True)
        st.download_button("Download radiation evidence CSV", evidence.to_csv(index=False).encode("utf-8"), "radiation_supported_evidence_ranking.csv", "text/csv")
    else:
        st.warning("Radiation-supported ranking file not found in 02_Processed_Data.")

with tabs[3]:
    st.subheader("Radiation-pattern comparison")
    st.write("The two available cuts are shown for S5_OFF, S4_ON and Split at their respective frequencies.")
    images = [
        ("Phi = 0° cut", GRAPH_DIR / "radiation_comparison_phi_0.png"),
        ("Phi = 90° cut", GRAPH_DIR / "radiation_comparison_phi_90.png"),
    ]
    cols = st.columns(2)
    for col, (label, image_path) in zip(cols, images):
        with col:
            st.markdown(f"**{label}**")
            if image_path.exists():
                st.image(str(image_path), use_container_width=True)
            else:
                st.warning(f"Image not found: {image_path.name}")
    st.info("The plots show angular variation and deep nulls. Two cuts are not a complete 3D pattern. Define the intended radiation objective before declaring a universal best.")

with tabs[4]:
    st.subheader("Validation results")
    st.markdown("**Held-out 18–19 GHz frequency-band evaluation**")
    metrics = pd.DataFrame([
        ["S11", "4.6877 dB", "6.9497 dB", "-0.7531"],
        ["Gain", "0.3996", "0.7354", "0.7648"],
        ["VSWR", "0.1054", "0.1275", "0.6062"],
    ], columns=["Target", "MAE", "RMSE", "R²"])
    st.dataframe(metrics, use_container_width=True, hide_index=True)
    st.warning("S11 generalization to the held-out band is weak. Do not treat the ML ranking as proof of a global optimum. "
               "Use simulation-derived candidate evidence for corroboration and clearly separate predicted from observed values.")
    st.markdown("**Known limitations**")
    st.markdown(
        "- Radiation-pattern data are sparse compared with electrical performance data.\n"
        "- Gain units/reference must be confirmed from the original export.\n"
        "- Only phi = 0° and 90° pattern cuts are currently compared.\n"
        "- The 40/30/30 score is a project-defined relative score, not a physical law.\n"
        "- Experimental validation should not be claimed unless measurements were performed."
    )
