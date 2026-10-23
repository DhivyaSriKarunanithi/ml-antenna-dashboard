from pathlib import Path
import re
import warnings

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# ML-Based Reconfigurable Planar Antenna — visual, interactive dashboard
# Expected layout:
# ML_Antenna_Project/
#   02_Processed_Data/  (ranking CSVs and any prepared result tables)
#   04_Models/          (saved .joblib models)
#   05_Graphs/          (radiation comparison PNGs)
#   06_Dashboard/antenna_dashboard.py
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="ML Antenna Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
DATA_DIR = PROJECT_ROOT / "02_Processed_Data"
MODEL_DIR = PROJECT_ROOT / "04_Models"
GRAPH_DIR = PROJECT_ROOT / "05_Graphs"

CONFIGS = [
    "Conventional", "S1_OFF", "S1_ON", "S2_OFF", "S2_ON", "S3_OFF",
    "S3_ON", "S4_OFF", "S4_ON", "S5_OFF", "S5_ON", "Slot", "Split"
]
MODEL_FILES = {
    "S11": "s11_config_aware_model.joblib",
    "Gain": "gain_config_aware_model.joblib",
    "VSWR": "vswr_config_aware_model.joblib",
}

# Modern, calm styling. No result values are hardcoded into the charts.
st.markdown("""
<style>
  .block-container {padding-top: 1.6rem; padding-bottom: 2.5rem; max-width: 1500px;}
  h1, h2, h3 {letter-spacing: -0.025em;}
  .hero {padding: 1.35rem 1.5rem; border-radius: 18px; border: 1px solid rgba(100,116,139,.25);
         background: linear-gradient(115deg, rgba(37,99,235,.09), rgba(14,165,233,.035)); margin-bottom: 1rem;}
  .eyebrow {font-size: .78rem; text-transform: uppercase; letter-spacing: .12em; font-weight: 700; opacity: .72;}
  .hero-title {font-size: 1.8rem; font-weight: 750; line-height: 1.2; margin: .25rem 0 .55rem 0;}
  .hero-copy {font-size: 1rem; opacity: .82; max-width: 900px;}
  div[data-testid="stMetric"] {border: 1px solid rgba(100,116,139,.22); padding: 1rem 1rem .85rem; border-radius: 14px; background: rgba(148,163,184,.045);}
  div[data-testid="stMetricLabel"] {font-size: .9rem;}
  .small-note {font-size: .88rem; opacity: .75;}
  div[data-testid="stTabs"] button {font-weight: 600;}
</style>
""", unsafe_allow_html=True)


def load_csv(filename):
    path = DATA_DIR / filename
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        st.error(f"Could not read `{filename}`: {exc}")
        return None


@st.cache_resource(show_spinner=False)
def load_model(filename):
    path = MODEL_DIR / filename
    if not path.exists():
        return None
    return joblib.load(path)


def switch_features(configuration):
    name = str(configuration).upper()
    match = re.search(r"S([1-5])", name)
    switch_number = int(match.group(1)) if match else 0
    switch_state = 1 if "_ON" in name else 0
    return switch_number, switch_state


def numeric_column(df, candidates):
    """Return the first matching column name, case-insensitively."""
    if df is None:
        return None
    by_lower = {str(c).lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in by_lower:
            return by_lower[candidate.lower()]
    return None


def config_column(df):
    return numeric_column(df, ["configuration", "config", "antenna_configuration"])


def frequency_column(df):
    return numeric_column(df, ["frequency_GHz", "frequency_ghz", "frequency", "freq_GHz", "freq"])


def metric_column(df, metric, predicted=True):
    if metric == "S11":
        choices = (["predicted_S11_dB", "Predicted S11 (dB)", "S11_predicted", "S11_prediction"] if predicted
                   else ["S11_dB", "S11 (dB)", "S11", "measured_S11_dB", "actual_S11_dB"])
    elif metric == "Gain":
        choices = (["predicted_gain", "Predicted gain", "gain_predicted", "gain_prediction"] if predicted
                   else ["peak_gain_dB", "gain_dB", "Gain_dB", "gain", "measured_gain"])
    else:
        choices = (["predicted_VSWR", "Predicted VSWR", "VSWR_predicted", "VSWR_prediction"] if predicted
                   else ["VSWR", "vswr", "measured_VSWR", "actual_VSWR"])
    return numeric_column(df, choices)


def sort_ml_ranking(df):
    if df is None or df.empty:
        return df
    score = numeric_column(df, ["ML_balanced_score_0_100", "balanced_score_0_100", "balanced_score", "score"])
    if score:
        return df.sort_values(score, ascending=False, kind="stable")
    return df.copy()


def sort_evidence(df):
    if df is None or df.empty:
        return df
    score = numeric_column(df, ["balanced_score_0_100", "balanced_score", "score"])
    if score:
        return df.sort_values(score, ascending=False, kind="stable")
    rank = numeric_column(df, ["evidence_rank", "rank"])
    if rank:
        return df.sort_values(rank, ascending=True, kind="stable")
    return df.copy()


def get_lead(df, predicted=True):
    ranked = sort_ml_ranking(df) if predicted else sort_evidence(df)
    return None if ranked is None or ranked.empty else ranked.iloc[0]


def candidate_labels(df):
    """Create readable, unique labels for configuration/frequency operating points."""
    view = df.copy()
    ccol = config_column(view)
    fcol = frequency_column(view)
    if ccol and fcol:
        freq = pd.to_numeric(view[fcol], errors="coerce")
        view["_candidate_label"] = view[ccol].astype(str) + " · " + freq.map(
            lambda x: f"{x:.3f} GHz" if pd.notna(x) else "frequency N/A"
        )
    elif ccol:
        view["_candidate_label"] = view[ccol].astype(str)
    else:
        view["_candidate_label"] = view.index.astype(str)
    return view


def plot_bar(df, label_col, value_col, title, x_label, top_n=10, ascending=False):
    view = df[[label_col, value_col]].copy()
    view[value_col] = pd.to_numeric(view[value_col], errors="coerce")
    view = view.dropna().sort_values(value_col, ascending=ascending).head(top_n)
    if view.empty:
        st.info("There are no numeric values available for this chart.")
        return
    view = view.sort_values(value_col, ascending=True)
    fig, ax = plt.subplots(figsize=(9, max(3.2, len(view) * 0.42)))
    bars = ax.barh(view[label_col].astype(str), view[value_col], height=0.66)
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(x_label)
    ax.grid(axis="x", alpha=0.22)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, labelsize=9)
    for bar, value in zip(bars, view[value_col]):
        ax.annotate(f"{value:.3g}", (bar.get_width(), bar.get_y() + bar.get_height()/2),
                    xytext=(5, 0), textcoords="offset points", va="center", fontsize=8)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def plot_metric_frequency(ml, evidence, metric, selected_config=None):
    f_ml, y_ml = frequency_column(ml), metric_column(ml, metric, predicted=True)
    f_ev, y_ev = frequency_column(evidence), metric_column(evidence, metric, predicted=False)
    if (ml is None or f_ml is None or y_ml is None) and (evidence is None or f_ev is None or y_ev is None):
        st.info(f"A frequency-versus-{metric} graph needs matching frequency and {metric} columns in the available CSVs. The current ranking files may not contain them.")
        return
    fig, ax = plt.subplots(figsize=(10, 4.8))
    plotted = False
    if ml is not None and f_ml and y_ml:
        d = ml.copy()
        ccol = config_column(d)
        if selected_config and ccol:
            d = d[d[ccol].astype(str) == selected_config]
        d[f_ml] = pd.to_numeric(d[f_ml], errors="coerce")
        d[y_ml] = pd.to_numeric(d[y_ml], errors="coerce")
        d = d.dropna(subset=[f_ml, y_ml]).sort_values(f_ml)
        if not d.empty:
            if not selected_config and ccol:
                for cfg, group in d.groupby(ccol, sort=False):
                    group = group.sort_values(f_ml)
                    ax.plot(group[f_ml], group[y_ml], marker="o", markersize=2.5, linewidth=1.25, alpha=.8, label=f"ML: {cfg}")
            else:
                ax.plot(d[f_ml], d[y_ml], marker="o", markersize=3.5, linewidth=1.7, label="ML model predictions")
            plotted = True
    if evidence is not None and f_ev and y_ev:
        d = evidence.copy()
        ccol = config_column(d)
        if selected_config and ccol:
            d = d[d[ccol].astype(str) == selected_config]
        d[f_ev] = pd.to_numeric(d[f_ev], errors="coerce")
        d[y_ev] = pd.to_numeric(d[y_ev], errors="coerce")
        d = d.dropna(subset=[f_ev, y_ev]).sort_values(f_ev)
        if not d.empty:
            if not selected_config and ccol:
                for cfg, group in d.groupby(ccol, sort=False):
                    ax.scatter(group[f_ev], group[y_ev], s=36, marker="D", label=f"Evidence: {cfg}", zorder=3)
            else:
                ax.scatter(d[f_ev], d[y_ev], s=45, marker="D", label="Simulation-derived candidate evidence", zorder=3)
            plotted = True
    if not plotted:
        plt.close(fig)
        st.info("No numeric frequency/metric rows were found for the selected configuration.")
        return
    ax.set_title(f"{metric} across frequency", loc="left", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Frequency (GHz)")
    y_label = {"S11": "S11 (dB)", "Gain": "Gain (unit/reference to confirm)", "VSWR": "VSWR"}[metric]
    ax.set_ylabel(y_label)
    if metric == "S11":
        ax.axhline(-10, linestyle="--", linewidth=1, label="S11 matching limit (−10 dB)")
    if metric == "VSWR":
        ax.axhline(2, linestyle="--", linewidth=1, label="VSWR matching limit (2)")
    ax.grid(alpha=.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


ml = sort_ml_ranking(load_csv("ml_predicted_candidate_ranking.csv"))
evidence = sort_evidence(load_csv("radiation_supported_evidence_ranking.csv"))

st.markdown("""
<div class="hero">
  <div class="eyebrow">Engineering • Machine learning • Antenna performance</div>
  <div class="hero-title">ML-Based Reconfigurable Planar Antenna</div>
  <div class="hero-copy">Explore model predictions, compare candidate configurations, and review the simulation-derived evidence. The recommendation is a data-supported candidate—not proof of a universal optimum.</div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Dashboard controls")
    st.caption("Use the filters to explore candidate results.")
    config_filter = st.selectbox("Configuration filter", ["All configurations"] + CONFIGS, index=0)
    top_n = st.slider("Candidates to show", min_value=3, max_value=20, value=10, step=1)
    st.divider()
    st.markdown("### Project files")
    st.caption(f"Data: `{DATA_DIR}`")
    st.caption(f"Models: `{MODEL_DIR}`")
    st.caption(f"Graphs: `{GRAPH_DIR}`")
    st.caption("If a section says a file is missing, check the folder structure before deploying.")


tabs = st.tabs([
    "Executive overview", "Interactive prediction", "Candidate ranking",
    "Performance graphs", "Radiation patterns", "Model validation"
])

# 1. EXECUTIVE OVERVIEW
with tabs[0]:
    st.subheader("At a glance")
    ml_lead = get_lead(ml, predicted=True)
    ev_lead = get_lead(evidence, predicted=False)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Configurations represented", len(CONFIGS))
    c2.metric("Matching requirement", "S11 ≤ −10 dB")
    c3.metric("VSWR requirement", "≤ 2")
    c4.metric("Ranking method", "Balanced score")

    left, right = st.columns([1.05, 1])
    with left:
        st.markdown("#### ML model's leading candidate")
        if ml_lead is not None:
            cfg = str(ml_lead.get(config_column(ml), "Not available")) if config_column(ml) else "Not available"
            freq_col = frequency_column(ml)
            freq = pd.to_numeric(ml_lead.get(freq_col), errors="coerce") if freq_col else float("nan")
            st.success("Leading candidate in the saved ML ranking")
            st.markdown(f"### {cfg}")
            if pd.notna(freq):
                st.markdown(f"**Candidate frequency:** {freq:.3f} GHz")
            cols = st.columns(3)
            for col, metric, unit in zip(cols, ["S11", "Gain", "VSWR"], ["dB", "", ""]):
                mc = metric_column(ml, metric, predicted=True)
                if mc and pd.notna(pd.to_numeric(ml_lead.get(mc), errors="coerce")):
                    val = float(ml_lead[mc])
                    col.metric(f"Predicted {metric}", f"{val:.3f} {unit}".strip())
                else:
                    col.metric(f"Predicted {metric}", "N/A")
            score_col = numeric_column(ml, ["ML_balanced_score_0_100", "balanced_score_0_100", "balanced_score", "score"])
            if score_col:
                st.caption(f"Saved ranking score: {float(ml_lead[score_col]):.2f} (relative to the candidate set)")
            st.caption("These values are model predictions, not new HFSS simulation outputs.")
        else:
            st.warning("ML ranking CSV is missing or empty. Add `ml_predicted_candidate_ranking.csv` to `02_Processed_Data`.")
    with right:
        st.markdown("#### Leading simulation/radiation-supported candidate")
        if ev_lead is not None:
            cfg_col = config_column(evidence)
            cfg = str(ev_lead[cfg_col]) if cfg_col else "Not available"
            freq_col = frequency_column(evidence)
            freq = pd.to_numeric(ev_lead.get(freq_col), errors="coerce") if freq_col else float("nan")
            st.info("Separate evidence-based comparison")
            st.markdown(f"### {cfg}")
            if pd.notna(freq):
                st.markdown(f"**Candidate frequency:** {freq:.3f} GHz")
            cols = st.columns(3)
            for col, metric, unit in zip(cols, ["S11", "Gain", "VSWR"], ["dB", "", ""]):
                mc = metric_column(evidence, metric, predicted=False)
                if mc and pd.notna(pd.to_numeric(ev_lead.get(mc), errors="coerce")):
                    col.metric(f"Evidence {metric}", f"{float(ev_lead[mc]):.3f} {unit}".strip())
                else:
                    col.metric(f"Evidence {metric}", "N/A")
            st.caption("Uses the saved simulation/radiation evidence table; it is not a new model prediction.")
        else:
            st.warning("Radiation-supported ranking CSV is missing or empty.")

    st.markdown("#### How to read this result")
    st.write("The ML ranking shows which candidate scores highest under the saved model-based criteria. The evidence ranking separately compares candidates with available simulation/radiation data. If the leaders differ, report both rather than merging the two claims.")
    if ml is not None and not ml.empty:
        score_col = numeric_column(ml, ["ML_balanced_score_0_100", "balanced_score_0_100", "balanced_score", "score"])
        label_col = config_column(ml)
        if score_col and label_col:
            st.markdown("#### Top candidates by ML ranking score")
            ml_labels = candidate_labels(ml)
            plot_bar(ml_labels, "_candidate_label", score_col, "ML ranking: leading configuration / frequency candidates", "Project-defined relative score (0–100)", top_n=5)
    st.caption("Matching thresholds: S11 ≤ −10 dB and VSWR ≤ 2. The balanced score uses project-defined weights (gain 40%, S11 30%, VSWR 30%); it is not a universal antenna-quality metric.")

# 2. INTERACTIVE PREDICTION
with tabs[1]:
    st.subheader("Predict performance for a chosen candidate")
    st.write("Choose a configuration and frequency. The saved ML models estimate S11, gain and VSWR for that input.")
    p1, p2 = st.columns([1, 1])
    with p1:
        prediction_config = st.selectbox("Antenna configuration", CONFIGS, index=CONFIGS.index("S5_OFF"), key="predict_config")
    with p2:
        prediction_frequency = st.number_input("Frequency (GHz)", min_value=1.0, max_value=20.0, value=17.910, step=0.010, format="%.3f")
    st.caption("The current prepared data covers approximately 1–20 GHz. Predictions outside the training/data region should not be trusted without validation.")
    if st.button("Run ML prediction", type="primary", use_container_width=False):
        try:
            model_objects = {name: load_model(filename) for name, filename in MODEL_FILES.items()}
            missing = [name for name, obj in model_objects.items() if obj is None]
            if missing:
                st.error("Missing model file(s): " + ", ".join(MODEL_FILES[n] for n in missing) + ". Check `04_Models`.")
            else:
                switch_number, switch_state = switch_features(prediction_config)
                X = pd.DataFrame([{
                    "frequency_GHz": float(prediction_frequency),
                    "switch_number": switch_number,
                    "switch_state": switch_state,
                    "configuration": prediction_config,
                }])
                values = {
                    "S11": float(model_objects["S11"].predict(X)[0]),
                    "Gain": float(model_objects["Gain"].predict(X)[0]),
                    "VSWR": float(model_objects["VSWR"].predict(X)[0]),
                }
                st.session_state["last_prediction"] = {
                    "configuration": prediction_config,
                    "frequency_GHz": float(prediction_frequency),
                    **values,
                }
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

    result = st.session_state.get("last_prediction")
    if result:
        st.divider()
        st.markdown(f"#### Prediction result — {result['configuration']} at {result['frequency_GHz']:.3f} GHz")
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted S11", f"{result['S11']:.3f} dB")
        c2.metric("Predicted gain", f"{result['Gain']:.3f}")
        c3.metric("Predicted VSWR", f"{result['VSWR']:.3f}")
        passes = result["S11"] <= -10 and result["VSWR"] <= 2
        if passes:
            st.success("Prediction passes both current matching thresholds: S11 ≤ −10 dB and VSWR ≤ 2.")
        else:
            st.warning("Prediction does not pass both matching thresholds. Review S11 and VSWR before considering this candidate.")
        # A simple, clear visual comparison against the project thresholds.
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
        axes[0].bar(["Predicted S11"], [result["S11"]])
        axes[0].axhline(-10, color="gray", linestyle="--", label="Limit −10 dB")
        axes[0].set_ylabel("S11 (dB)")
        axes[0].legend(frameon=False, fontsize=8)
        axes[1].bar(["Predicted VSWR"], [result["VSWR"]])
        axes[1].axhline(2, color="gray", linestyle="--", label="Limit 2")
        axes[1].set_ylabel("VSWR")
        axes[1].legend(frameon=False, fontsize=8)
        for ax in axes:
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=.2)
            ax.set_axisbelow(True)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.warning("These are model predictions, not new HFSS results. The held-out 18–19 GHz evaluation showed weak S11 generalization, so treat this result as a screening estimate.")

# 3. RANKING
with tabs[2]:
    st.subheader("Which candidates rank highest?")
    if ml is None or ml.empty:
        st.warning("ML ranking CSV not found or empty. Expected: `02_Processed_Data/ml_predicted_candidate_ranking.csv`.")
    else:
        view = ml.copy()
        ccol = config_column(view)
        if config_filter != "All configurations" and ccol:
            view = view[view[ccol].astype(str) == config_filter]
        score_col = numeric_column(view, ["ML_balanced_score_0_100", "balanced_score_0_100", "balanced_score", "score"])
        label_col = ccol or view.columns[0]
        if score_col:
            view_labels = candidate_labels(view)
            plot_bar(view_labels, "_candidate_label", score_col, "Model-ranked configuration / frequency candidates", "Project-defined relative score (0–100)", top_n=top_n)
        else:
            st.info("No recognized score column was found. Showing the saved table order without inventing a score.")
        st.dataframe(view.head(top_n), use_container_width=True, hide_index=True)
        st.download_button("Download filtered ML ranking CSV", view.to_csv(index=False).encode("utf-8"), "ml_predicted_candidate_ranking_filtered.csv", "text/csv")

    st.divider()
    st.markdown("#### Simulation/radiation-supported comparison")
    if evidence is None or evidence.empty:
        st.warning("Radiation-supported CSV not found or empty. Expected: `02_Processed_Data/radiation_supported_evidence_ranking.csv`.")
    else:
        ev_view = evidence.copy()
        ccol = config_column(ev_view)
        if config_filter != "All configurations" and ccol:
            ev_view = ev_view[ev_view[ccol].astype(str) == config_filter]
        score_col = numeric_column(ev_view, ["balanced_score_0_100", "balanced_score", "score"])
        label_col = ccol or ev_view.columns[0]
        if score_col:
            ev_labels = candidate_labels(ev_view)
            plot_bar(ev_labels, "_candidate_label", score_col, "Evidence-supported configuration / frequency candidates", "Project-defined relative score", top_n=top_n)
        st.dataframe(ev_view.head(top_n), use_container_width=True, hide_index=True)
        st.download_button("Download filtered evidence CSV", ev_view.to_csv(index=False).encode("utf-8"), "radiation_evidence_ranking_filtered.csv", "text/csv")

# 4. PERFORMANCE GRAPHS
with tabs[3]:
    st.subheader("Performance across frequency")
    st.write("The charts use the available saved ranking tables. Model estimates and simulation-derived candidate evidence are labelled separately.")
    graph_config = st.selectbox("Focus configuration", ["All configurations"] + CONFIGS, index=0, key="graph_config")
    chosen_config = None if graph_config == "All configurations" else graph_config
    chart_tabs = st.tabs(["S11 vs frequency", "Gain vs frequency", "VSWR vs frequency"])
    for chart_tab, metric in zip(chart_tabs, ["S11", "Gain", "VSWR"]):
        with chart_tab:
            plot_metric_frequency(ml, evidence, metric, chosen_config)
    st.caption("If a curve is unavailable, the current saved ranking files may contain only shortlisted points. Add the prepared per-frequency CSV data to make a continuous trend plot possible.")

# 5. RADIATION PATTERNS
with tabs[4]:
    st.subheader("Radiation-pattern evidence")
    st.write("Available angular-cut comparisons for selected candidates. These are existing plots, not generated from assumed data.")
    image_options = [
        ("Phi = 0° cut", GRAPH_DIR / "radiation_comparison_phi_0.png"),
        ("Phi = 90° cut", GRAPH_DIR / "radiation_comparison_phi_90.png"),
    ]
    available = [(label, path) for label, path in image_options if path.exists()]
    if available:
        cols = st.columns(len(available))
        for col, (label, path) in zip(cols, available):
            with col:
                st.markdown(f"#### {label}")
                st.image(str(path), use_container_width=True)
    else:
        st.warning("Radiation comparison images were not found. Expected in `05_Graphs`: `radiation_comparison_phi_0.png` and `radiation_comparison_phi_90.png`.")
    st.info("Only phi = 0° and phi = 90° cuts are currently compared. Two cuts do not establish the complete 3D radiation pattern or prove universal radiation superiority.")
    if evidence is not None and not evidence.empty:
        st.markdown("#### Available evidence table")
        st.dataframe(evidence.head(top_n), use_container_width=True, hide_index=True)

# 6. VALIDATION
with tabs[5]:
    st.subheader("How reliable are the predictions?")
    st.write("Held-out frequency-band evaluation: 18–19 GHz. Metrics are reproduced from the existing project validation summary.")
    metrics = pd.DataFrame([
        {"Target": "S11", "MAE": 4.6877, "RMSE": 6.9497, "R²": -0.7531, "MAE unit": "dB"},
        {"Target": "Gain", "MAE": 0.3996, "RMSE": 0.7354, "R²": 0.7648, "MAE unit": "gain units unconfirmed"},
        {"Target": "VSWR", "MAE": 0.1054, "RMSE": 0.1275, "R²": 0.6062, "MAE unit": "ratio"},
    ])
    m1, m2 = st.columns(2)
    with m1:
        st.markdown("#### R² by target")
        plot_bar(metrics, "Target", "R²", "Held-out R² (higher is better)", "R²", top_n=3, ascending=True)
    with m2:
        st.markdown("#### MAE by target")
        plot_bar(metrics, "Target", "MAE", "Held-out MAE (lower is better; units differ)", "MAE (target-specific units)", top_n=3, ascending=False)
    st.dataframe(metrics, use_container_width=True, hide_index=True)
    st.warning("S11 generalization in the held-out 18–19 GHz band is weak (negative R²). Do not describe the ranking as proof of a global optimum.")
    st.markdown("#### What this means")
    st.markdown("""
- **MAE** is the average absolute prediction error; smaller is better.
- **R²** indicates how well predictions explain variation in the held-out data; a negative value means the model performed worse than a simple mean-baseline on that split.
- The 40/30/30 balanced score is project-defined and depends on the candidates and normalization used.
- The gain units/reference should be confirmed from the original simulation export.
- Radiation data currently include two principal-plane cuts, not a complete 3D pattern.
- Do not claim physical experimental validation unless physical measurements were performed.
""")

st.divider()
st.caption("ML-assisted decision support • Predictions and simulation-derived evidence are kept distinct • Verify all results against source data before final submission")
