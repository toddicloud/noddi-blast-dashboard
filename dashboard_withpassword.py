import streamlit as st
import pandas as pd
import numpy as np
import os
import re
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm

import streamlit as st

# ============================
# PASSWORD PROTECTION
# ============================

def check_password():
    def password_entered():
        if st.session_state["password"] == st.secrets["app_password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input(
            "Enter password to access the dashboard:",
            type="password",
            on_change=password_entered,
            key="password",
        )
        return False
    elif not st.session_state["password_correct"]:
        st.text_input(
            "Enter password to access the dashboard:",
            type="password",
            on_change=password_entered,
            key="password",
        )
        st.error("❌ Incorrect password")
        return False
    else:
        return True


if not check_password():
    st.stop()

st.set_page_config(page_title="NODDI ODI + Blast Exposure Dashboard", layout="wide")

# ============================
# PATHS (FROM dash1.py — CORRECT)
# ============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ODI_FILE = os.path.join(BASE_DIR, "individual_ODI_complete.csv")
BLAST_FILE = os.path.join(BASE_DIR, "blastforunc.txt")

# ============================
# NORMALIZATION (FROM dash1.py)
# ============================

def normalize_subject_id(s):
    """Strip whitespace and .zip; keep trailing underscore as-is."""
    if pd.isna(s):
        return s
    s = str(s).strip()
    if s.endswith(".zip"):
        s = s[:-4]
    return s

# ============================
# DATA LOADING (REPLACED WITH dash1.py VERSION)
# ============================

@st.cache_data
def load_odi_data(odi_path=ODI_FILE):
    """
    Load individual_ODI_complete.csv and return:
      - odi_df: cleaned numeric data (one row per subject)
      - roi_names: dict {column_name: ROI_name}
    """
    df = pd.read_csv(odi_path)

    # Rename subjectid column (it has a leading space)
    orig_sub_col = df.columns[0]
    df = df.rename(columns={orig_sub_col: "subjectid"})

    # Row 0 contains ROI labels for the region columns
    roi_row = df.iloc[0]

    # ROI columns = everything except subjectid & scantype
    roi_cols = [c for c in df.columns if c not in ["subjectid", "scantype"]]

    roi_names = {col: str(roi_row[col]).strip() for col in roi_cols}

    # Drop the ROI-label row to get numeric data
    df = df.iloc[1:].copy()

    # Clean subjectid strings
    df["subjectid"] = df["subjectid"].astype(str).str.strip()

    # Convert ROI columns to numeric
    for col in roi_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Add a normalized subject id for merging
    df["subnorm"] = df["subjectid"].apply(normalize_subject_id)

    return df, roi_names


@st.cache_data
def load_blast_data(blast_path=BLAST_FILE):
    """Load blastforunc.txt (subjectid age timeofexposure)."""
    blast = pd.read_csv(blast_path, delim_whitespace=True)

    blast["subjectid"] = blast["subjectid"].astype(str).str.strip()
    blast["age"] = pd.to_numeric(blast["age"], errors="coerce")
    blast["timeofexposure"] = pd.to_numeric(blast["timeofexposure"], errors="coerce")

    blast["subnorm"] = blast["subjectid"].apply(normalize_subject_id)

    return blast


@st.cache_data
def merge_odi_blast():
    odi_df, roi_names = load_odi_data()
    blast_df = load_blast_data()

    merged = odi_df.merge(
        blast_df[["age", "timeofexposure", "subnorm"]],
        on="subnorm",
        how="inner",
    )
    return merged, roi_names

# ============================
# PARTIAL CORRELATION (FROM dash1.py)
# ============================

def partial_corr(df, x, y, covariates):
    """
    Compute partial correlation between x and y controlling for covariates.
    """
    X_cov = df[covariates]
    X = sm.add_constant(X_cov)

    res_x = sm.OLS(df[x], X, missing="drop").fit().resid
    res_y = sm.OLS(df[y], X, missing="drop").fit().resid

    r, p = stats.pearsonr(res_x, res_y)
    return r, p

# ============================
# LOAD DATA
# ============================

try:
    df, roi_names = merge_odi_blast()
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

st.title("NODDI ODI + Blast Exposure Dashboard")
st.write("This dashboard joins NODDI ODI values with blast exposure and age for correlation analysis.")

st.sidebar.header("Data overview")
st.sidebar.write(f"Number of ODI subjects: {df['subjectid'].nunique()}")
st.sidebar.write(f"Number of ROI columns: {len(roi_names)}")

    # Show available ROI sets (simple heuristic: those whose name contains left/right, etc)
st.subheader("Available brain regions (first 50)")
roi_list = list(roi_names.values())
st.write(pd.DataFrame({"ROI name": roi_list}).head(50))

# ============================
# REGION COLUMNS (FROM roi_names)
# ============================

region_cols = list(roi_names.keys())

# ============================
# QUESTION PARSER (FUZZY 6-CHAR VERSION)
# ============================

def parse_question_simple(question, region_cols):
    q = question.lower().strip()

    # ----------------------------
    # TASK DETECTION
    # ----------------------------
    task = None

    # ✅ NEW: highest/max correlation across all regions
    if any(x in q for x in [
        "highest", "strongest", "max", "maximum"
    ]) and any(x in q for x in [
        "correlation", "corelation"
    ]):
        if any(x in q for x in [
            "age-corrected", "age corrected", "age-correct", "age-correction",
            "controlling for age", "partial"
        ]):
            task = "max_age_corrected_correlation"
        else:
            task = "max_correlation"

    elif any(x in q for x in [
        "age-corrected", "age corrected", "age-correct", "age-correction",
        "controlling for age", "partial"
    ]) and any(x in q for x in [
        "blast", "timeofexposure", "exposure"
    ]):
        task = "age_corrected_correlation"

    elif "correlation" in q or "corelation" in q:
        task = "simple_correlation"

    # ----------------------------
    # REGION FUZZY MATCH (unchanged)
    # ----------------------------
    region_candidates = []

    tokens = re.findall(r"[a-zA-Z_]+", q)
    tokens = [t for t in tokens if len(t) >= 4]

    region_key = None
    if tokens:
        region_key = max(tokens, key=len).lower()[:6]

    if region_key:
        for col in region_cols:
            col_key = col.lower()[:6]
            roi_key = roi_names.get(col, "").lower()[:6]
            full_roi = roi_names.get(col, "").lower()

            if (
                col_key == region_key
                or roi_key == region_key
                or region_key in col.lower()
                or region_key in full_roi
            ):
                region_candidates.append(col)

    return task, region_candidates

# ============================
# MANUAL REGION ANALYSIS
# ============================

st.subheader("Manual Region-wise Analysis")
selected_region_manual = st.selectbox(
    "Select a brain region:",
    region_cols,
    format_func=lambda c: roi_names.get(c, c)
)

if st.button("Run Manual Analysis"):
    x = df[selected_region_manual]
    y = df["timeofexposure"]
    z = df["age"]

    r_xy = np.corrcoef(x, y)[0, 1]
    r_xz = np.corrcoef(x, z)[0, 1]
    r_yz = np.corrcoef(y, z)[0, 1]

    r_partial = (r_xy - r_xz * r_yz) / np.sqrt((1 - r_xz**2) * (1 - r_yz**2))

    t_stat = r_partial * np.sqrt((len(x) - 3) / (1 - r_partial**2))
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(x) - 3))

    st.write(f"Partial r (controlling for age): {r_partial:.4f}")
    st.write(f"p-value: {p_val:.4e}")

    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.set_xlabel(roi_names.get(selected_region_manual, selected_region_manual))
    ax.set_ylabel("Blast Exposure")
    st.pyplot(fig)

# ============================
# NATURAL LANGUAGE QUESTION SECTION
# ============================

st.subheader("Ask a Question About This Data")
question = st.text_input(
        "Example: 'What is the correlation between NODDI ODI and blast exposure with age-correction for brain region CINGULATE_GYRUS_left?'""  Enter your question:"
    )


if question:
    task, region_candidates = parse_question_simple(question, region_cols)

    st.write("### Parsing Result")
    st.write("Detected task:", task)
    st.write("Region candidates:", region_candidates)
    st.write("Raw question:", question)

    region = None

    if region_candidates:
        if len(region_candidates) == 1:
            region = region_candidates[0]
            st.success(f"Auto-selected region: {roi_names.get(region, region)}")
        else:
            st.warning("Multiple possible brain regions found:")
            region = st.selectbox(
                "Please choose the correct brain region:",
                region_candidates,
                format_func=lambda c: roi_names.get(c, c)
            )
    else:
        st.error("No matching brain regions found.")

    # ✅ NEW: HIGHEST SIMPLE CORRELATION ACROSS ALL REGIONS
    if task == "max_correlation":
        results = []
        for col in region_cols:
            sub_df = df[[col, "timeofexposure"]].dropna()
            if len(sub_df) < 5:
                continue
            r, p = stats.pearsonr(sub_df[col], sub_df["timeofexposure"])
            results.append((col, r, p))

        if results:
            best_col, best_r, best_p = max(results, key=lambda x: abs(x[1]))

            st.success(f"Highest correlation region: {roi_names.get(best_col, best_col)}")
            st.write(f"r = {best_r:.4f}")
            st.write(f"p = {best_p:.4e}")

            fig, ax = plt.subplots()
            ax.scatter(df[best_col], df["timeofexposure"])
            ax.set_xlabel(roi_names.get(best_col, best_col))
            ax.set_ylabel("Blast Exposure")
            st.pyplot(fig)
        else:
            st.error("Not enough valid data to compute correlations.")

    # ✅ NEW: HIGHEST AGE-CORRECTED CORRELATION ACROSS ALL REGIONS
    elif task == "max_age_corrected_correlation":
        results = []
        for col in region_cols:
            sub_df = df[[col, "timeofexposure", "age"]].dropna()
            if len(sub_df) < 5:
                continue
            r, p = partial_corr(sub_df, col, "timeofexposure", ["age"])
            results.append((col, r, p))

        if results:
            best_col, best_r, best_p = max(results, key=lambda x: abs(x[1]))

            st.success(f"Highest AGE-CORRECTED correlation region: {roi_names.get(best_col, best_col)}")
            st.write(f"Partial r = {best_r:.4f}")
            st.write(f"p = {best_p:.4e}")

            fig, ax = plt.subplots()
            ax.scatter(df[best_col], df["timeofexposure"])
            ax.set_xlabel(roi_names.get(best_col, best_col))
            ax.set_ylabel("Blast Exposure")
            st.pyplot(fig)
        else:
            st.error("Not enough valid data to compute age-corrected correlations.")

    # ✅ EXISTING SINGLE-REGION AGE-CORRECTED CASE
    elif task == "age_corrected_correlation" and region is not None:
        sub_df = df[[region, "timeofexposure", "age"]].dropna()

        r, p = partial_corr(sub_df, region, "timeofexposure", ["age"])

        st.success(f"Partial r (ODI vs timeofexposure, controlling for age): {r:.4f}")
        st.write(f"p-value: {p:.4e}")

        fig, ax = plt.subplots()
        ax.scatter(sub_df[region], sub_df["timeofexposure"])
        ax.set_xlabel(roi_names.get(region, region))
        ax.set_ylabel("Blast Exposure")
        st.pyplot(fig)

    elif task is None:
        st.warning("The question parser could not confidently map your question to a known analysis.")

