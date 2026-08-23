import streamlit as st
import pandas as pd
import numpy as np
import os
import re
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
from docx import Document

st.set_page_config(page_title="Multi-Metric NODDI + Blast Dashboard", layout="wide")

# ============================
# PATHS
# ============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BLAST_FILE = os.path.join(BASE_DIR, "blastforunc.txt")
DOCUMENT_FILE = os.path.join(BASE_DIR, "NODDI_Diffusion_MRI_Dashboard.docx")

# ============================
# METRIC FILES + DEFINITIONS
# ============================

METRIC_FILES = {
    "ODI": "individual_ODI_complete.csv",
    "NDI": "individual_NDI_complete.csv",

    "AIC": "individual_AIC_complete.csv",
    "AICc": "individual_AICc_complete.csv",
    "BIC": "individual_BIC_complete.csv",
    "LogLikelihood": "individual_LogLikelihood_complete.csv",

    "EC_kappa": "individual_NODDI_EC.kappa_complete.csv",
    "EC_phi": "individual_NODDI_EC.phi_complete.csv",
    "EC_theta": "individual_NODDI_EC.theta_complete.csv",
    "EC_vec0": "individual_NODDI_EC.vec0_complete.csv",
    "EC_dperp0": "individual_NODDI_EC.dperp0_complete.csv",

    "IC_kappa": "individual_NODDI_IC.kappa_complete.csv",
    "IC_phi": "individual_NODDI_IC.phi_complete.csv",
    "IC_theta": "individual_NODDI_IC.theta_complete.csv",
    "IC_vec0": "individual_NODDI_IC.vec0_complete.csv",

    "w_csf": "individual_w_csf.w_complete.csv",
    "w_ec": "individual_w_ec.w_complete.csv",
    "w_ic": "individual_w_ic.w_complete.csv",
}

METRIC_DESCRIPTIONS = {
    "ODI": "Orientation Dispersion Index – angular variation of neurites.",
    "NDI": "Neurite Density Index – intracellular tissue fraction.",

    "EC_kappa": "Extracellular compartment dispersion (kappa).",
    "EC_phi": "Extracellular azimuthal angle (phi).",
    "EC_theta": "Extracellular polar angle (theta).",
    "EC_vec0": "Extracellular primary fiber direction.",
    "EC_dperp0": "Extracellular perpendicular diffusivity.",

    "IC_kappa": "Intracellular compartment dispersion.",
    "IC_phi": "Intracellular azimuthal angle.",
    "IC_theta": "Intracellular polar angle.",
    "IC_vec0": "Intracellular primary fiber direction.",

    "w_csf": "CSF free-water fraction.",
    "w_ec": "Extracellular free-water fraction.",
    "w_ic": "Intracellular free-water fraction.",

    "AIC": "Akaike Information Criterion.",
    "AICc": "Corrected Akaike Information Criterion.",
    "BIC": "Bayesian Information Criterion.",
    "LogLikelihood": "Model log-likelihood."
}

# ============================
# NORMALIZATION
# ============================

def normalize_subject_id(s):
    if pd.isna(s):
        return s
    s = str(s).strip()
    if s.endswith(".zip"):
        s = s[:-4]
    return s

# ============================
# DATA LOADING
# ============================

@st.cache_data
def load_metric_data(metric_name):
    df = pd.read_csv(os.path.join(BASE_DIR, METRIC_FILES[metric_name]))
    # first column is subjectid
    df = df.rename(columns={df.columns[0]: "subjectid"})

    # first row holds ROI names
    roi_row = df.iloc[0]
    roi_cols = [c for c in df.columns if c not in ["subjectid", "scantype"]]
    roi_names = {c: str(roi_row[c]).strip() for c in roi_cols}

    # drop the ROI-name row
    df = df.iloc[1:].copy()
    df["subjectid"] = df["subjectid"].astype(str).str.strip()

    for c in roi_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["subnorm"] = df["subjectid"].apply(normalize_subject_id)
    return df, roi_names, roi_cols

@st.cache_data
def load_blast_data():
    blast = pd.read_csv(BLAST_FILE, sep=r"\s+")
    blast["subjectid"] = blast["subjectid"].astype(str).str.strip()
    blast["age"] = pd.to_numeric(blast["age"], errors="coerce")
    blast["timeofexposure"] = pd.to_numeric(blast["timeofexposure"], errors="coerce")
    blast["subnorm"] = blast["subjectid"].apply(normalize_subject_id)
    return blast

def merge_metric_blast(metric_name):
    if metric_name not in METRIC_FILES:
        st.error(
            f"Unrecognized metric: '{metric_name}'. "
            f"Available metrics are: {list(METRIC_FILES.keys())}"
        )
        return None, None, None

    df, roi_names, roi_cols = load_metric_data(metric_name)
    blast = load_blast_data()
    merged = df.merge(
        blast[["subnorm", "age", "timeofexposure"]],
        on="subnorm",
        how="inner"
    )
    return merged, roi_names, roi_cols

# ============================
# PARTIAL CORRELATION
# ============================

def partial_corr(df, x, y, covariates):
    X = sm.add_constant(df[covariates])
    rx = sm.OLS(df[x], X, missing="drop").fit().resid
    ry = sm.OLS(df[y], X, missing="drop").fit().resid
    return stats.pearsonr(rx, ry)

# ============================
# TASK + METRIC PARSER
# ============================

def parse_task_and_metric(question: str):
    """
    Returns (metric_guess, task)

    task ∈ {"max_age_corrected", "age_corrected",
            "max_corr", "simple_corr",
            "global_significant_exposure",
            "significant_age_corrected",
            "show_intro_methods", None}
    """
    q = question.lower().strip()

    # ---- Document display command ----
    # Handle this before metric parsing because it does not require imaging data.
    if "introduction" in q and ("methods" in q or "method" in q):
        return None, "show_intro_methods"

    # ---- Guess metric from text ----
    metric_guess = None
    for m in METRIC_FILES:
        if m.lower() in q:
            metric_guess = m
            break

    # Default to ODI if user says "NODDI" or "ODI" but we didn't pick it up earlier
    if metric_guess is None:
        if "odi" in q or "noddi" in q:
            metric_guess = "ODI"

    # ---- Common query features ----
    has_all_regions = any(
        x in q
        for x in [
            "all regions",
            "all brain regions",
            "all noddi regions",
            "which regions",
            "which brain regions",
        ]
    )
    has_significant = any(
        x in q for x in ["significant", "significantly", "sig "]
    )
    has_max = any(
        x in q for x in ["highest", "strongest", "max", "maximum"]
    )
    has_corr = "correlation" in q or "correlat" in q
    has_age = any(
        x in q
        for x in [
            "age",
            "age-corrected",
            "age corrected",
            "corrected for age",
            "controlling for age",
            "control for age",
            "adjusted for age",
            "age-adjusted",
            "age adjusted",
        ]
    )
    has_blast = any(
        x in q for x in ["blast", "exposure", "timeofexposure"]
    )

    # IMPORTANT: check the age-corrected global query BEFORE the ordinary
    # global-significant query, because an age-corrected question also
    # contains "all regions", "significant", and "exposure".
    if has_all_regions and has_significant and has_age and has_blast:
        if metric_guess is None:
            metric_guess = "ODI"
        return metric_guess, "significant_age_corrected"

    # ---- Global "all regions significantly correlated with exposure" ----
    if has_all_regions and has_significant and has_blast:
        if metric_guess is None:
            metric_guess = "ODI"
        return metric_guess, "global_significant_exposure"

    # ---- Other patterns ----
    if has_max and has_age and has_blast:
        task = "max_age_corrected"
    elif has_age and has_blast:
        task = "age_corrected"
    elif has_max and has_blast:
        task = "max_corr"
    elif has_corr and has_blast:
        task = "simple_corr"
    else:
        task = None

    return metric_guess, task

# ============================
# WORD DOCUMENT DISPLAY
# ============================

def display_introduction_and_methods(docx_file):
    """Read and display the Introduction and Methods sections of a Word document."""

    if not os.path.exists(docx_file):
        st.error(f"Could not find document: {docx_file}")
        return

    try:
        doc = Document(docx_file)
    except Exception as e:
        st.error(f"Could not read {docx_file}: {e}")
        return

    introduction_paragraphs = []
    methods_paragraphs = []
    current_section = None

    introduction_headings = {
        "introduction",
        "1. introduction",
        "1 introduction",
    }

    methods_headings = {
        "methods",
        "method",
        "2. methods",
        "2 methods",
        "2. method",
        "2 method",
        "materials and methods",
        "2. materials and methods",
        "2 materials and methods",
    }

    methods_stop_headings = {
        "results",
        "3. results",
        "3 results",
        "discussion",
        "3. discussion",
        "3 discussion",
        "4. discussion",
        "4 discussion",
        "conclusion",
        "conclusions",
        "references",
        "acknowledgments",
        "acknowledgements",
    }

    for para in doc.paragraphs:
        text = para.text.strip()

        if not text:
            continue

        text_key = re.sub(r"\s+", " ", text.lower()).strip().rstrip(":")
        style_name = ""
        if para.style is not None and para.style.name:
            style_name = para.style.name.lower()

        if text_key in introduction_headings:
            current_section = "introduction"
            continue

        if text_key in methods_headings:
            current_section = "methods"
            continue

        if current_section == "methods" and text_key in methods_stop_headings:
            current_section = None
            break

        item = {
            "text": text,
            "style": style_name,
        }

        if current_section == "introduction":
            introduction_paragraphs.append(item)
        elif current_section == "methods":
            methods_paragraphs.append(item)

    st.title("Introduction and Methods")
    st.caption(f"Source: {os.path.basename(docx_file)}")

    st.header("Introduction")
    if introduction_paragraphs:
        for item in introduction_paragraphs:
            if item["style"].startswith("heading"):
                st.subheader(item["text"])
            else:
                st.write(item["text"])
    else:
        st.warning("Could not find an Introduction section in the document.")

    st.header("Methods")
    if methods_paragraphs:
        for item in methods_paragraphs:
            if item["style"].startswith("heading"):
                st.subheader(item["text"])
            else:
                st.write(item["text"])
    else:
        st.warning("Could not find a Methods section in the document.")

# ============================
# FUZZY 6-CHAR REGION MATCHING
# ============================

def fuzzy_region_match(question, region_cols, roi_names):
    q = question.lower().strip()
    raw_tokens = re.findall(r"[a-zA-Z_]+", q)

    tokens = []
    for t in raw_tokens:
        tc = re.sub(r"[^a-zA-Z]", "", t).lower()
        if len(tc) >= 4:
            tokens.append(tc)

    region_candidates = set()

    for t in tokens:
        prefix = t[:6]

        for col in region_cols:
            col_lower = col.lower()
            col_key = col_lower[:6]

            roi_full = roi_names.get(col, "")
            roi_lower = roi_full.lower()
            roi_key = roi_lower[:6]

            if (
                prefix == col_key
                or prefix == roi_key
                or prefix in col_lower
                or prefix in roi_lower
                or t in roi_lower
            ):
                region_candidates.add(col)

    return sorted(region_candidates, key=lambda c: roi_names.get(c, c))

# ============================
# PLOTTING
# ============================

def scatter_with_stats(x, y, r, p, title, xlabel):
    st.markdown(f"**r = {r:.4f}   p = {p:.4e}   N = {len(x)}**")

    fig, ax = plt.subplots()
    ax.scatter(x, y)

    if len(x) >= 2:
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, slope * xs + intercept)

    ax.text(
        0.05, 0.95,
        f"r={r:.3f}\np={p:.2e}",
        transform=ax.transAxes,
        va="top"
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Blast Exposure (timeofexposure)")
    ax.set_title(title)
    st.pyplot(fig)

# ============================
# UI
# ============================

st.title("Multi-Metric NODDI + Blast Exposure Dashboard")

# ---- Sidebar: manual metric selection ----
metric_sidebar = st.sidebar.selectbox("Select Metric (Manual)", list(METRIC_FILES.keys()))
if metric_sidebar in METRIC_DESCRIPTIONS:
    st.sidebar.info(METRIC_DESCRIPTIONS[metric_sidebar])

df_sidebar, roi_names_sidebar, region_cols_sidebar = merge_metric_blast(metric_sidebar)

st.subheader("Ask a Question")

# ---- Example questions with larger font ----
st.markdown("""
<div style="font-size:20px;">
<b>Example questions:</b><br><br>
• What is the correlation between NODDI ODI and blast exposure for the CINGULATE_GYRUS_left?<br>
• What is the age-corrected correlation between ODI and timeofexposure for the THALAMUS_right?<br>
• Which brain region shows the strongest correlation between ODI with blast exposure?<br>
• Show me all regions where ODI is significantly correlated with exposure?<br>
• Show me all regions where ODI is significantly correlated with exposure after correcting for age?<br>
• What are the available brain regions?<br>
• What are the available NODDI metrics?<br>
• Show me the introduction and methods<br>
</div>
""", unsafe_allow_html=True)

question = st.text_input("Ask a question about the data:")

# ---- Direct document command ----
# Handle this before the general question parser so it cannot be
# misclassified as a brain-region or correlation question.
if question:
    q_direct = question.lower().strip()
    if (
        "introduction" in q_direct
        and ("methods" in q_direct or "method" in q_direct)
    ):
        display_introduction_and_methods(DOCUMENT_FILE)
        st.stop()

# ---- Metric listing ----
if question and ("available noddi metrics" in question.lower() or "list metrics" in question.lower()):
    st.subheader("Available NODDI Metrics")
    dfm = pd.DataFrame(
        [(k, METRIC_DESCRIPTIONS.get(k, "")) for k in METRIC_FILES],
        columns=["Metric", "Description"]
    )
    st.dataframe(dfm, use_container_width=True)
    st.stop()
# ---- Brain region listing ----
if question and (
    "available brain regions" in question.lower()
    or "list brain regions" in question.lower()
    or "what are the available brain regions" in question.lower()
    or "what brain regions are available" in question.lower()
):
    if df_sidebar is None or roi_names_sidebar is None or region_cols_sidebar is None:
        st.error("Could not load data for the selected metric.")
        st.stop()

    st.subheader(f"Available brain regions for metric: {metric_sidebar}")

    regions_df = pd.DataFrame(
        {
            "Region Column": region_cols_sidebar,
            "Region Name": [roi_names_sidebar.get(c, c) for c in region_cols_sidebar],
        }
    )
    st.dataframe(regions_df, use_container_width=True)
    st.caption(
        "Region Column = column name in the CSV; "
        "Region Name = human-readable label taken from the first row of the metric file."
    )
    st.stop()

# ============================
# MAIN QUESTION HANDLING
# ============================

if question:
    # 1) Parse task + metric
    metric_guess, task = parse_task_and_metric(question)

    # Special document command: no imaging metric or CSV loading is needed.
    if task == "show_intro_methods":
        display_introduction_and_methods(DOCUMENT_FILE)
        st.stop()

    if task is None:
        st.warning(
            "The question parser could not confidently map your question. "
            "Try mentioning a brain region (e.g., 'cingul', 'thalam', 'hippoc') "
            "or ask for 'all regions where ODI is significantly correlated with exposure'."
        )
        st.stop()

    # Fallback metric: use sidebar choice if none inferred
    if metric_guess is None:
        metric_guess = metric_sidebar

    # 2) Load data for that metric
    df_q, roi_names_q, region_cols_q = merge_metric_blast(metric_guess)
    if df_q is None:
        st.stop()

    if metric_guess in METRIC_DESCRIPTIONS:
        st.info(f"**{metric_guess}** — {METRIC_DESCRIPTIONS[metric_guess]}")

    # 3) Special case: global all-regions significant correlation with exposure
    if task == "global_significant_exposure":
        st.subheader("Regions where metric is significantly correlated with blast exposure")

        results = []
        for col in region_cols_q:
            sub = df_q[[col, "timeofexposure"]].replace([np.inf, -np.inf], np.nan).dropna()
 # TEMPORARY: exclude subjects with zero blast exposure
            sub = sub[sub["timeofexposure"] > 0]

            if len(sub) < 5:
                continue
            r, p = stats.pearsonr(sub[col], sub["timeofexposure"])
            results.append({
                "Region Column": col,
                "Region Name": roi_names_q.get(col, col),
                "r": r,
                "p": p,
                "n": len(sub)
            })

        if not results:
            st.warning("Not enough data to compute correlations for any region.")
            st.stop()

        res_df = pd.DataFrame(results)
        alpha = 0.05
        sig_df = res_df[res_df["p"] < alpha].sort_values("p")
        if sig_df.empty:
            st.info(
                f"No regions showed a statistically significant correlation "
                f"(p < {alpha})."
            )

        else:
            st.write(f"Regions with significant correlation (p < {alpha}):")
            st.caption("Click a row below to display the correlation plot.")

            display_df = sig_df.reset_index(drop=True)

            event = st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row",
                key="significant_regions_table",
            )

            # ------------------------------------------------------------
            # Plot selected region
            # ------------------------------------------------------------
            if event.selection.rows:
                selected_row = event.selection.rows[0]

                selected_col = display_df.iloc[selected_row]["Region Column"]
                selected_name = display_df.iloc[selected_row]["Region Name"]

                sub = (
                    df_q[[selected_col, "timeofexposure"]]
                    .replace([np.inf, -np.inf], np.nan)
                    .dropna()
                )
                sub = sub[sub["timeofexposure"] > 0]
                if len(sub) >= 5:
                    r, p = stats.pearsonr(
                        sub[selected_col],
                        sub["timeofexposure"],
                    )

                    st.subheader(
                        f"{metric_guess} vs. Blast Exposure: {selected_name}"
                    )

                    fig, ax = plt.subplots(figsize=(8, 6))

                    ax.scatter(
                        sub["timeofexposure"],
                        sub[selected_col],
                        s=60,
                        alpha=0.75,
                    )

                    # Linear regression line
                    slope, intercept = np.polyfit(
                        sub["timeofexposure"],
                        sub[selected_col],
                        1,
                    )

                    xline = np.linspace(
                        sub["timeofexposure"].min(),
                        sub["timeofexposure"].max(),
                        100,
                    )

                    yline = slope * xline + intercept

                    ax.plot(
                        xline,
                        yline,
                        linewidth=2,
                    )

                    ax.set_xlabel("Blast Exposure")
                    ax.set_ylabel(metric_guess)

                    ax.set_title(
                        f"{selected_name}\n"
                        f"Pearson r = {r:.3f}, "
                        f"p = {p:.4g}, "
                        f"n = {len(sub)}"
                    )

                    ax.grid(alpha=0.25)

                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                    st.write(
                        f"**Region:** {selected_name}  \n"
                        f"**Metric:** {metric_guess}  \n"
                        f"**Pearson r:** {r:.3f}  \n"
                        f"**p-value:** {p:.4g}  \n"
                        f"**n:** {len(sub)}"
                    )

                else:
                    st.warning(
                        "Not enough data to plot this region."
                    )

        st.stop()
    # 4) For other tasks, possibly need region info
    region_candidates = fuzzy_region_match(question, region_cols_q, roi_names_q)

    chosen_region = None

    # For tasks that use a single region, let the user pick if ambiguous
    if task not in ["max_corr", "max_age_corrected","significant_age_corrected"]:
        if len(region_candidates) == 1:
            chosen_region = region_candidates[0]
        elif len(region_candidates) > 1:
            chosen_region = st.selectbox(
                "Multiple regions matched. Please choose one:",
                region_candidates,
                format_func=lambda c: roi_names_q.get(c, c)
            )
        else:
            st.warning(
                "No matching brain regions found for that question. "
                "Try using a distinctive part of the region name "
                "(e.g., 'hippoc', 'cingul', 'thalam')."
            )
            st.stop()

    # ============================================================
    # ALL SIGNIFICANT AGE-CORRECTED REGIONS
    # ============================================================

    if task == "significant_age_corrected":

        results = []

        for col in region_cols_q:
            sub = (
                df_q[[col, "timeofexposure", "age"]]
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )

            # TEMPORARY: exclude subjects with zero blast exposure
            sub = sub[sub["timeofexposure"] > 0]

            if len(sub) < 5:
                continue

            # Partial correlation controlling for age
            r, p = partial_corr(
                sub,
                col,
                "timeofexposure",
                ["age"]
            )

            results.append({
                "Region Column": col,
                "Region Name": roi_names_q.get(col, col),
                "r": r,
                "p": p,
                "N": len(sub),
            })

        if not results:
            st.warning(
                "Not enough data to compute age-corrected "
                "correlations for any region."
            )
            st.stop()

        res_df = pd.DataFrame(results)

        alpha = 0.05

        sig_df = (
            res_df[res_df["p"] < alpha]
            .sort_values("p")
            .reset_index(drop=True)
        )

        if sig_df.empty:
            st.info(
                f"No regions showed a statistically significant "
                f"age-corrected correlation (p < {alpha})."
            )

        else:
            st.write(
                f"Regions with significant age-corrected "
                f"correlation (p < {alpha}):"
            )

            st.caption(
                "Partial Pearson correlation controlling for age. "
                "Click a row below to display the age-corrected plot."
            )

            event = st.dataframe(
                sig_df,
                use_container_width=True,
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row",
                key="significant_age_corrected_regions_table",
            )

            # ------------------------------------------------------------
            # Plot selected region
            # ------------------------------------------------------------

            if event.selection.rows:
                selected_row = event.selection.rows[0]

                selected_col = sig_df.iloc[selected_row]["Region Column"]
                selected_name = sig_df.iloc[selected_row]["Region Name"]

                sub = (
                    df_q[
                        [
                            selected_col,
                            "timeofexposure",
                            "age",
                        ]
                    ]
                    .replace([np.inf, -np.inf], np.nan)
                    .dropna()
                )

                # TEMPORARY: exclude subjects with zero blast exposure
                sub = sub[sub["timeofexposure"] > 0]

                if len(sub) >= 5:

                    # ----------------------------------------------------
                    # Partial correlation controlling for age
                    # ----------------------------------------------------

                    r, p = partial_corr(
                        sub,
                        selected_col,
                        "timeofexposure",
                        ["age"]
                    )

                    # ----------------------------------------------------
                    # Remove age effect from brain metric
                    # ----------------------------------------------------

                    slope_y, intercept_y, _, _, _ = stats.linregress(
                        sub["age"],
                        sub[selected_col]
                    )

                    metric_resid = (
                        sub[selected_col]
                        - (
                            intercept_y
                            + slope_y * sub["age"]
                        )
                    )

                    # ----------------------------------------------------
                    # Remove age effect from blast exposure
                    # ----------------------------------------------------

                    slope_x, intercept_x, _, _, _ = stats.linregress(
                        sub["age"],
                        sub["timeofexposure"]
                    )

                    exposure_resid = (
                        sub["timeofexposure"]
                        - (
                            intercept_x
                            + slope_x * sub["age"]
                        )
                    )

                    # ----------------------------------------------------
                    # Plot residualized values
                    # ----------------------------------------------------

                    st.subheader(
                        f"{metric_guess} vs. Blast Exposure: "
                        f"{selected_name}"
                    )

                    st.caption(
                        "Age-corrected partial correlation. "
                        "Both axes show residuals after removing "
                        "the linear effect of age."
                    )

                    fig, ax = plt.subplots(figsize=(8, 6))

                    ax.scatter(
                        exposure_resid,
                        metric_resid,
                        s=60,
                        alpha=0.75,
                    )

                    # Linear regression line
                    slope, intercept = np.polyfit(
                        exposure_resid,
                        metric_resid,
                        1,
                    )

                    xline = np.linspace(
                        exposure_resid.min(),
                        exposure_resid.max(),
                        100,
                    )

                    yline = slope * xline + intercept

                    ax.plot(
                        xline,
                        yline,
                        linewidth=2,
                    )

                    ax.set_xlabel(
                        "Blast Exposure (age-corrected residual)"
                    )

                    ax.set_ylabel(
                        f"{metric_guess} (age-corrected residual)"
                    )

                    ax.set_title(
                        f"{selected_name}\n"
                        f"Age-corrected r = {r:.3f}, "
                        f"p = {p:.4g}, "
                        f"n = {len(sub)}"
                    )

                    ax.grid(alpha=0.25)

                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                    st.write(
                        f"**Region:** {selected_name}  \n"
                        f"**Metric:** {metric_guess}  \n"
                        f"**Partial Pearson r:** {r:.3f}  \n"
                        f"**p-value:** {p:.4g}  \n"
                        f"**Covariate:** Age  \n"
                        f"**n:** {len(sub)}"
                    )

                else:
                    st.warning(
                        "Not enough data to plot this region."
                    )

        st.stop()

 
    # ========================
    # MAX SIMPLE CORRELATION
    # ========================

    if task == "max_corr":
        best = None
        for col in region_cols_q:
            sub = (
                df_q[[col, "timeofexposure"]]
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )

# TEMPORARY: exclude subjects with zero blast exposure
            sub = sub[sub["timeofexposure"] > 0]
            if len(sub) < 5:
                continue
            r, p = stats.pearsonr(sub[col], sub["timeofexposure"])
            if best is None or abs(r) > abs(best[1]):
                best = (col, r, p, sub)

        if best:
            col, r, p, sub = best
            scatter_with_stats(
                sub[col].to_numpy(),
                sub["timeofexposure"].to_numpy(),
                r, p,
                f"{metric_guess} — {roi_names_q.get(col, col)} (MAX correlation)",
                roi_names_q.get(col, col)
            )
        else:
            st.warning("Not enough data to compute max correlation across regions.")

    # ========================
    # MAX AGE-CORRECTED
    # ========================

    elif task == "max_age_corrected":
        best = None
        for col in region_cols_q:
            sub = df_q[[col, "timeofexposure", "age"]].dropna()
            if len(sub) < 5:
                continue
            r, p = partial_corr(sub, col, "timeofexposure", ["age"])
            if best is None or abs(r) > abs(best[1]):
                best = (col, r, p, sub)

        if best:
            col, r, p, sub = best
            scatter_with_stats(
                sub[col].to_numpy(),
                sub["timeofexposure"].to_numpy(),
                r, p,
                f"{metric_guess} — {roi_names_q.get(col, col)} (MAX age-corrected)",
                roi_names_q.get(col, col)
            )
        else:
            st.warning("Not enough data to compute max age-corrected correlation across regions.")

    # ========================
    # SINGLE-REGION ANALYSIS
    # ========================

    elif chosen_region is not None:

        if task == "age_corrected":
            sub = df_q[[chosen_region, "timeofexposure", "age"]].dropna()
            if len(sub) >= 5:
                r, p = partial_corr(sub, chosen_region, "timeofexposure", ["age"])
                scatter_with_stats(
                    sub[chosen_region].to_numpy(),
                    sub["timeofexposure"].to_numpy(),
                    r, p,
                    f"{metric_guess} — {roi_names_q.get(chosen_region, chosen_region)} (age-corrected)",
                    roi_names_q.get(chosen_region, chosen_region)
                )
            else:
                st.warning("Not enough data for age-corrected correlation in that region.")

        elif task == "simple_corr":
            sub = df_q[[chosen_region, "timeofexposure"]].dropna()
            if len(sub) >= 5:
                r, p = stats.pearsonr(sub[chosen_region], sub["timeofexposure"])
                scatter_with_stats(
                    sub[chosen_region].to_numpy(),
                    sub["timeofexposure"].to_numpy(),
                    r, p,
                    f"{metric_guess} — {roi_names_q.get(chosen_region, chosen_region)}",
                    roi_names_q.get(chosen_region, chosen_region)
                )
            else:
                st.warning("Not enough data for simple correlation in that region.")

