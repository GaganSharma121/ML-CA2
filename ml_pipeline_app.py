"""
ML Pipeline Dashboard — Full Streamlit App
Run: streamlit run ml_pipeline_app.py
Requirements: streamlit plotly scikit-learn pandas numpy
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import time
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# PAGE CONFIG & GLOBAL STYLES
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="ML Pipeline Studio",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    /* ── base ── */
    [data-testid="stAppViewContainer"] { background: #0f1117; }
    [data-testid="stHeader"] { background: transparent; }
    section[data-testid="stSidebar"] { background: #161b27; }

    /* ── step pill strip ── */
    .step-strip {
        display: flex; align-items: flex-start; gap: 0;
        overflow-x: auto; padding: 1.4rem 0 1rem;
        scrollbar-width: none;
    }
    .step-strip::-webkit-scrollbar { display: none; }

    .step-item {
        display: flex; flex-direction: column; align-items: center;
        min-width: 110px; position: relative;
    }
    .step-item:not(:last-child)::after {
        content: '';
        position: absolute; top: 22px; left: calc(50% + 22px);
        width: calc(100% - 44px); height: 2px;
        background: #2a3040;
    }
    .step-item.done::after  { background: #4ade80; }
    .step-item.active::after { background: #4ade80; }

    .step-circle {
        width: 44px; height: 44px; border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-size: 16px; font-weight: 700; z-index: 1;
        border: 2px solid #2a3040; background: #1a2030; color: #6b7280;
        transition: all .3s;
    }
    .step-item.done  .step-circle { background: #166534; border-color: #4ade80; color: #4ade80; }
    .step-item.active .step-circle { background: #1d3a5f; border-color: #60a5fa; color: #60a5fa;
        box-shadow: 0 0 0 4px rgba(96,165,250,.2); }

    .step-label {
        margin-top: 8px; font-size: 11px; font-weight: 600;
        text-align: center; color: #6b7280; letter-spacing: .3px;
        max-width: 100px; line-height: 1.3;
    }
    .step-item.done  .step-label  { color: #4ade80; }
    .step-item.active .step-label { color: #60a5fa; }

    /* ── card panels ── */
    .panel {
        background: #161b27; border: 1px solid #1f2937;
        border-radius: 14px; padding: 1.4rem 1.6rem; margin-bottom: 1.2rem;
    }
    .panel-title {
        font-size: 15px; font-weight: 700; color: #e5e7eb;
        margin-bottom: .6rem; letter-spacing: .3px;
    }
    .kpi-row { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: .8rem; }
    .kpi-box {
        flex: 1; min-width: 110px; background: #1a2030;
        border: 1px solid #1f2937; border-radius: 10px;
        padding: .8rem 1rem; text-align: center;
    }
    .kpi-val { font-size: 22px; font-weight: 700; color: #60a5fa; }
    .kpi-lbl { font-size: 11px; color: #9ca3af; margin-top: 3px; }

    /* ── section header ── */
    .sec-header {
        font-size: 20px; font-weight: 800; color: #f9fafb;
        border-left: 4px solid #60a5fa; padding-left: 12px;
        margin-bottom: 1rem;
    }

    /* ── badge ── */
    .badge {
        display: inline-block; padding: 3px 10px; border-radius: 20px;
        font-size: 11px; font-weight: 700; letter-spacing: .4px;
    }
    .badge-blue  { background: #1d3a5f; color: #60a5fa; }
    .badge-green { background: #14532d; color: #4ade80; }
    .badge-amber { background: #451a03; color: #fbbf24; }
    .badge-red   { background: #450a0a; color: #f87171; }

    /* ── streamlit overrides ── */
    .stButton > button {
        border-radius: 8px !important; font-weight: 600 !important;
        padding: .45rem 1.2rem !important;
    }
    div[data-testid="stMetric"] label { font-size: 12px !important; }
    .stTabs [data-baseweb="tab"] { font-size: 13px; font-weight: 600; }
    .stDataFrame { border-radius: 10px; }
    [data-testid="stExpander"] { border: 1px solid #1f2937 !important; border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# SESSION STATE INIT
# ──────────────────────────────────────────────
STEPS = [
    "Problem Type", "Data Input", "EDA",
    "Data Engineering", "Feature Selection",
    "Data Split", "Model Selection",
    "Training & CV", "Performance",
    "Hyperparameter Tuning"
]
STEP_ICONS = ["🎯","📂","🔍","🛠️","📊","✂️","🤖","🏋️","📈","⚙️"]

defaults = dict(
    current_step=0, problem_type=None,
    df=None, df_clean=None, target_col=None,
    selected_features=None, outlier_mask=None,
    X_train=None, X_test=None, y_train=None, y_test=None,
    selected_model=None, trained_model=None, k_folds=5,
    test_size=0.2, scaler_choice="StandardScaler",
    encoding_done=False, outlier_method=None,
    feature_selection_method=None, selected_final_features=None,
    cv_results=None, tuned_model=None,
)
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def go_to(step): st.session_state.current_step = step
def next_step(): st.session_state.current_step = min(st.session_state.current_step + 1, len(STEPS)-1)
def prev_step(): st.session_state.current_step = max(st.session_state.current_step - 1, 0)


# ──────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────
st.markdown("""
<div style='padding:1.2rem 0 .4rem; text-align:center;'>
  <span style='font-size:32px; font-weight:900; color:#f9fafb; letter-spacing:-1px;'>
    🧠 ML Pipeline Studio
  </span><br>
  <span style='font-size:13px; color:#6b7280;'>End-to-end machine learning — from raw data to tuned model</span>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# STEP PROGRESS STRIP
# ──────────────────────────────────────────────
cur = st.session_state.current_step
strip_html = "<div class='step-strip'>"
for i, (label, icon) in enumerate(zip(STEPS, STEP_ICONS)):
    state = "active" if i == cur else ("done" if i < cur else "")
    strip_html += f"""
    <div class='step-item {state}'>
      <div class='step-circle'>{icon if i < cur else (icon if i == cur else str(i+1))}</div>
      <div class='step-label'>{label}</div>
    </div>"""
strip_html += "</div>"
st.markdown(strip_html, unsafe_allow_html=True)
st.markdown("<hr style='border-color:#1f2937; margin:.4rem 0 1.4rem;'>", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
PLOTLY_DARK = dict(
    template="plotly_dark",
    paper_bgcolor="#161b27",
    plot_bgcolor="#161b27",
    font=dict(family="Inter, sans-serif", color="#e5e7eb"),
)

def plotly_fig(fig, height=360):
    fig.update_layout(**PLOTLY_DARK, margin=dict(l=30,r=20,t=40,b=30), height=height)
    st.plotly_chart(fig, use_container_width=True)

def status_badge(text, kind="blue"):
    st.markdown(f"<span class='badge badge-{kind}'>{text}</span>", unsafe_allow_html=True)

def section_header(text):
    st.markdown(f"<div class='sec-header'>{text}</div>", unsafe_allow_html=True)

def nav_buttons(back=True, next_label="Next →", next_disabled=False):
    c1, _, c2 = st.columns([2,6,2])
    if back and st.session_state.current_step > 0:
        with c1:
            if st.button("← Back"):
                prev_step(); st.rerun()
    with c2:
        if st.button(next_label, disabled=next_disabled, type="primary"):
            next_step(); st.rerun()


# ──────────────────────────────────────────────
# STEP 0 — PROBLEM TYPE
# ──────────────────────────────────────────────
if cur == 0:
    section_header("Step 1 — Choose Problem Type")
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("""
        <div class='panel' style='border-color:#1d3a5f; cursor:pointer;'>
          <div style='font-size:36px; text-align:center;'>🏷️</div>
          <div style='font-size:17px; font-weight:800; color:#60a5fa; text-align:center; margin:.4rem 0 .3rem;'>Classification</div>
          <div style='font-size:13px; color:#9ca3af; text-align:center;'>Predict discrete labels or categories<br>
          <span class='badge badge-blue' style='margin-top:6px;'>Logistic Reg</span>
          <span class='badge badge-blue' style='margin-top:6px;'>SVM</span>
          <span class='badge badge-blue' style='margin-top:6px;'>Random Forest</span>
          </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Select Classification", use_container_width=True):
            st.session_state.problem_type = "Classification"
            next_step(); st.rerun()

    with col2:
        st.markdown("""
        <div class='panel' style='border-color:#14532d; cursor:pointer;'>
          <div style='font-size:36px; text-align:center;'>📉</div>
          <div style='font-size:17px; font-weight:800; color:#4ade80; text-align:center; margin:.4rem 0 .3rem;'>Regression</div>
          <div style='font-size:13px; color:#9ca3af; text-align:center;'>Predict continuous numeric values<br>
          <span class='badge badge-green' style='margin-top:6px;'>Linear Reg</span>
          <span class='badge badge-green' style='margin-top:6px;'>SVR</span>
          <span class='badge badge-green' style='margin-top:6px;'>Random Forest</span>
          </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Select Regression", use_container_width=True):
            st.session_state.problem_type = "Regression"
            next_step(); st.rerun()


# ──────────────────────────────────────────────
# STEP 1 — DATA INPUT
# ──────────────────────────────────────────────
elif cur == 1:
    section_header("Step 2 — Data Input & PCA Visualisation")
    pt_color = "blue" if st.session_state.problem_type == "Classification" else "green"
    status_badge(f"Problem: {st.session_state.problem_type}", pt_color)
    st.markdown("<br>", unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload CSV dataset", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        df.columns = df.columns.str.strip()
        st.session_state.df = df
    
    if st.session_state.df is not None:
        df = st.session_state.df

        # KPI row
        st.markdown(f"""
        <div class='kpi-row'>
          <div class='kpi-box'><div class='kpi-val'>{df.shape[0]:,}</div><div class='kpi-lbl'>Rows</div></div>
          <div class='kpi-box'><div class='kpi-val'>{df.shape[1]}</div><div class='kpi-lbl'>Columns</div></div>
          <div class='kpi-box'><div class='kpi-val'>{df.isnull().sum().sum()}</div><div class='kpi-lbl'>Missing Values</div></div>
          <div class='kpi-box'><div class='kpi-val'>{df.select_dtypes(include=np.number).shape[1]}</div><div class='kpi-lbl'>Numeric Cols</div></div>
          <div class='kpi-box'><div class='kpi-val'>{df.select_dtypes(exclude=np.number).shape[1]}</div><div class='kpi-lbl'>Categorical Cols</div></div>
        </div>
        """, unsafe_allow_html=True)

        tab_data, tab_pca = st.tabs(["📋 Data Preview", "🔵 PCA Visualisation"])

        with tab_data:
            st.dataframe(df.head(20), use_container_width=True, height=320)

        with tab_pca:
            st.subheader("PCA Configuration")
            cols = df.columns.tolist()
            target_sel = st.selectbox("Select Target Feature", cols, index=len(cols)-1, key="pca_target")
            feat_opts = [c for c in cols if c != target_sel]
            pca_features = st.multiselect("Select Features for PCA", feat_opts, default=feat_opts[:min(8,len(feat_opts))])

            if pca_features and len(pca_features) >= 2:
                from sklearn.preprocessing import StandardScaler, LabelEncoder
                from sklearn.decomposition import PCA

                Xp = df[pca_features].copy()
                # encode categoricals
                for c in Xp.select_dtypes(exclude=np.number).columns:
                    Xp[c] = LabelEncoder().fit_transform(Xp[c].astype(str))
                Xp = Xp.fillna(Xp.median())
                Xp_sc = StandardScaler().fit_transform(Xp)

                n_comp = min(3, len(pca_features))
                pca = PCA(n_components=n_comp)
                comps = pca.fit_transform(Xp_sc)
                ev = pca.explained_variance_ratio_ * 100

                target_col_pca = df[target_sel].astype(str)

                c3, c4 = st.columns([2,1])
                with c3:
                    if n_comp >= 3:
                        fig = px.scatter_3d(
                            x=comps[:,0], y=comps[:,1], z=comps[:,2],
                            color=target_col_pca,
                            labels={"x":f"PC1 ({ev[0]:.1f}%)","y":f"PC2 ({ev[1]:.1f}%)","z":f"PC3 ({ev[2]:.1f}%)"},
                            title="PCA — 3D Scatter",
                            color_discrete_sequence=px.colors.qualitative.Vivid,
                        )
                    else:
                        fig = px.scatter(
                            x=comps[:,0], y=comps[:,1],
                            color=target_col_pca,
                            labels={"x":f"PC1 ({ev[0]:.1f}%)","y":f"PC2 ({ev[1]:.1f}%)"},
                            title="PCA — 2D Scatter",
                            color_discrete_sequence=px.colors.qualitative.Vivid,
                        )
                    plotly_fig(fig, 440)

                with c4:
                    fig2 = px.bar(
                        x=[f"PC{i+1}" for i in range(n_comp)],
                        y=ev, color=ev,
                        color_continuous_scale="Blues",
                        title="Explained Variance %",
                        labels={"x":"Component","y":"Variance %"},
                    )
                    plotly_fig(fig2, 220)

                    # loading heatmap
                    load_df = pd.DataFrame(
                        pca.components_[:n_comp].T,
                        index=pca_features,
                        columns=[f"PC{i+1}" for i in range(n_comp)]
                    )
                    fig3 = px.imshow(
                        load_df, text_auto=".2f",
                        color_continuous_scale="RdBu_r",
                        title="PCA Loadings",
                        aspect="auto",
                    )
                    plotly_fig(fig3, 220)

        # Target selection for pipeline
        st.markdown("---")
        st.subheader("Select Pipeline Target Column")
        target = st.selectbox("Target column", df.columns.tolist(), index=len(df.columns)-1, key="pipeline_target")
        st.session_state.target_col = target

        nav_buttons(back=True, next_label="Proceed to EDA →")


# ──────────────────────────────────────────────
# STEP 2 — EDA
# ──────────────────────────────────────────────
elif cur == 2:
    section_header("Step 3 — Exploratory Data Analysis")
    df = st.session_state.df
    target = st.session_state.target_col

    if df is None:
        st.warning("Please upload data first."); st.stop()

    # Summary stats
    with st.expander("📊 Descriptive Statistics", expanded=True):
        st.dataframe(df.describe(include="all").T.style.background_gradient(cmap="Blues", axis=1), use_container_width=True)

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔢 Distributions","🔗 Correlations","📦 Box Plots","📌 Target Analysis","⚠️ Missing Data"])

    with tab1:
        sel = st.selectbox("Pick column", num_cols, key="dist_col")
        fig = px.histogram(df, x=sel, marginal="violin", color_discrete_sequence=["#60a5fa"],
                           title=f"Distribution — {sel}")
        plotly_fig(fig)

    with tab2:
        if num_cols:
            corr = df[num_cols].corr()
            fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                            title="Correlation Heatmap", aspect="auto")
            plotly_fig(fig, 500)

    with tab3:
        sel_box = st.multiselect("Choose columns", num_cols, default=num_cols[:min(5,len(num_cols))], key="box_sel")
        if sel_box:
            fig = px.box(df[sel_box], color_discrete_sequence=px.colors.qualitative.Vivid,
                         title="Box Plots — selected features")
            plotly_fig(fig)

    with tab4:
        if target:
            tc = df[target]
            if tc.dtype == object or tc.nunique() <= 20:
                vc = tc.value_counts().reset_index()
                vc.columns = [target, "count"]
                fig = px.pie(vc, names=target, values="count",
                             color_discrete_sequence=px.colors.qualitative.Vivid,
                             title=f"Target distribution — {target}")
            else:
                fig = px.histogram(df, x=target, color_discrete_sequence=["#a78bfa"],
                                   title=f"Target distribution — {target}")
            plotly_fig(fig)

            # Scatter vs numeric features
            if num_cols:
                feat_vs = st.selectbox("Feature vs Target", [c for c in num_cols if c != target], key="feat_vs")
                # Ensure target column is numeric for scatter plot
                y_data = df[target]
                from pandas.api.types import is_numeric_dtype
                if not is_numeric_dtype(y_data):
                    # Try to convert to numeric, coerce errors to NaN
                    y_data = pd.to_numeric(y_data, errors='coerce')
                if is_numeric_dtype(y_data):
                    fig2 = px.scatter(df.assign(**{target: y_data}), x=feat_vs, y=target, trendline="lowess",
                                      color_discrete_sequence=["#34d399"],
                                      title=f"{feat_vs} vs {target}")
                    plotly_fig(fig2)
                else:
                    st.warning(f"Target column '{target}' is not numeric and cannot be plotted as a scatter plot.")

    with tab5:
        miss = df.isnull().sum().reset_index()
        miss.columns = ["Column","Missing"]
        miss["Pct"] = (miss["Missing"] / len(df) * 100).round(2)
        miss = miss[miss["Missing"] > 0]
        if miss.empty:
            st.success("✅ No missing values detected!")
        else:
            fig = px.bar(miss, x="Column", y="Pct", color="Pct",
                         color_continuous_scale="Reds",
                         title="Missing Value %", labels={"Pct":"% Missing"})
            plotly_fig(fig)
            st.dataframe(miss, use_container_width=True)

    nav_buttons(back=True, next_label="Data Engineering →")


# ──────────────────────────────────────────────
# STEP 3 — DATA ENGINEERING
# ──────────────────────────────────────────────
elif cur == 3:
    section_header("Step 4 — Data Engineering & Cleaning")
    df = st.session_state.df.copy()
    target = st.session_state.target_col

    if df is None: st.warning("No data."); st.stop()

    from sklearn.preprocessing import LabelEncoder

    # ─ Encoding
    with st.expander("🔤 Encode Categorical Features", expanded=True):
        cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
        if target in cat_cols: cat_cols.remove(target)
        if cat_cols:
            enc_method = st.radio("Encoding method", ["Label Encoding", "One-Hot Encoding"], horizontal=True)
            if st.button("Apply Encoding"):
                if enc_method == "Label Encoding":
                    for c in cat_cols:
                        df[c] = LabelEncoder().fit_transform(df[c].astype(str))
                else:
                    df = pd.get_dummies(df, columns=cat_cols)
                # encode target if classification
                if st.session_state.problem_type == "Classification":
                    if df[target].dtype == object:
                        df[target] = LabelEncoder().fit_transform(df[target].astype(str).str.strip())
                st.session_state.df_clean = df
                st.session_state.encoding_done = True
                st.success(f"✅ Encoded {len(cat_cols)} columns with {enc_method}")
        else:
            st.info("No categorical columns found (excluding target).")
            if st.session_state.problem_type == "Classification" and df[target].dtype == object:
                df[target] = LabelEncoder().fit_transform(df[target].astype(str).str.strip())
            st.session_state.df_clean = df

    df_work = st.session_state.df_clean if st.session_state.df_clean is not None else df

    # ─ Imputation
    with st.expander("🩹 Handle Missing Values"):
        num_cols = df_work.select_dtypes(include=np.number).columns.tolist()
        miss_cols = [c for c in num_cols if df_work[c].isnull().any()]
        if miss_cols:
            imp_method = st.selectbox("Imputation method", ["Mean","Median","Mode"], key="imp_meth")
            if st.button("Apply Imputation"):
                for c in miss_cols:
                    if imp_method == "Mean":    df_work[c].fillna(df_work[c].mean(), inplace=True)
                    elif imp_method == "Median": df_work[c].fillna(df_work[c].median(), inplace=True)
                    else:                        df_work[c].fillna(df_work[c].mode()[0], inplace=True)
                st.session_state.df_clean = df_work
                st.success(f"✅ Imputed {len(miss_cols)} columns with {imp_method}")
        else:
            st.success("✅ No missing values in numeric columns.")

    # ─ Outlier Detection
    with st.expander("🔍 Outlier Detection & Removal", expanded=True):
        num_cols_od = [c for c in df_work.select_dtypes(include=np.number).columns if c != target]
        method = st.selectbox("Detection method", ["IQR","Isolation Forest","DBSCAN","OPTICS"], key="od_method")
        st.session_state.outlier_method = method

        if st.button("Detect Outliers"):
            from sklearn.ensemble import IsolationForest
            from sklearn.cluster import DBSCAN, OPTICS
            from sklearn.preprocessing import StandardScaler

            X_od = df_work[num_cols_od].copy().fillna(df_work[num_cols_od].median())

            if method == "IQR":
                mask = pd.Series([False]*len(df_work), index=df_work.index)
                for c in num_cols_od:
                    q1, q3 = df_work[c].quantile(.25), df_work[c].quantile(.75)
                    iqr = q3 - q1
                    mask |= (df_work[c] < q1 - 1.5*iqr) | (df_work[c] > q3 + 1.5*iqr)
            elif method == "Isolation Forest":
                pred = IsolationForest(contamination=0.05, random_state=42).fit_predict(X_od)
                mask = pd.Series(pred == -1, index=df_work.index)
            elif method == "DBSCAN":
                X_sc = StandardScaler().fit_transform(X_od)
                labels = DBSCAN(eps=2.5, min_samples=5).fit_predict(X_sc)
                mask = pd.Series(labels == -1, index=df_work.index)
            else:
                X_sc = StandardScaler().fit_transform(X_od)
                labels = OPTICS(min_samples=5).fit_predict(X_sc)
                mask = pd.Series(labels == -1, index=df_work.index)

            st.session_state.outlier_mask = mask
            n_out = mask.sum()
            pct = n_out/len(df_work)*100

            st.markdown(f"""
            <div class='kpi-row'>
              <div class='kpi-box'><div class='kpi-val' style='color:#f87171;'>{n_out}</div><div class='kpi-lbl'>Outliers Detected</div></div>
              <div class='kpi-box'><div class='kpi-val'>{pct:.1f}%</div><div class='kpi-lbl'>of Dataset</div></div>
              <div class='kpi-box'><div class='kpi-val'>{len(df_work)-n_out}</div><div class='kpi-lbl'>Clean Rows</div></div>
            </div>
            """, unsafe_allow_html=True)

            # Visualise on first 2 numeric cols
            if len(num_cols_od) >= 2:
                fig = px.scatter(
                    df_work, x=num_cols_od[0], y=num_cols_od[1],
                    color=mask.astype(str).map({"True":"Outlier","False":"Normal"}),
                    color_discrete_map={"Outlier":"#f87171","Normal":"#34d399"},
                    title=f"Outliers via {method}",
                    opacity=0.6,
                )
                plotly_fig(fig)

        if st.session_state.outlier_mask is not None and st.session_state.outlier_mask.sum() > 0:
            st.markdown(f"<span class='badge badge-red'>⚠️ {st.session_state.outlier_mask.sum()} outliers found</span>", unsafe_allow_html=True)
            if st.button("🗑️ Remove Outliers from Dataset", type="primary"):
                df_work = df_work[~st.session_state.outlier_mask].reset_index(drop=True)
                st.session_state.df_clean = df_work
                st.session_state.outlier_mask = None
                st.success(f"✅ Outliers removed. New shape: {df_work.shape}")
                st.rerun()

    st.dataframe(df_work.head(10), use_container_width=True)
    nav_buttons(back=True, next_label="Feature Selection →")


# ──────────────────────────────────────────────
# STEP 4 — FEATURE SELECTION
# ──────────────────────────────────────────────
elif cur == 4:
    section_header("Step 5 — Feature Selection")
    df_work = st.session_state.df_clean if st.session_state.df_clean is not None else st.session_state.df
    target = st.session_state.target_col

    if df_work is None: st.warning("No data."); st.stop()

    all_feats = [c for c in df_work.select_dtypes(include=np.number).columns if c != target]

    method = st.selectbox("Feature selection method", [
        "Variance Threshold","Correlation with Target","Information Gain (MI)"
    ], key="fs_method")

    if st.button("Run Feature Selection"):
        from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, mutual_info_regression

        X = df_work[all_feats].fillna(df_work[all_feats].median())
        y = df_work[target]

        if method == "Variance Threshold":
            thresh = st.session_state.get("var_thresh", 0.01)
            sel = VarianceThreshold(threshold=thresh)
            sel.fit(X)
            selected = [f for f, s in zip(all_feats, sel.get_support()) if s]
            scores = pd.DataFrame({"Feature": all_feats, "Variance": X.var().values})
            scores = scores.sort_values("Variance", ascending=False)
            fig = px.bar(scores, x="Feature", y="Variance", color="Variance",
                         color_continuous_scale="Blues", title="Feature Variances")
            plotly_fig(fig)

        elif method == "Correlation with Target":
            from pandas.api.types import is_numeric_dtype
            y_numeric = y
            if not is_numeric_dtype(y):
                y_numeric = pd.to_numeric(y, errors='coerce')
            if is_numeric_dtype(y_numeric):
                corrs = X.corrwith(y_numeric).abs().reset_index()
                corrs.columns = ["Feature","AbsCorr"]
                corrs = corrs.sort_values("AbsCorr", ascending=False)
                selected = corrs[corrs["AbsCorr"] > 0.05]["Feature"].tolist()
                fig = px.bar(corrs, x="Feature", y="AbsCorr", color="AbsCorr",
                             color_continuous_scale="Viridis", title="Abs Correlation with Target")
                plotly_fig(fig)
            else:
                st.warning(f"Target column '{target}' is not numeric and cannot compute correlation with features.")

        else:  # MI
            mi_fn = mutual_info_classif if st.session_state.problem_type == "Classification" else mutual_info_regression
            mi = mi_fn(X, y, random_state=42)
            mi_df = pd.DataFrame({"Feature": all_feats, "MI Score": mi}).sort_values("MI Score", ascending=False)
            selected = mi_df[mi_df["MI Score"] > 0]["Feature"].tolist()
            fig = px.bar(mi_df, x="Feature", y="MI Score", color="MI Score",
                         color_continuous_scale="Plasma", title="Mutual Information with Target")
            plotly_fig(fig)

        st.session_state.selected_final_features = selected
        st.success(f"✅ {len(selected)} features selected out of {len(all_feats)}")

    if st.session_state.selected_final_features:
        st.subheader("✏️ Adjust Selected Features")
        var_thresh = st.slider("Variance Threshold (for VT method)", 0.0, 1.0, 0.01, 0.01, key="var_thresh")
        final = st.multiselect(
            "Final features to use",
            all_feats,
            default=st.session_state.selected_final_features,
            key="manual_feat_sel"
        )
        if st.button("Confirm Feature Set"):
            st.session_state.selected_final_features = final
            st.success(f"✅ Using {len(final)} features.")

    nav_buttons(back=True, next_label="Data Split →",
                next_disabled=not st.session_state.selected_final_features)


# ──────────────────────────────────────────────
# STEP 5 — DATA SPLIT
# ──────────────────────────────────────────────
elif cur == 5:
    section_header("Step 6 — Train / Test Split")
    df_work = st.session_state.df_clean if st.session_state.df_clean is not None else st.session_state.df
    target = st.session_state.target_col
    feats = st.session_state.selected_final_features

    if df_work is None or not feats: st.warning("Complete previous steps first."); st.stop()

    col1, col2 = st.columns(2)
    with col1:
        test_sz = st.slider("Test size", 0.1, 0.5, 0.2, 0.05, key="test_sz")
    with col2:
        scaler = st.selectbox("Feature scaling", ["StandardScaler","MinMaxScaler","None"], key="scaler_sel")

    stratify_opt = False
    if st.session_state.problem_type == "Classification":
        stratify_opt = st.checkbox("Stratified split (recommended for classification)", value=True)

    if st.button("Apply Split", type="primary"):
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler, MinMaxScaler

        X = df_work[feats].fillna(df_work[feats].median())
        y = df_work[target]
        strat = y if stratify_opt else None

        try:
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_sz, random_state=42, stratify=strat)
            split_msg = "Stratified split applied" if strat is not None else "Random split applied"
        except ValueError as e:
            if "least populated class" in str(e):
                # Fall back to non-stratified split if stratification fails
                st.warning("⚠️ Stratification failed due to classes with too few members. Using random split instead.")
                X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_sz, random_state=42, stratify=None)
                split_msg = "Random split applied (stratification not possible)"
            else:
                raise e

        if scaler != "None":
            sc = StandardScaler() if scaler == "StandardScaler" else MinMaxScaler()
            X_tr = pd.DataFrame(sc.fit_transform(X_tr), columns=feats)
            X_te = pd.DataFrame(sc.transform(X_te), columns=feats)

        st.session_state.X_train = X_tr
        st.session_state.X_test  = X_te
        st.session_state.y_train = y_tr
        st.session_state.y_test  = y_te
        st.session_state.test_size = test_sz

        st.markdown(f"""
        <div class='kpi-row'>
          <div class='kpi-box'><div class='kpi-val'>{len(X_tr)}</div><div class='kpi-lbl'>Train samples</div></div>
          <div class='kpi-box'><div class='kpi-val'>{len(X_te)}</div><div class='kpi-lbl'>Test samples</div></div>
          <div class='kpi-box'><div class='kpi-val'>{len(feats)}</div><div class='kpi-lbl'>Features</div></div>
          <div class='kpi-box'><div class='kpi-val'>{(1-test_sz)*100:.0f}% / {test_sz*100:.0f}%</div><div class='kpi-lbl'>Train / Test ratio</div></div>
        </div>
        """, unsafe_allow_html=True)

        st.info(f"✅ {split_msg}")

        # Pie chart of split
        fig = px.pie(values=[len(X_tr), len(X_te)], names=["Train","Test"],
                     color_discrete_sequence=["#3b82f6","#f59e0b"],
                     title="Dataset Split")
        plotly_fig(fig, 260)

    nav_buttons(back=True, next_label="Model Selection →",
                next_disabled=st.session_state.X_train is None)


# ──────────────────────────────────────────────
# STEP 6 — MODEL SELECTION
# ──────────────────────────────────────────────
elif cur == 6:
    section_header("Step 7 — Model Selection")
    pt = st.session_state.problem_type

    models_cls = {
        "Logistic Regression": "A fast linear classifier — great baseline for binary/multiclass problems.",
        "SVM (Classifier)": "Support Vector Machine with kernel trick. Effective in high-dimensional spaces.",
        "Random Forest (Classifier)": "Ensemble of decision trees, robust to noise and overfitting.",
        "K-Means (Unsupervised)": "Clusters data into K groups. Use when labels aren't available.",
    }
    models_reg = {
        "Linear Regression": "Fits a linear relationship between features and target.",
        "SVM (Regressor)": "SVR — support vector regression with kernel options.",
        "Random Forest (Regressor)": "Ensemble regressor. Strong out-of-box performance.",
    }
    model_map = models_cls if pt == "Classification" else models_reg

    cols = st.columns(2)
    for i, (name, desc) in enumerate(model_map.items()):
        with cols[i % 2]:
            selected = st.session_state.selected_model == name
            border = "border-color:#60a5fa;" if selected else ""
            st.markdown(f"""
            <div class='panel' style='{border}'>
              <div class='panel-title'>{name}</div>
              <div style='font-size:12px; color:#9ca3af;'>{desc}</div>
            </div>""", unsafe_allow_html=True)
            if st.button(f"✔ Select {name}", key=f"sel_{name}"):
                st.session_state.selected_model = name
                st.rerun()

    # SVM kernel options
    if st.session_state.selected_model and "SVM" in st.session_state.selected_model:
        st.session_state.svm_kernel = st.selectbox("SVM Kernel", ["rbf","linear","poly","sigmoid"], key="svm_k")

    if st.session_state.selected_model:
        st.success(f"✅ Selected: **{st.session_state.selected_model}**")

    nav_buttons(back=True, next_label="Training & CV →",
                next_disabled=not st.session_state.selected_model)


# ──────────────────────────────────────────────
# STEP 7 — TRAINING & K-FOLD
# ──────────────────────────────────────────────
elif cur == 7:
    section_header("Step 8 — Model Training & K-Fold Cross Validation")

    X_tr = st.session_state.X_train
    y_tr = st.session_state.y_train
    model_name = st.session_state.selected_model

    if X_tr is None: st.warning("Complete previous steps first."); st.stop()

    k = st.slider("Number of folds (K)", 2, 15, 5, key="k_folds_slider")
    st.session_state.k_folds = k

    if st.button("🏋️ Train Model with K-Fold CV", type="primary"):

        from sklearn.preprocessing import LabelEncoder
        kernel = st.session_state.get("svm_kernel","rbf")

        model_registry = {
            "Logistic Regression":          LogisticRegression(max_iter=1000, random_state=42),
            "SVM (Classifier)":             SVC(kernel=kernel, probability=True, random_state=42),
            "Random Forest (Classifier)":   RandomForestClassifier(n_estimators=100, random_state=42),
            "Linear Regression":            LinearRegression(),
            "SVM (Regressor)":              SVR(kernel=kernel),
            "Random Forest (Regressor)":    RandomForestRegressor(n_estimators=100, random_state=42),
        }

        if model_name == "K-Means (Unsupervised)":
            model = KMeans(n_clusters=int(y_tr.nunique()), random_state=42)
            with st.spinner("Training K-Means..."):
                model.fit(X_tr)
            st.session_state.trained_model = model
            st.success("✅ K-Means trained (no cross-validation for unsupervised).")
        else:
            model = model_registry[model_name]
            pt = st.session_state.problem_type

            # Encode target for classification if not numeric

            y_enc = y_tr
            if pt == "Classification":
                try:
                    # Always ensure target is properly encoded for classification
                    if hasattr(y_tr, 'dtype') and y_tr.dtype == object:
                        # Strip whitespace and encode string targets
                        y_clean = y_tr.astype(str).str.strip()
                        le = LabelEncoder()
                        y_enc = le.fit_transform(y_clean)
                        st.session_state.target_encoder = le  # Save encoder for later use
                    else:
                        # Target is already numeric
                        y_enc = y_tr
                except Exception as e:
                    st.warning(f"Could not encode target labels: {e}")
                    st.stop()
                scoring = "accuracy"
                cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
            else:
                scoring = "r2"
                cv = KFold(n_splits=k, shuffle=True, random_state=42)

            with st.spinner(f"Training with {k}-fold CV..."):
                t0 = time.time()
                cv_scores = cross_val_score(model, X_tr, y_enc, cv=cv, scoring=scoring, n_jobs=-1)
                elapsed = time.time() - t0
                model.fit(X_tr, y_enc)

            st.session_state.trained_model = model
            st.session_state.cv_results = cv_scores

            st.markdown(f"""
            <div class='kpi-row'>
              <div class='kpi-box'><div class='kpi-val'>{cv_scores.mean():.4f}</div><div class='kpi-lbl'>Mean CV {scoring}</div></div>
              <div class='kpi-box'><div class='kpi-val'>{cv_scores.std():.4f}</div><div class='kpi-lbl'>Std Dev</div></div>
              <div class='kpi-box'><div class='kpi-val'>{cv_scores.max():.4f}</div><div class='kpi-lbl'>Best Fold</div></div>
              <div class='kpi-box'><div class='kpi-val'>{elapsed:.1f}s</div><div class='kpi-lbl'>Train time</div></div>
            </div>
            """, unsafe_allow_html=True)

            # Fold scores bar chart
            fold_df = pd.DataFrame({
                "Fold": [f"Fold {i+1}" for i in range(k)],
                "Score": cv_scores,
                "Rel": (cv_scores - cv_scores.mean()) / (cv_scores.std() + 1e-9)
            })
            fig = px.bar(fold_df, x="Fold", y="Score", color="Score",
                         color_continuous_scale="Blues",
                         title=f"{k}-Fold CV Scores ({scoring})")
            fig.add_hline(y=cv_scores.mean(), line_dash="dash", line_color="#f59e0b",
                          annotation_text=f"Mean={cv_scores.mean():.4f}")
            plotly_fig(fig)

    nav_buttons(back=True, next_label="Performance Metrics →",
                next_disabled=st.session_state.trained_model is None)


# ──────────────────────────────────────────────
# STEP 8 — PERFORMANCE METRICS
# ──────────────────────────────────────────────
elif cur == 8:
    section_header("Step 9 — Performance Metrics & Overfitting Analysis")

    model = st.session_state.trained_model
    X_tr, X_te = st.session_state.X_train, st.session_state.X_test
    y_tr, y_te = st.session_state.y_train, st.session_state.y_test
    pt = st.session_state.problem_type

    if model is None: st.warning("Train a model first."); st.stop()

    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        confusion_matrix, roc_auc_score, roc_curve,
        mean_squared_error, mean_absolute_error, r2_score,
    )

    y_tr_pred = model.predict(X_tr)
    y_te_pred = model.predict(X_te)

    # For classification, ensure true labels match prediction type
    if pt == "Classification":
        # Check if predictions are numeric but true labels are strings
        if hasattr(y_tr_pred, 'dtype') and np.issubdtype(y_tr_pred.dtype, np.number):
            if hasattr(y_tr, 'dtype') and y_tr.dtype == object:
                # Need to encode true labels to match predictions
                if hasattr(st.session_state, 'target_encoder') and st.session_state.target_encoder is not None:
                    y_tr_encoded = st.session_state.target_encoder.transform(y_tr.astype(str).str.strip())
                    y_te_encoded = st.session_state.target_encoder.transform(y_te.astype(str).str.strip())
                else:
                    # Create encoder on the fly
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    y_tr_encoded = le.fit_transform(y_tr.astype(str).str.strip())
                    y_te_encoded = le.transform(y_te.astype(str).str.strip())
            else:
                # True labels are already numeric
                y_tr_encoded = y_tr
                y_te_encoded = y_te
        else:
            # Predictions are not numeric, use true labels as is
            y_tr_encoded = y_tr
            y_te_encoded = y_te
    else:
        y_tr_encoded = y_tr
        y_te_encoded = y_te

    if pt == "Classification":
        tr_acc = accuracy_score(y_tr_encoded, y_tr_pred)
        te_acc = accuracy_score(y_te_encoded, y_te_pred)
        te_f1  = f1_score(y_te_encoded, y_te_pred, average="weighted", zero_division=0)
        te_pr  = precision_score(y_te_encoded, y_te_pred, average="weighted", zero_division=0)
        te_rc  = recall_score(y_te_encoded, y_te_pred, average="weighted", zero_division=0)
        gap    = tr_acc - te_acc

        overfit_label = "✅ Good Fit"
        badge = "green"
        if gap > 0.15: overfit_label, badge = "⚠️ Overfitting", "amber"
        if tr_acc < 0.65: overfit_label, badge = "⚠️ Underfitting", "red"

        st.markdown(f"""
        <div class='kpi-row'>
          <div class='kpi-box'><div class='kpi-val'>{tr_acc:.4f}</div><div class='kpi-lbl'>Train Accuracy</div></div>
          <div class='kpi-box'><div class='kpi-val'>{te_acc:.4f}</div><div class='kpi-lbl'>Test Accuracy</div></div>
          <div class='kpi-box'><div class='kpi-val'>{te_f1:.4f}</div><div class='kpi-lbl'>F1 Score</div></div>
          <div class='kpi-box'><div class='kpi-val'>{te_pr:.4f}</div><div class='kpi-lbl'>Precision</div></div>
          <div class='kpi-box'><div class='kpi-val'>{te_rc:.4f}</div><div class='kpi-lbl'>Recall</div></div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"<span class='badge badge-{badge}'>{overfit_label} — Train/Test gap: {gap:.4f}</span>", unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["Confusion Matrix","ROC Curve","Fit Analysis"])

        with tab1:
            cm = confusion_matrix(y_te_encoded, y_te_pred)
            fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                            title="Confusion Matrix", labels={"x":"Predicted","y":"Actual"})
            plotly_fig(fig)

        with tab2:
            try:
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X_te)
                    classes = model.classes_
                    if len(classes) == 2:
                        fpr, tpr, _ = roc_curve(y_te_encoded, probs[:,1])
                        auc_score = roc_auc_score(y_te_encoded, probs[:,1])
                        fig = px.area(x=fpr, y=tpr, title=f"ROC Curve (AUC={auc_score:.4f})",
                                      labels={"x":"FPR","y":"TPR"},
                                      color_discrete_sequence=["#60a5fa"])
                        fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                                      line=dict(dash="dash", color="#6b7280"))
                        plotly_fig(fig)
                    else:
                        st.info("Multi-class ROC not shown (use OvR for multi-class).")
            except Exception as e:
                st.warning(f"ROC not available: {e}")

        with tab3:
            fig = go.Figure()
            fig.add_bar(name="Train Accuracy", x=["Accuracy"], y=[tr_acc],
                        marker_color="#3b82f6")
            fig.add_bar(name="Test Accuracy", x=["Accuracy"], y=[te_acc],
                        marker_color="#f59e0b")
            fig.update_layout(title="Train vs Test Accuracy", barmode="group", **PLOTLY_DARK)
            st.plotly_chart(fig, use_container_width=True)

            if gap > 0.15:
                st.markdown("""<div class='panel' style='border-color:#fbbf24;'>
                <div class='panel-title' style='color:#fbbf24;'>⚠️ Overfitting Detected</div>
                <div style='font-size:13px; color:#9ca3af;'>Model performs significantly better on training than test data. 
                Try: more training data, regularisation, pruning, dropout, reducing model complexity.</div>
                </div>""", unsafe_allow_html=True)
            elif tr_acc < 0.65:
                st.markdown("""<div class='panel' style='border-color:#f87171;'>
                <div class='panel-title' style='color:#f87171;'>⚠️ Underfitting Detected</div>
                <div style='font-size:13px; color:#9ca3af;'>Model is too simple. 
                Try: more features, higher model complexity, more training epochs, better features.</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""<div class='panel' style='border-color:#4ade80;'>
                <div class='panel-title' style='color:#4ade80;'>✅ Model Generalises Well</div>
                <div style='font-size:13px; color:#9ca3af;'>Train and test accuracy are close. Good fit!</div>
                </div>""", unsafe_allow_html=True)

    else:  # Regression
        tr_r2   = r2_score(y_tr, y_tr_pred)
        te_r2   = r2_score(y_te, y_te_pred)
        te_rmse = np.sqrt(mean_squared_error(y_te, y_te_pred))
        te_mae  = mean_absolute_error(y_te, y_te_pred)
        gap     = tr_r2 - te_r2

        badge = "green"
        label = "✅ Good Fit"
        if gap > 0.15: label, badge = "⚠️ Overfitting", "amber"
        if te_r2 < 0.4: label, badge = "⚠️ Underfitting", "red"

        st.markdown(f"""
        <div class='kpi-row'>
          <div class='kpi-box'><div class='kpi-val'>{tr_r2:.4f}</div><div class='kpi-lbl'>Train R²</div></div>
          <div class='kpi-box'><div class='kpi-val'>{te_r2:.4f}</div><div class='kpi-lbl'>Test R²</div></div>
          <div class='kpi-box'><div class='kpi-val'>{te_rmse:.2f}</div><div class='kpi-lbl'>RMSE</div></div>
          <div class='kpi-box'><div class='kpi-val'>{te_mae:.2f}</div><div class='kpi-lbl'>MAE</div></div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"<span class='badge badge-{badge}'>{label} — R² gap: {gap:.4f}</span>", unsafe_allow_html=True)

        tab1, tab2 = st.tabs(["Actual vs Predicted","Residuals"])
        with tab1:
            fig = px.scatter(x=y_te, y=y_te_pred, labels={"x":"Actual","y":"Predicted"},
                             title="Actual vs Predicted",
                             color_discrete_sequence=["#34d399"], opacity=0.6)
            mn, mx = float(y_te.min()), float(y_te.max())
            fig.add_scatter(x=[mn,mx], y=[mn,mx], mode="lines",
                            line=dict(color="#f87171", dash="dash"), name="Perfect fit")
            plotly_fig(fig)
        with tab2:
            resid = y_te.values - y_te_pred
            fig = px.histogram(x=resid, nbins=40, title="Residual Distribution",
                               color_discrete_sequence=["#a78bfa"])
            fig.add_vline(x=0, line_dash="dash", line_color="#f59e0b")
            plotly_fig(fig)

    nav_buttons(back=True, next_label="Hyperparameter Tuning →")


# ──────────────────────────────────────────────
# STEP 9 — HYPERPARAMETER TUNING
# ──────────────────────────────────────────────
elif cur == 9:
    section_header("Step 10 — Hyperparameter Tuning")
    model_name = st.session_state.selected_model
    X_tr = st.session_state.X_train
    y_tr = st.session_state.y_train
    X_te = st.session_state.X_test
    y_te = st.session_state.y_test
    pt   = st.session_state.problem_type

    if X_tr is None: st.warning("Complete previous steps."); st.stop()

    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.svm import SVC, SVR
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import accuracy_score, r2_score

    param_grids = {
        "Logistic Regression": {
            "C": [0.01, 0.1, 1, 10, 100],
            "solver": ["lbfgs","saga"],
            "max_iter": [200, 500],
        },
        "SVM (Classifier)": {
            "C": [0.1, 1, 10],
            "kernel": ["rbf","linear","poly"],
            "gamma": ["scale","auto"],
        },
        "Random Forest (Classifier)": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
        },
        "Linear Regression": {
            "fit_intercept": [True, False],
        },
        "SVM (Regressor)": {
            "C": [0.1, 1, 10],
            "kernel": ["rbf","linear"],
            "epsilon": [0.1, 0.2, 0.5],
        },
        "Random Forest (Regressor)": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
        },
    }

    base_model_map = {
        "Logistic Regression":         LogisticRegression(random_state=42),
        "SVM (Classifier)":            SVC(probability=True, random_state=42),
        "Random Forest (Classifier)":  RandomForestClassifier(random_state=42),
        "Linear Regression":           LinearRegression(),
        "SVM (Regressor)":             SVR(),
        "Random Forest (Regressor)":   RandomForestRegressor(random_state=42),
    }

    if model_name not in param_grids:
        st.info(f"Hyperparameter tuning not available for {model_name} (unsupervised model).")
        nav_buttons(back=True, next_label="✅ Finish"); st.stop()

    pg = param_grids[model_name]
    st.markdown(f"""
    <div class='panel'>
      <div class='panel-title'>Parameter Grid for {model_name}</div>
      {"".join(f"<div style='font-size:12px; color:#9ca3af; margin:3px 0;'><b style='color:#e5e7eb;'>{k}</b>: {v}</div>" for k,v in pg.items())}
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1: search_type = st.selectbox("Search strategy", ["GridSearchCV","RandomizedSearchCV"])
    with c2: n_iter = st.slider("n_iter (RandomizedSearch)", 5, 50, 10) if search_type == "RandomizedSearchCV" else 0
    with c3: cv_k = st.slider("CV folds for tuning", 2, 10, 3)

    scoring = "accuracy" if pt == "Classification" else "r2"

    if st.button("🚀 Run Hyperparameter Search", type="primary"):
        base = base_model_map[model_name]

        # Use encoded labels for classification if encoder exists
        y_tr_tune = y_tr
        y_te_tune = y_te
        if pt == "Classification":
            # Check if we need to encode labels for tuning
            if hasattr(st.session_state, 'target_encoder') and st.session_state.target_encoder is not None:
                y_tr_tune = st.session_state.target_encoder.transform(y_tr.astype(str).str.strip())
                y_te_tune = st.session_state.target_encoder.transform(y_te.astype(str).str.strip())
            elif hasattr(y_tr, 'dtype') and y_tr.dtype == object:
                # Create encoder on the fly if needed
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y_tr_tune = le.fit_transform(y_tr.astype(str).str.strip())
                y_te_tune = le.transform(y_te.astype(str).str.strip())

        with st.spinner(f"Running {search_type}... this may take a minute"):
            if search_type == "GridSearchCV":
                searcher = GridSearchCV(base, pg, cv=cv_k, scoring=scoring, n_jobs=-1, verbose=0)
            else:
                searcher = RandomizedSearchCV(base, pg, n_iter=n_iter, cv=cv_k,
                                              scoring=scoring, n_jobs=-1, random_state=42, verbose=0)
            searcher.fit(X_tr, y_tr_tune)

        best = searcher.best_estimator_
        st.session_state.tuned_model = best

        # Compare default vs tuned
        default = base_model_map[model_name]
        default.fit(X_tr, y_tr_tune)
        def_score = (accuracy_score if pt=="Classification" else r2_score)(y_te_tune, default.predict(X_te))
        tuned_score = (accuracy_score if pt=="Classification" else r2_score)(y_te_tune, best.predict(X_te))
        delta = (tuned_score - def_score) * 100

        st.markdown(f"""
        <div class='kpi-row'>
          <div class='kpi-box'><div class='kpi-val'>{searcher.best_score_:.4f}</div><div class='kpi-lbl'>Best CV Score</div></div>
          <div class='kpi-box'><div class='kpi-val'>{def_score:.4f}</div><div class='kpi-lbl'>Default Test {scoring}</div></div>
          <div class='kpi-box'><div class='kpi-val'>{tuned_score:.4f}</div><div class='kpi-lbl'>Tuned Test {scoring}</div></div>
          <div class='kpi-box'><div class='kpi-val' style='color:{"#4ade80" if delta>=0 else "#f87171"};'>{delta:+.2f}%</div><div class='kpi-lbl'>Improvement</div></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class='panel' style='border-color:#60a5fa;'>
          <div class='panel-title'>🏆 Best Parameters</div>
          {"".join(f"<div style='font-size:13px; color:#9ca3af; margin:4px 0;'><b style='color:#e5e7eb;'>{k}</b>: {v}</div>" for k,v in searcher.best_params_.items())}
        </div>
        """, unsafe_allow_html=True)

        # CV results heatmap (for small grids)
        cv_res = pd.DataFrame(searcher.cv_results_)
        param_cols = [c for c in cv_res.columns if c.startswith("param_")]
        if len(param_cols) >= 2 and len(cv_res) <= 100:
            pivot_data = cv_res[param_cols + ["mean_test_score"]].copy()
            pivot_data.columns = [c.replace("param_","") for c in param_cols] + ["Score"]
            fig = px.scatter(pivot_data,
                             x=pivot_data.columns[0],
                             y="Score",
                             color="Score",
                             color_continuous_scale="Viridis",
                             title="CV Results — Parameter Space",
                             size="Score")
            plotly_fig(fig)

        # Comparison bar
        fig2 = go.Figure()
        fig2.add_bar(name="Default Model", x=[scoring], y=[def_score], marker_color="#6b7280")
        fig2.add_bar(name="Tuned Model",   x=[scoring], y=[tuned_score], marker_color="#60a5fa")
        fig2.update_layout(title="Default vs Tuned Model Performance", barmode="group", **PLOTLY_DARK)
        st.plotly_chart(fig2, use_container_width=True)

    nav_buttons(back=True, next_label="🎉 Pipeline Complete!")

    if st.session_state.current_step == len(STEPS):
        st.balloons()
        st.success("🎉 ML Pipeline Complete! Your model has been trained, evaluated, and tuned.")


# ──────────────────────────────────────────────
# SIDEBAR — PIPELINE SUMMARY
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📋 Pipeline Summary")
    st.markdown(f"**Problem:** {st.session_state.problem_type or '—'}")
    df_s = st.session_state.df_clean if st.session_state.df_clean is not None else st.session_state.df
    if df_s is not None:
        st.markdown(f"**Data shape:** {df_s.shape[0]} × {df_s.shape[1]}")
    st.markdown(f"**Target:** `{st.session_state.target_col or '—'}`")
    feats = st.session_state.selected_final_features
    st.markdown(f"**Features:** {len(feats) if feats else '—'}")
    st.markdown(f"**Model:** {st.session_state.selected_model or '—'}")
    st.markdown(f"**CV Folds:** {st.session_state.k_folds}")
    if st.session_state.cv_results is not None:
        cv = st.session_state.cv_results
        st.markdown(f"**CV Score:** {cv.mean():.4f} ± {cv.std():.4f}")

    st.markdown("---")
    st.markdown("**Jump to Step:**")
    for i, (label, icon) in enumerate(zip(STEPS, STEP_ICONS)):
        if i <= cur:
            if st.button(f"{icon} {label}", key=f"jump_{i}"):
                go_to(i); st.rerun()