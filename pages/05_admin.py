import streamlit as st
import base64
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import streamlit.components.v1 as components

# --- Page Config ---
BASE_DIR = Path(__file__).resolve().parent.parent
favicon_path = BASE_DIR / "assets" / "favicon_zoom.ico"
logo_path = BASE_DIR / "assets" / "logo_white_cropped.png"

with open(logo_path, "rb") as img_file:
    logo_base64 = base64.b64encode(img_file.read()).decode()

st.set_page_config(page_title="Admin Panel", page_icon=str(favicon_path), layout="wide")

# --- Constants ---
MODELS_DIR = BASE_DIR / "models"
METADATA_FILE = MODELS_DIR / "model_registry.json"
ADMIN_PASSWORD = "2026"
DEFAULT_MODEL = "Final_Ridge_Model_23_24_25.pkl"  # cannot be deleted

REQUIRED_COLUMNS = [
    'Year', 'Month', 'YPH',
    'T200', 'U709', 'Epson Salt', 'Dolomite', 'Urea', 'Compost', 'BioShield',
    'temperature_2m_mean', 'rain_sum', 'wind_speed_10m_mean', 'et0_fao_evapotranspiration'
]

DROP_COLUMNS = [
    'temperature_2m_min', 'temperature_2m_max',
    'relative_humidity_2m_min', 'relative_humidity_2m_max',
    'wind_speed_10m_max', 'precipitation_sum',
    'UT925', 'BioShakthi', 'Zinc', 'sunshine_duration',
    'relative_humidity_2m_mean'
]

MONTH_MAP = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4,
    'May': 5, 'June': 6, 'July': 7, 'August': 8,
    'September': 9, 'October': 10, 'November': 11, 'December': 12,
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4,
    'Jun': 6, 'Jul': 7, 'Aug': 8,
    'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
}

WINDOW_SIZE = 3
ALPHA = 5.0

# --- CSS ---
st.markdown("""
<style>
.stApp { background-color: white !important; }
header[data-testid="stHeader"] { display: none !important; }
.block-container { padding: 0rem !important; max-width: 100% !important; }
section[data-testid="stSidebar"] { display: none; }
div[data-testid="stDecoration"] { display: none !important; }

h1, h2, h3 { color: #1B4332 !important; }
p, li { color: black !important; }
label { color: black !important; font-weight: 500; }

.header {
    position: sticky;
    top: 0;
    z-index: 999;
    background-color: #1B4332;
    padding: 10px 10px;
    color: white;
    display: flex;
    justify-content: space-between;
    align-items: center;
    height: 70px;
}

.logo {
    display: block;
    height: 100%;
    width: auto;
    object-fit: contain;
    margin: 0;
    padding: 0;
}

.header a {
    color: white;
    text-decoration: none;
    margin-left: 20px;
    font-weight: bold;
}

.header a:hover { color: #D8F3DC; }

.block-container {
    padding: 0rem !important;
    max-width: 100% !important;
}

.admin-badge {
    display: inline-block;
    background-color: #D95F02;
    color: white !important;
    font-size: 0.72rem;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 12px;
    margin-left: 5px;
    letter-spacing: 1.5px;
}

.section-heading {
    color: #1B4332 !important; font-weight: 700;
    font-size: 1.1rem; margin: 20px 0 10px 0;
    padding-left: 12px; border-left: 4px solid #D95F02;
}

.info-card {
    background: #f7f7f7; border: 1px solid #e0e0e0;
    border-radius: 8px; padding: 18px 22px; margin-bottom: 16px;
}
.info-card b { color: #1B4332 !important; }
.info-card td, .info-card span { color: #222 !important; }

.pill-active {
    display: inline-block; 
    background: #1B4332;
    color: white !important; 
    padding: 3px 14px;
    border-radius: 12px; 
    font-size: 0.82rem; 
    font-weight: 700; 
    margin-top: 8px;
}

span.pill-active {
    color: #ffffff !important;
}

.box-warning {
    background: #FFF8E1; 
    border: 1px solid #FFC107;
    border-left: 5px solid #FFC107; 
    border-radius: 6px;
    padding: 12px 16px; 
    color: #6d4c00 !important;
    margin: 10px 0; font-size: 0.95rem;
}
.box-success {
    background: #E8F5E9; border: 1px solid #4CAF50;
    border-left: 5px solid #1B4332; border-radius: 6px;
    padding: 12px 16px; color: #1B4332 !important;
    margin: 10px 0; font-size: 0.95rem;
}
.box-error {
    background: #FFEBEE; border: 1px solid #EF9A9A;
    border-left: 5px solid #C62828; border-radius: 6px;
    padding: 12px 16px; color: #7f0000 !important;
    margin: 10px 0; font-size: 0.95rem;
}

.metric-table {
    width: 100%; border-collapse: collapse;
    font-size: 0.93rem; margin-bottom: 16px;
}
.metric-table th {
    background-color: #1B4332; color: white !important;
    padding: 9px 14px; text-align: center; font-weight: 600;
}
.metric-table td {
    padding: 8px 14px; text-align: center;
    border-bottom: 1px solid #e8e8e8; color: #111 !important;
}
.metric-table tr:nth-child(even) td { background-color: #f5f5f5; }
.metric-table tr.avg-row td {
    font-weight: 700; background: #E8F5E9 !important; color: #1B4332 !important;
}

/* ── Selectbox — light grey ── */
.stSelectbox div[data-baseweb="select"] {
    background-color: #f5f5f5 !important;
    color: #222 !important;
    border: 1.5px solid #bdbdbd !important;
    border-radius: 6px !important;
}
.stSelectbox div[data-baseweb="select"] > div {
    background-color: #f5f5f5 !important;
    color: #222 !important;
}
.stSelectbox svg { fill: #1B4332 !important; }

/* Targets the individual items in the list */
div[data-baseweb="popover"] li {
    background-color: #f5f5f5 !important;
    color: #222 !important;
}

/* Changes the color when you hover over an option */
div[data-baseweb="popover"] li:hover {
    background-color: #e0e0e0 !important;
    color: #1B4332 !important; /* Optional: text turns green on hover */
}

/* Fix for the "no results" or selected state background */
div[data-baseweb="popover"] div {
    background-color: #f5f5f5 !important;
    color: #222 !important;
}

/* ── Buttons ── */
div.stButton > button {
    background-color: #1B4332 !important;
    color: white !important;
    border-radius: 6px; border: none; height: 2.4em;
    width: 100%; font-weight: 600;
}
div.stButton > button:hover { background-color: #2D6A4F !important; color: white !important; }
div.stButton > button p,
div.stButton > button span,
div.stButton > button div { color: white !important; }

/* Delete button — red style */
div.stButton > button[kind="delete-btn"],
.delete-btn > div > button {
    background-color: #C62828 !important;
    color: white !important;
}
.delete-btn > div > button:hover {
    background-color: #b71c1c !important;
}

input[type="password"], input[type="text"] {
    background-color: white !important; color: black !important;
    border: 1px solid #ccc !important; border-radius: 6px !important;
}

button[data-baseweb="tab"] { color: #1B4332 !important; font-weight: 600; }
button[data-baseweb="tab"][aria-selected="true"] {
    border-bottom: 2px solid #D95F02 !important; color: #D95F02 !important;
}

/* ── File uploader — light grey ── */
[data-testid="stFileUploader"] > section {
    background-color: #f5f5f5 !important;
    border: 1.5px dashed #bdbdbd !important;
    border-radius: 8px !important;
    transition: background-color 0.2s ease;
}

/* Hover state for the upload box */
[data-testid="stFileUploader"] > section:hover {
    background-color: #e0e0e0 !important;
    border-color: #1B4332 !important;
}

/* ── Text and Icon colors (Drag & Drop / Limit) ── */
/* Targets the specific container for dropzone instructions */
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] small,
[data-testid="stFileDropzoneInstructions"] span,
[data-testid="stFileDropzoneInstructions"] small {
    color: #1B4332 !important; 
    font-weight: 600 !important;
    opacity: 1 !important; /* Forces the color to be solid green */
}

/* Fallback for general text inside the uploader area */
[data-testid="stFileUploader"] section div,
[data-testid="stFileUploader"] section small {
    color: #1B4332 !important;
}

/* Ensures the upload icon also matches the header green */
[data-testid="stFileUploader"] svg {
    fill: #1B4332 !important;
}

/* The "Browse files" button */
[data-testid="stFileUploader"] button {
    background-color: white !important;
    color: #1B4332 !important;
    border: 1px solid #ccc !important;
    border-radius: 6px !important;
}

[data-testid="stFileUploader"] button:hover {
    background-color: #f0f0f0 !important;
    border-color: #1B4332 !important;
}

/* Uploaded file name area */
[data-testid="stFileUploaderFile"] {
    background-color: white !important;
    border: 1px solid #e0e0e0 !important;
    border-radius: 6px !important;
}

[data-testid="stFileUploaderFile"] * {
    color: #1B4332 !important;
}

.footer {
    background-color: #1B4332;
    color: white;
    padding: 20px;
}

.footer p, .footer a { 
    color: white; !important
}

.footer-link {
    text-decoration: none !important;
    cursor: pointer;
    color: white !important;
}

.footer-link:hover {
    color: Grey;
    text-decoration: none !important;
}

.flex-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-left: auto;
}

.flex-around {
    display: flex;
    justify-content: space-around;
    align-items: center;
}

.center-text { 
    text-align: center; 
}

/* This targets every element inside the footer to override the "black" rule */
.footer p, 
.footer a, 
.footer b, 
.footer div { 
    color: white !important; 
}

.footer-link:hover {
    color: #D8F3DC !important; /* Light green hover instead of Grey */
    text-decoration: none !important;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown(f"""
<div class="header">
    <img src="data:image/png;base64,{logo_base64}" class="logo">
    <div style="display:flex; gap:10px;">
        <span class="admin-badge">ADMIN</span>
        <a href="/home" target="_self">Home</a>
        <a href="/tea_yield" target="_self">Yield Prediction Page</a>
        <a href="/disease_detection" target="_self">Disease Detection Page</a>
        <a href="/chatbot" target="_self">Chatbot Page</a>
        <a href="#"></a>
    </div>
</div>
""", unsafe_allow_html=True)


# Helpers

def load_registry():
    if METADATA_FILE.exists():
        with open(METADATA_FILE, "r") as f:
            return json.load(f)
    return {"active_model": None, "models": []}


def save_registry(registry):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METADATA_FILE, "w") as f:
        json.dump(registry, f, indent=2)


def validate_csv(df):
    errors = []
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {', '.join(missing)}")
    if 'YPH' in df.columns and df['YPH'].isna().all():
        errors.append("YPH column is entirely empty.")
    if 'Year' in df.columns and df['Year'].nunique() < 4:
        errors.append(f"Only {df['Year'].nunique()} unique year(s) found. Minimum 4 required.")
    return errors


def preprocess(df):
    df = df.copy()
    df = df[df['YPH'].notna()]
    if df['Month'].dtype == object:
        df['Month'] = df['Month'].map(MONTH_MAP)
    df = df.drop(columns=['Month'])
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])
    df['Dolomite_moisture'] = df['Dolomite'] * (df['rain_sum'] - df['et0_fao_evapotranspiration'])
    for lag in [1, 12]:
        df[f'YPH_lag{lag}'] = df['YPH'].shift(lag).bfill()
    df['Compost_lag1'] = df['Compost'].shift(1).fillna(0)
    return df


def run_walk_forward(df):
    target   = 'YPH'
    features = [c for c in df.columns if c not in ['YPH', 'Year', 'Compost']]
    test_years = sorted(df[df['Year'] >= df['Year'].min() + WINDOW_SIZE]['Year'].unique())
    train_res, test_res = [], []

    for ty in test_years:
        tr_yrs = [ty - i for i in range(WINDOW_SIZE, 0, -1)]
        tr_df  = df[df['Year'].isin(tr_yrs)].copy()
        te_df  = df[df['Year'] == ty].copy()
        if tr_df.empty or te_df.empty:
            continue
        pipe = Pipeline([('imputer', SimpleImputer(strategy='median')),
                         ('scaler',  StandardScaler()),
                         ('ridge',   Ridge(alpha=ALPHA, random_state=42))])
        pipe.fit(tr_df[features], tr_df[target])
        def m(yt, yp):
            return {'RMSE': round(float(np.sqrt(mean_squared_error(yt, yp))), 3),
                    'MAE':  round(float(mean_absolute_error(yt, yp)), 3),
                    'R²':   round(float(r2_score(yt, yp)), 3)}
        train_res.append({'Year': ty, 'Train Years': f"{tr_yrs[0]}–{tr_yrs[-1]}",
                          **m(tr_df[target], pipe.predict(tr_df[features]))})
        test_res.append( {'Year': ty, 'Train Years': f"{tr_yrs[0]}–{tr_yrs[-1]}",
                          **m(te_df[target], pipe.predict(te_df[features]))})
    return train_res, test_res, features


def train_final_model(df, features):
    target = 'YPH'
    window_years = sorted(df['Year'].unique())[-WINDOW_SIZE:]
    df_w = df[df['Year'].isin(window_years)].copy()
    pipe = Pipeline([('imputer', SimpleImputer(strategy='median')),
                     ('scaler',  StandardScaler()),
                     ('ridge',   Ridge(alpha=ALPHA, random_state=42))])
    pipe.fit(df_w[features], df_w[target])
    return pipe, window_years


def render_metrics_table(results):
    rows = ""
    for r in results:
        rows += f"<tr><td>{r['Year']}</td><td>{r['Train Years']}</td><td>{r['RMSE']}</td><td>{r['MAE']}</td><td>{r['R²']}</td></tr>"
    avg = lambda k: round(float(np.mean([r[k] for r in results])), 3)
    rows += f'<tr class="avg-row"><td colspan="2">Average</td><td>{avg("RMSE")}</td><td>{avg("MAE")}</td><td>{avg("R²")}</td></tr>'
    st.markdown(f"""
    <table class="metric-table">
        <thead><tr><th>Test Year</th><th>Train Years</th><th>RMSE</th><th>MAE</th><th>R²</th></tr></thead>
        <tbody>{rows}</tbody>
    </table>""", unsafe_allow_html=True)


def delete_model(filename):
    """Delete model .pkl file and remove from registry. Cannot delete default model."""
    if filename == DEFAULT_MODEL:
        return False, "Default model cannot be deleted."
    reg = load_registry()
    # If it's active, block deletion
    if reg.get("active_model") == filename:
        return False, "Cannot delete the currently active model. Switch to another model first."
    # Remove from registry
    reg["models"] = [m for m in reg["models"] if m["filename"] != filename]
    save_registry(reg)
    # Delete file from disk
    model_file = MODELS_DIR / filename
    if model_file.exists():
        model_file.unlink()
    return True, f"{filename} deleted successfully."


# Login

if "admin_authenticated" not in st.session_state:
    st.session_state["admin_authenticated"] = False

if not st.session_state["admin_authenticated"]:
    col1, col_left_gap, col_main, col_right_gap, col5 = st.columns([0.25, 0.25, 7, 0.25, 0.25])
    with col_main:
        st.markdown('<div class="page-body">', unsafe_allow_html=True)
        st.title("GPTea Admin Panel")
        st.markdown('<p class="section-heading">Admin Login</p>', unsafe_allow_html=True)
        pw_col, _, _ = st.columns([2, 1, 3])
        with pw_col:
            password = st.text_input("Enter Admin Password", type="password")
            if st.button("Login"):
                if password == ADMIN_PASSWORD:
                    st.session_state["admin_authenticated"] = True
                    st.rerun()
                else:
                    st.markdown('<div class="box-error">Incorrect password. Please try again.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.stop()


# Main Panel
col1, col_left_gap, col_main, col_right_gap, col5 = st.columns([0.25, 0.25, 7, 0.25, 0.25])
with col_main:
    st.title("GPTea Admin Panel")

    registry = load_registry()

    # Active Model
    st.markdown('<p class="section-heading">Current Active Model</p>', unsafe_allow_html=True)

    if registry.get("active_model"):
        active = next((m for m in registry["models"] if m["filename"] == registry["active_model"]), None)
        if active:
            train_yrs  = active.get("training_years", [])
            sweet_spot = max(train_yrs) + 1 if train_yrs else "N/A"
            st.markdown(f"""
            <div class="info-card">
                <table style="width:100%;border:none;border-collapse:collapse;">
                    <tr>
                        <td style="padding:5px 0;width:55%;"><b>Model File:</b>&nbsp;<span>{active['filename']}</span></td>
                        <td style="padding:5px 0;"><b>Registered:</b>&nbsp;<span>{active.get('registered_at','N/A')}</span></td>
                    </tr>
                    <tr>
                        <td style="padding:5px 0;"><b>Training Years:</b>&nbsp;<span>{', '.join(str(y) for y in train_yrs)}</span></td>
                        <td style="padding:5px 0;"><b>Optimal Prediction Year:</b>&nbsp;<span style="color:#D95F02;font-weight:700;">{sweet_spot}</span></td>
                    </tr>
                </table>
                <span class="pill-active">ACTIVE</span>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown('<div class="box-warning"> Active model listed but details not found in registry.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="box-warning">️ No active model set. Train or select a model below.</div>', unsafe_allow_html=True)

    # Switch & Manage Models
    if registry.get("models"):
        st.markdown('<p class="section-heading">Model Registry — Switch & Manage Models</p>', unsafe_allow_html=True)

        model_labels = {
            m["filename"]: (
                f"{m['filename']}  |  Trained on: {', '.join(str(y) for y in m.get('training_years',[]))}  |  "
                f"Optimal prediction year: {max(m['training_years'])+1 if m.get('training_years') else 'N/A'}"
            ) for m in registry["models"]
        }
        cur = registry.get("active_model")
        idx = list(model_labels).index(cur) if cur in model_labels else 0

        selected = st.selectbox("Select model to activate",
                                options=list(model_labels.keys()),
                                format_func=lambda x: model_labels[x],
                                index=idx)

        sw_col, del_col, _ = st.columns([1, 1, 4])
        with sw_col:
            if st.button("Set as Active Model"):
                if (MODELS_DIR / selected).exists():
                    registry["active_model"] = selected
                    save_registry(registry)
                    st.markdown(f'<div class="box-success"> Switched to <b>{selected}</b></div>', unsafe_allow_html=True)
                    st.rerun()
                else:
                    st.markdown(f'<div class="box-error"> File not found on disk: {selected}</div>', unsafe_allow_html=True)

        with del_col:
            if selected == DEFAULT_MODEL:
                st.markdown("""
                <div style="margin-top:4px; padding: 6px 12px; background:#f0f0f0;
                     border-radius:6px; color:#888; font-size:0.88rem;
                     text-align:center; border: 1px solid #ddd;">
                    🔒 Default — cannot delete
                </div>""", unsafe_allow_html=True)
            elif selected == cur:
                st.markdown("""
                <div style="margin-top:4px; padding: 6px 12px; background:#fff3e0;
                     border-radius:6px; color:#e65100; font-size:0.88rem;
                     text-align:center; border: 1px solid #ffcc80;">
                    Switch model before deleting
                </div>""", unsafe_allow_html=True)
            else:
                if st.button(f"🗑 Delete Selected Model"):
                    success, msg = delete_model(selected)
                    if success:
                        st.markdown(f'<div class="box-success"> {msg}</div>', unsafe_allow_html=True)
                        st.rerun()
                    else:
                        st.markdown(f'<div class="box-error">{msg}</div>', unsafe_allow_html=True)

    # Upload & Retrain
    st.markdown('<p class="section-heading">Upload New Data & Retrain Model</p>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload dataset CSV (must include: Year, Month, YPH, all weather & fertilizer columns)",
        type=["csv"]
    )

    if uploaded_file:
        try:
            raw_df = pd.read_csv(uploaded_file)
            st.markdown(f'<div class="box-success"><b>{uploaded_file.name}</b> — {raw_df.shape[0]} rows, {raw_df.shape[1]} columns</div>', unsafe_allow_html=True)

            errors = validate_csv(raw_df)
            if errors:
                for e in errors:
                    st.markdown(f'<div class="box-error">{e}</div>', unsafe_allow_html=True)
            else:
                years_found = sorted(raw_df['Year'].unique())
                st.markdown(f'<div class="box-success">Validation passed — Years detected: <b>{", ".join(str(y) for y in years_found)}</b></div>', unsafe_allow_html=True)

                if "wfv_results" not in st.session_state:
                    st.session_state.update({"wfv_results": None, "wfv_df": None, "wfv_features": None})

                run_col, _ = st.columns([1.8, 5])
                with run_col:
                    if st.button("▶  Run Walk-Forward Validation"):
                        with st.spinner("Running walk-forward validation…"):
                            processed = preprocess(raw_df)
                            tr, te, feats = run_walk_forward(processed)
                            st.session_state.update({"wfv_results": (tr, te), "wfv_df": processed, "wfv_features": feats})

                if st.session_state.get("wfv_results"):
                    tr, te = st.session_state["wfv_results"]
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("### Walk-Forward Validation Results")
                    tab_test, tab_train = st.tabs(["Test Metrics (Unseen Data)", " Train Metrics"])
                    with tab_test:
                        render_metrics_table(te)
                    with tab_train:
                        render_metrics_table(tr)

                    st.markdown('<div class="box-warning"> <b>Review results carefully before confirming.</b> The final model will be trained on the last 3 years and set as active.</div>', unsafe_allow_html=True)

                    cf_col, _ = st.columns([1.8, 5])
                    with cf_col:
                        if st.button(" Confirm & Train Final Model"):
                            with st.spinner("Training final model…"):
                                pipe, window_years = train_final_model(
                                    st.session_state["wfv_df"],
                                    st.session_state["wfv_features"]
                                )
                                model_name = f"Ridge_Model_{'_'.join(str(y) for y in window_years)}.pkl"
                                joblib.dump(pipe, MODELS_DIR / model_name)

                                reg = load_registry()
                                reg["models"] = [m for m in reg["models"] if m["filename"] != model_name]
                                reg["models"].append({
                                    "filename": model_name,
                                    "training_years": [int(y) for y in window_years],
                                    "registered_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                })
                                reg["active_model"] = model_name
                                save_registry(reg)
                                st.session_state.update({"wfv_results": None, "wfv_df": None, "wfv_features": None})

                                sweet = max(window_years) + 1
                                components.html(f"""
                                <script>
                                    alert("Model Trained and Activated!\\n\\nFile: {model_name}\\nTrained on: {', '.join(str(y) for y in window_years)}\\nOptimal Year: {sweet}");
                                </script>
                                """, height=0)
                                st.session_state["admin_authenticated"] = True
                                st.rerun()

        except Exception as e:
            st.markdown(f'<div class="box-error"> Error reading CSV: {str(e)}</div>', unsafe_allow_html=True)

    # Logout
    st.markdown("<br>", unsafe_allow_html=True)
    lo_col, _ = st.columns([1, 6])
    with lo_col:
        if st.button("Logout"):
            st.session_state["admin_authenticated"] = False
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div class="footer">
    <div class="flex-around">
        <div>
            <b><p style="font-size:25px">GPTea</b></p>
           <p style="font-size:18px; max-width: 300px; line-height: 1.4;">
                AI platform for Sri Lankan tea industrials to predict tea yield,
                detect diseases in leaves and clarify doubts with Chatbot.
           </p>
        </div>
        <div>
            <b><p style="font-size:20px">Quick Links</b><br></p>
            <p>
                <a href="#" class="footer-link">Home</a> |
                <a href="#" class="footer-link">Innovation</a> |
                <a href="#" class="footer-link">Research Team</a>
            </p>
        </div>
        <div>
            <b><p>Resources</b><br></p>
            <p>
                <a href="#" class="footer-link">Documentation</a> |
                <a href="#" class="footer-link">Research Paper</a> |
                <a href="#" class="footer-link">Dataset</a>
            </p>
        </div>
    </div>
    <br>
    <p class="center-text">© 2026 GPTea Research Project. All rights reserved.</p>
</div>
""", unsafe_allow_html=True)