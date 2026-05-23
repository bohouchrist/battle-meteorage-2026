"""Démo Météorage 2026 — simulation visuelle d'alertes foudre en direct.

Single-page app, theme storm.
"""
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore", category=FutureWarning)

# ═════════════════════════════════════════════════════════════════
# CONSTANTES
# ═════════════════════════════════════════════════════════════════
ROOT       = Path(__file__).resolve().parent.parent
JURY_PATH  = ROOT / "segment_alerts_all_airports_eval.csv"
MODELS_DIR = ROOT / "models"

THETA_DEFAULT = 0.30
RATIO         = 0.98
MAX_GAP_MIN   = 30
DIST_3KM      = 3.0

FEATURES = [
    "h_cos", "h_sin", "doy_cos", "doy_sin", "saison",
    "dist_centre", "dist_avg_5", "dist_min_so_far",
    "silence_min", "freq_5min", "rang", "rang_norm",
    "airport_enc",
]

SAISON_LABELS = {1: "❄️ Hiver", 2: "🌸 Printemps", 3: "☀️ Été", 4: "🍂 Automne"}
SAISON_GRADIENT = {
    1: ("#3B82F6", "#1E40AF"),   # winter blue
    2: ("#A7F3D0", "#10B981"),   # spring green
    3: ("#FCD34D", "#D97706"),   # summer orange
    4: ("#F59E0B", "#92400E"),   # autumn ocre
}

AIRPORT_INFO = {
    "Ajaccio":  {"icao": "LFKJ", "country": "🇫🇷 France",   "coords": "41.92°N, 8.80°E"},
    "Bastia":   {"icao": "LFKB", "country": "🇫🇷 France",   "coords": "42.55°N, 9.48°E"},
    "Biarritz": {"icao": "LFBZ", "country": "🇫🇷 France",   "coords": "43.47°N, 1.53°W"},
    "Nantes":   {"icao": "LFRS", "country": "🇫🇷 France",   "coords": "47.16°N, 1.61°W"},
    "Pise":     {"icao": "LIRP", "country": "🇮🇹 Italie",   "coords": "43.68°N, 10.39°E"},
}

# ═════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="⚡ Météorage Live",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═════════════════════════════════════════════════════════════════
# CSS — design storm/dark
# ═════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── BASE ───────────────────────────────────────────────────── */
.stApp {
  background: radial-gradient(ellipse at top, #1e293b 0%, #0f172a 60%, #020617 100%);
}
.block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 1500px; }

/* texte par défaut clair */
.stApp, .stApp p, .stApp div, .stApp span, .stApp label { color: #E5E7EB; }
h1, h2, h3, h4 { color: #F8FAFC !important; }

/* ── HEADER STREAMLIT (bande blanche en haut) ──────────────── */
header[data-testid="stHeader"] {
  background: linear-gradient(90deg, #0f172a 0%, #1e293b 50%, #0f172a 100%) !important;
  border-bottom: 1px solid #334155 !important;
}
header[data-testid="stHeader"] * { color: #94A3B8 !important; }
[data-testid="stToolbar"] { background-color: transparent !important; }
[data-testid="stStatusWidget"] { color: #94A3B8 !important; }
[data-testid="stDecoration"] { background: linear-gradient(90deg, #FBBF24, #F59E0B) !important; }

/* ── HERO ───────────────────────────────────────────────────── */
.hero {
  background: linear-gradient(135deg, #1e3a8a 0%, #581c87 50%, #be185d 100%);
  padding: 28px 40px;
  border-radius: 16px;
  margin-bottom: 24px;
  box-shadow: 0 10px 40px rgba(91, 33, 182, 0.3);
  position: relative;
  overflow: hidden;
}
.hero::before {
  content: "⚡";
  position: absolute;
  font-size: 200px;
  right: -20px; top: -50px;
  opacity: 0.08;
  transform: rotate(15deg);
}
.hero h1 {
  font-size: 2.8rem !important;
  font-weight: 900 !important;
  margin: 0 !important;
  background: linear-gradient(90deg, #FBBF24 0%, #F59E0B 50%, #FCD34D 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  letter-spacing: -1px;
}
.hero .subtitle {
  font-size: 1.05rem;
  color: #CBD5E1;
  margin-top: 4px;
  font-weight: 400;
}

/* ── AIRPORT CARD (compact) ─────────────────────────────────── */
.airport-card {
  background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
  border: 1px solid #334155;
  border-radius: 12px;
  padding: 14px 18px;
  height: 100%;
  box-shadow: 0 4px 12px rgba(0,0,0,0.3);
}
.airport-card .header-row {
  display: flex; align-items: center; gap: 12px;
  padding-bottom: 10px;
  border-bottom: 1px solid #334155;
  margin-bottom: 10px;
}
.airport-card .plane {
  font-size: 1.8rem;
  flex-shrink: 0;
  filter: drop-shadow(0 2px 6px rgba(251,191,36,0.4));
}
.airport-card .name-block { flex: 1; min-width: 0; }
.airport-card .name {
  font-size: 1.3rem;
  font-weight: 900;
  color: #F8FAFC;
  letter-spacing: -0.5px;
  line-height: 1.1;
}
.airport-card .icao {
  font-size: 0.7rem;
  color: #94A3B8;
  letter-spacing: 2px;
  font-weight: 700;
  margin-top: 2px;
}
.airport-card .alert-tag {
  background: #334155;
  color: #FBBF24;
  padding: 4px 10px;
  border-radius: 6px;
  font-size: 0.75rem;
  font-weight: 700;
  white-space: nowrap;
}
.airport-card .details { display: grid; grid-template-columns: 1fr 1fr; gap: 6px 14px; }
.airport-card .detail-row {
  display: flex; justify-content: space-between;
  font-size: 0.82rem;
  padding: 3px 0;
}
.airport-card .detail-row .k { color: #94A3B8; }
.airport-card .detail-row .v { color: #F1F5F9; font-weight: 600; }

/* ── BIG NUMBER CARDS ───────────────────────────────────────── */
.bignum {
  border-radius: 16px;
  padding: 20px 24px;
  text-align: center;
  height: 100%;
  box-shadow: 0 4px 16px rgba(0,0,0,0.3);
  border: 1px solid #334155;
}
.bignum.green { background: linear-gradient(135deg, #064e3b 0%, #022c22 100%); border-color: #10B981; }
.bignum.amber { background: linear-gradient(135deg, #78350f 0%, #451a03 100%); border-color: #F59E0B; }
.bignum.red   { background: linear-gradient(135deg, #7f1d1d 0%, #450a0a 100%); border-color: #EF4444; }
.bignum.blue  { background: linear-gradient(135deg, #1e3a8a 0%, #1e1b4b 100%); border-color: #3B82F6; }
.bignum.gray  { background: linear-gradient(135deg, #1f2937 0%, #111827 100%); border-color: #6B7280; }

.bignum .label {
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 2px;
  color: #94A3B8;
  font-weight: 700;
  margin-bottom: 6px;
}
.bignum .value {
  font-size: 2.6rem;
  font-weight: 900;
  color: #F8FAFC;
  line-height: 1;
  letter-spacing: -1px;
  font-variant-numeric: tabular-nums;
}
.bignum .unit {
  font-size: 1rem;
  color: #94A3B8;
  margin-left: 4px;
}
.bignum .extra {
  font-size: 0.8rem;
  color: #CBD5E1;
  margin-top: 6px;
}

/* ── STATUT BANNER ──────────────────────────────────────────── */
.status-banner {
  border-radius: 12px;
  padding: 18px 28px;
  margin: 16px 0;
  font-size: 1.1rem;
  font-weight: 600;
  display: flex;
  align-items: center;
  gap: 16px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.3);
}
.status-banner.active {
  background: linear-gradient(90deg, #064e3b 0%, #022c22 100%);
  border-left: 6px solid #10B981;
}
.status-banner.wait {
  background: linear-gradient(90deg, #78350f 0%, #451a03 100%);
  border-left: 6px solid #F59E0B;
}
.status-banner .dot {
  width: 14px; height: 14px;
  border-radius: 50%;
  animation: pulse 1.5s infinite;
}
.status-banner .dot.green { background: #10B981; box-shadow: 0 0 16px #10B981; }
.status-banner .dot.amber { background: #F59E0B; box-shadow: 0 0 16px #F59E0B; }
@keyframes pulse {
  0%, 100% { opacity: 1; transform: scale(1); }
  50%      { opacity: 0.6; transform: scale(1.2); }
}

/* ── FLASH ECLAIR ──────────────────────────────────────────── */
.lightning-flash {
  background: linear-gradient(135deg, #FBBF24 0%, #F59E0B 50%, #DC2626 100%);
  border-radius: 16px;
  padding: 16px 24px;
  text-align: center;
  margin: 12px 0;
  box-shadow: 0 0 30px rgba(251, 191, 36, 0.5),
              0 0 60px rgba(245, 158, 11, 0.3);
  animation: flash 1.5s ease-out;
  position: relative;
  overflow: hidden;
}
@keyframes flash {
  0%   { transform: scale(0.95); box-shadow: 0 0 80px rgba(251,191,36,0.9); }
  30%  { transform: scale(1.02); }
  100% { transform: scale(1); }
}
.lightning-flash .icon {
  font-size: 3rem;
  margin-right: 12px;
  display: inline-block;
  animation: shake 0.4s ease-in-out;
  filter: drop-shadow(0 0 16px #fff);
}
@keyframes shake {
  0%, 100% { transform: rotate(0); }
  25% { transform: rotate(-10deg) scale(1.1); }
  75% { transform: rotate(10deg) scale(1.1); }
}
.lightning-flash .text {
  display: inline-block;
  font-size: 1.4rem;
  font-weight: 900;
  color: #1f2937;
  vertical-align: middle;
  letter-spacing: 0.5px;
}
.lightning-flash .meta {
  display: block;
  font-size: 0.95rem;
  color: #1f2937;
  margin-top: 4px;
  font-weight: 700;
}

/* ── VERIFICATION CARD ──────────────────────────────────────── */
.verif-card {
  border-radius: 10px;
  padding: 14px 18px;
  margin: 8px 0;
  font-size: 0.95rem;
  display: flex;
  align-items: center;
  gap: 14px;
}
.verif-card.ok   { background: rgba(16,185,129,0.12); border-left: 4px solid #10B981; }
.verif-card.fail { background: rgba(239,68,68,0.12);  border-left: 4px solid #EF4444; }
.verif-card.idle { background: rgba(107,114,128,0.12); border-left: 4px solid #6B7280; }
.verif-card .badge {
  font-weight: 900;
  font-size: 0.85rem;
  padding: 4px 10px;
  border-radius: 6px;
  white-space: nowrap;
}
.verif-card.ok   .badge { background: #10B981; color: white; }
.verif-card.fail .badge { background: #EF4444; color: white; }
.verif-card.idle .badge { background: #6B7280; color: white; }

/* ── SIDEBAR ────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0f172a 0%, #020617 100%);
  border-right: 1px solid #1e293b;
}
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4 {
  color: #F8FAFC !important;
}
section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] .stCaption {
  color: #CBD5E1 !important;
}

/* ── SELECTBOX (sidebar + main) ──────────────────────────────── */
/* Conteneur extérieur de la selectbox */
.stSelectbox > div > div {
  background-color: #1e293b !important;
  border: 1px solid #334155 !important;
  color: #F1F5F9 !important;
}
/* Texte affiché (valeur sélectionnée) */
.stSelectbox div[data-baseweb="select"] > div,
.stSelectbox div[data-baseweb="select"] span,
.stSelectbox div[data-baseweb="select"] input {
  color: #F1F5F9 !important;
  background-color: transparent !important;
}
/* Icône chevron */
.stSelectbox svg { color: #94A3B8 !important; fill: #94A3B8 !important; }
/* Liste déroulante (popover) */
div[data-baseweb="popover"] ul, div[data-baseweb="menu"] {
  background-color: #1e293b !important;
  border: 1px solid #334155 !important;
}
div[data-baseweb="popover"] li, div[data-baseweb="menu"] li {
  color: #F1F5F9 !important;
}
div[data-baseweb="popover"] li:hover, div[data-baseweb="menu"] li:hover {
  background-color: #334155 !important;
}

/* ── SLIDER ─────────────────────────────────────────────────── */
.stSlider [data-baseweb="slider"] { color: #F1F5F9 !important; }
.stSlider [role="slider"] {
  background-color: #FBBF24 !important;
  border: 2px solid #92400E !important;
  box-shadow: 0 0 12px rgba(251,191,36,0.5) !important;
}
.stSlider [data-baseweb="slider"] > div > div > div {
  background: linear-gradient(90deg, #F59E0B, #FBBF24) !important;
}
.stSlider [data-testid="stTickBarMin"],
.stSlider [data-testid="stTickBarMax"] { color: #94A3B8 !important; }

/* ── BUTTONS ────────────────────────────────────────────────── */
.stButton > button {
  border-radius: 10px;
  font-weight: 700;
  background-color: #1e293b !important;
  color: #F1F5F9 !important;
  border: 1px solid #334155 !important;
  transition: all 0.2s;
}
.stButton > button:hover {
  background-color: #334155 !important;
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(251, 191, 36, 0.3);
  border-color: #FBBF24 !important;
}
.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, #FBBF24 0%, #F59E0B 100%) !important;
  color: #0F172A !important;
  border: none !important;
  font-weight: 900 !important;
}
.stButton > button[kind="primary"]:hover {
  background: linear-gradient(135deg, #FCD34D 0%, #FBBF24 100%) !important;
  box-shadow: 0 4px 20px rgba(251, 191, 36, 0.6) !important;
}

/* ── PROGRESS ───────────────────────────────────────────────── */
.stProgress > div > div { background: linear-gradient(90deg, #FBBF24, #F59E0B) !important; }
.stProgress > div > div > div { background-color: #1e293b !important; }

/* ── ALERT / WARNING / SUCCESS / INFO ─────────────────────────── */
.stAlert {
  background-color: #1e293b !important;
  color: #F1F5F9 !important;
  border-radius: 10px !important;
}
.stAlert [data-testid="stAlertContentInfo"]    { color: #93C5FD !important; }
.stAlert [data-testid="stAlertContentSuccess"] { color: #6EE7B7 !important; }
.stAlert [data-testid="stAlertContentWarning"] { color: #FCD34D !important; }
.stAlert [data-testid="stAlertContentError"]   { color: #FCA5A5 !important; }

/* ── EXPANDER ───────────────────────────────────────────────── */
.streamlit-expanderHeader, [data-testid="stExpander"] summary {
  background-color: #1e293b !important;
  color: #F1F5F9 !important;
  border: 1px solid #334155 !important;
  border-radius: 8px !important;
}

/* ── DATAFRAME ──────────────────────────────────────────────── */
.stDataFrame, .stDataFrame [data-testid="stTable"] {
  background-color: #1e293b !important;
  color: #F1F5F9 !important;
}

/* ── PLOTLY ─────────────────────────────────────────────────── */
.js-plotly-plot .plot-container { background: transparent !important; }

/* ── FOOTER STRIP ──────────────────────────────────────────── */
.footer-strip {
  background: rgba(15, 23, 42, 0.5);
  border-top: 1px solid #1e293b;
  padding: 12px 20px;
  margin-top: 24px;
  border-radius: 8px;
  font-size: 0.85rem;
  color: #94A3B8;
  text-align: center;
}

/* ── LIVE PULSE (sur valeur en temps réel) ────────────────── */
.live-dot {
  display: inline-block;
  width: 8px; height: 8px;
  border-radius: 50%;
  background: #10B981;
  box-shadow: 0 0 12px #10B981;
  animation: livepulse 1.2s infinite;
  vertical-align: middle;
  margin-right: 6px;
}
@keyframes livepulse {
  0%, 100% { opacity: 1; transform: scale(1); }
  50%      { opacity: 0.5; transform: scale(1.5); }
}

/* ── BIGNUM avec animation au refresh ──────────────────────── */
.bignum .value {
  animation: fadeInScale 0.4s ease-out;
}
@keyframes fadeInScale {
  0%   { opacity: 0.3; transform: scale(0.92); }
  100% { opacity: 1;   transform: scale(1); }
}
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════
# CHARGEMENT
# ═════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="⚡ Chargement du modèle…")
def load_model():
    gbs    = joblib.load(MODELS_DIR / "gbs_v7_model.pkl")
    scaler = joblib.load(MODELS_DIR / "gbs_v7_scaler.pkl")
    le     = joblib.load(MODELS_DIR / "gbs_v7_label_encoder.pkl")
    return gbs, scaler, le


@st.cache_data(show_spinner="🌩 Scoring jury 2023-2025…")
def load_and_score_jury():
    df = pd.read_csv(JURY_PATH)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df[df["alert_id"].notna()].copy()
    df = df.rename(columns={"alert_id": "airport_alert_id"})
    df = df.sort_values(["airport", "airport_alert_id", "date"]).reset_index(drop=True)

    g   = df.groupby(["airport", "airport_alert_id"])
    h   = df["date"].dt.hour + df["date"].dt.minute / 60
    doy = df["date"].dt.dayofyear
    df["h_cos"]   = np.cos(2 * np.pi * h / 24)
    df["h_sin"]   = np.sin(2 * np.pi * h / 24)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365)
    df["saison"]  = ((df["date"].dt.month % 12) // 3) + 1
    prev = g["date"].shift(1)
    df["silence_min"] = ((df["date"] - prev).dt.total_seconds() / 60).fillna(30).clip(0, 60)
    df["freq_5min"]   = (1 / df["silence_min"].clip(lower=0.5)).clip(upper=10)
    df["dist_centre"]     = df["dist"]
    df["dist_avg_5"]      = g["dist"].transform(lambda x: x.rolling(5, min_periods=1).mean())
    df["dist_min_so_far"] = g["dist"].cummin()
    df["rang"]            = g.cumcount()
    df["rang_norm"]       = df["rang"] / g["rang"].transform("max").clip(lower=1)

    gbs, scaler, le = load_model()
    df["airport_enc"] = le.transform(df["airport"])
    for col in FEATURES:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    X        = scaler.transform(df[FEATURES].values.astype(float))
    surv_fns = gbs.predict_survival_function(X)
    s30      = np.array([float(fn(MAX_GAP_MIN)) for fn in surv_fns])
    t_stars  = np.full(len(surv_fns), float(MAX_GAP_MIN))
    for i, (fn, s) in enumerate(zip(surv_fns, s30)):
        if s <= 0:
            continue
        idx = np.searchsorted(-fn.y, -(s / RATIO), side="left")
        if idx < len(fn.x):
            t_stars[i] = min(float(fn.x[idx]), float(MAX_GAP_MIN))
    df["confiance"]    = s30
    df["horizon_min"]  = t_stars
    df["fin_predite"]  = df["date"] + pd.to_timedelta(t_stars, unit="m")
    df["alert_saison"] = g["saison"].transform("first")
    return df


gbs_model, _, _ = load_model()
df_jury         = load_and_score_jury()


@st.cache_data(show_spinner=False)
def compute_global_stats(_df: pd.DataFrame, theta: float) -> dict:
    """Stats globales jury 2023-2025 pour un θ donné."""
    g = _df.groupby(["airport", "airport_alert_id"])
    t_regles = g["date"].max() + pd.Timedelta(minutes=MAX_GAP_MIN)
    above = _df[_df["confiance"] > theta]
    n_pred_applied = len(above)

    merged = t_regles.rename("t_regle").to_frame()
    if len(above):
        t_modeles = (above.groupby(["airport", "airport_alert_id"])["fin_predite"]
                          .min().rename("fin_predite"))
        merged = merged.join(t_modeles, how="left")
    else:
        merged["fin_predite"] = pd.NaT
    merged["t_modele"]     = merged["fin_predite"].combine_first(merged["t_regle"])
    merged["gain_min"]     = ((merged["t_regle"] - merged["t_modele"])
                              .dt.total_seconds().clip(lower=0) / 60)
    merged["modele_actif"] = merged["fin_predite"].notna()

    t_pred_map = merged[["t_modele"]].reset_index()
    z3 = _df[_df["dist"] < DIST_3KM][
        ["airport", "airport_alert_id", "date"]
    ].copy()
    z3 = z3.merge(t_pred_map, on=["airport", "airport_alert_id"], how="left")
    z3["manque"] = z3["date"] >= z3["t_modele"]

    return {
        "n_alertes":      len(merged),
        "n_actif":        int(merged["modele_actif"].sum()),
        "n_inactif":      int((~merged["modele_actif"]).sum()),
        "n_pred_applied": n_pred_applied,
        "n_strikes":      int(len(_df)),
        "gain_h":         float(merged["gain_min"].sum() / 60),
        "gain_moy_min":   float(merged.loc[merged["modele_actif"], "gain_min"].mean()) if merged["modele_actif"].any() else 0.0,
        "n_L3":           int(len(z3)),
        "n_manques":      int(z3["manque"].sum()),
        "risk_pct":       100 * int(z3["manque"].sum()) / len(z3) if len(z3) else 0.0,
    }


# ═════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚡ Météorage Live")
    st.caption("GBS Survival · jury 2023-2025")
    st.divider()

    st.markdown("### ✈️  Aéroport")
    airports = sorted(df_jury["airport"].unique())
    airport  = st.selectbox("Aéroport", airports, label_visibility="collapsed")

    st.markdown("### 🌦  Saison")
    saisons_dispo = sorted(
        df_jury[df_jury["airport"] == airport]["alert_saison"].unique().astype(int)
    )
    saison_opt = ["Toutes"] + [SAISON_LABELS[s] for s in saisons_dispo]
    saison_sel = st.selectbox("Saison", saison_opt, label_visibility="collapsed")
    saison_num = (None if saison_sel == "Toutes"
                  else {v: k for k, v in SAISON_LABELS.items()}[saison_sel])

    st.markdown("### 🌩  Alerte")
    df_filtered = df_jury[df_jury["airport"] == airport]
    if saison_num is not None:
        df_filtered = df_filtered[df_filtered["alert_saison"] == saison_num]
    summary = (df_filtered.groupby("airport_alert_id")
               .agg(n=("date", "count"),
                    dur=("date",
                         lambda x: round((x.max() - x.min()).total_seconds() / 60, 0)),
                    cmax=("confiance", "max"))
               .reset_index().sort_values("cmax", ascending=False))

    if len(summary) == 0:
        st.warning("Pas d'alerte avec ces filtres.")
        st.stop()

    opts = [
        f"#{r['airport_alert_id']}  ·  {int(r['n'])} ⚡  ·  "
        f"{int(r['dur'])} min  ·  S(30) max = {r['cmax']:.2f}"
        for _, r in summary.iterrows()
    ]
    alert_choice = st.selectbox(f"{len(summary)} alertes disponibles",
                                opts, label_visibility="collapsed")
    alert_id = alert_choice.split("·")[0].replace("#", "").strip()

    st.divider()

    st.markdown("### 🎚  Seuil de confiance θ")
    theta = st.slider(
        "θ", 0.00, 0.99, THETA_DEFAULT, 0.01, label_visibility="collapsed",
        help="Défaut 0.30 (calibré sur test). Monte pour plus de sécurité.",
    )
    if theta > THETA_DEFAULT + 0.005:
        st.markdown(f"<span style='color:#10B981'>🛡  Mode prudent (+{theta-THETA_DEFAULT:.2f})</span>",
                    unsafe_allow_html=True)
    elif theta < THETA_DEFAULT - 0.005:
        st.markdown(f"<span style='color:#F59E0B'>⚠️  Mode permissif ({theta-THETA_DEFAULT:+.2f})</span>",
                    unsafe_allow_html=True)
    else:
        st.markdown(f"<span style='color:#A78BFA'>✓  θ de production</span>",
                    unsafe_allow_html=True)

    st.divider()
    st.markdown("### ⏱  Cadence")
    delay_s = st.select_slider(
        "Délai entre éclairs", options=[1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
        value=2.5, format_func=lambda x: f"{x:.1f} s",
        label_visibility="collapsed",
    )

    st.divider()
    st.caption(
        "**Performance jury**\n\n"
        "103,5 h économisées\n\n"
        "0,10 % de risque\n\n"
        "20× sous la limite 2 %"
    )


# ═════════════════════════════════════════════════════════════════
# DATA POUR L'ALERTE
# ═════════════════════════════════════════════════════════════════
alert_df = (df_jury[
    (df_jury["airport"] == airport) &
    (df_jury["airport_alert_id"] == alert_id)
].sort_values("date").reset_index(drop=True).copy())

n_strikes    = len(alert_df)
t_regle_demo = alert_df["date"].max() + pd.Timedelta(minutes=MAX_GAP_MIN)
saison_id    = int(alert_df["alert_saison"].iloc[0])

key = (airport, alert_id)
if st.session_state.get("alert_key") != key:
    st.session_state["alert_key"] = key
    st.session_state["idx"]       = 0
    st.session_state["playing"]   = False

playing = st.session_state.get("playing", False)
idx     = min(st.session_state.get("idx", 0), n_strikes - 1)
seen    = alert_df.iloc[: idx + 1].copy()
current = alert_df.iloc[idx]


# ═════════════════════════════════════════════════════════════════
# HERO
# ═════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <h1>⚡ MÉTÉORAGE LIVE</h1>
  <div class="subtitle">Simulation d'alerte foudre en direct · Battle Data 2026 · GBS Survival v7</div>
</div>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════
# STRIP — PERFORMANCE GLOBALE JURY 2023-2025 (réactif à θ)
# ═════════════════════════════════════════════════════════════════
gstats = compute_global_stats(df_jury, theta)
pct_actif = 100 * gstats["n_actif"] / max(gstats["n_alertes"], 1)
pct_pred  = 100 * gstats["n_pred_applied"] / max(gstats["n_strikes"], 1)
risk_ok   = gstats["risk_pct"] < 2.0

st.markdown(f"""
<div style='background:linear-gradient(90deg,#0f172a 0%,#1e293b 50%,#0f172a 100%);
            border:1px solid #334155;border-radius:14px;padding:18px 24px;
            margin-bottom:20px;box-shadow:0 4px 12px rgba(0,0,0,0.3);'>
  <div style='font-size:0.7rem;text-transform:uppercase;letter-spacing:3px;
              color:#94A3B8;font-weight:700;margin-bottom:12px;text-align:center;'>
    🏆 Performance modèle de production · jury 2023-2025 · θ = {theta:.2f}
  </div>
  <div style='display:flex;justify-content:space-around;gap:16px;flex-wrap:wrap;'>
    <div style='text-align:center;flex:1;min-width:140px;'>
      <div style='font-size:2rem;font-weight:900;color:#10B981;line-height:1;'>
        {gstats['gain_h']:.1f}<span style='font-size:1rem;color:#94A3B8;'> h</span>
      </div>
      <div style='font-size:0.75rem;color:#94A3B8;margin-top:4px;'>Temps économisé</div>
    </div>
    <div style='text-align:center;flex:1;min-width:140px;'>
      <div style='font-size:2rem;font-weight:900;color:#10B981;line-height:1;'>
        {gstats['n_actif']}<span style='font-size:1rem;color:#94A3B8;'> / {gstats['n_alertes']}</span>
      </div>
      <div style='font-size:0.75rem;color:#94A3B8;margin-top:4px;'>
        Alertes levées par modèle ({pct_actif:.0f} %)
      </div>
    </div>
    <div style='text-align:center;flex:1;min-width:140px;'>
      <div style='font-size:2rem;font-weight:900;color:#6B7280;line-height:1;'>
        {gstats['n_inactif']}<span style='font-size:1rem;color:#94A3B8;'> / {gstats['n_alertes']}</span>
      </div>
      <div style='font-size:0.75rem;color:#94A3B8;margin-top:4px;'>
        Alertes en règle 30 min ({100-pct_actif:.0f} %)
      </div>
    </div>
    <div style='text-align:center;flex:1;min-width:140px;'>
      <div style='font-size:2rem;font-weight:900;color:#3B82F6;line-height:1;'>
        {gstats['n_pred_applied']:,}<span style='font-size:0.9rem;color:#94A3B8;'> ⚡</span>
      </div>
      <div style='font-size:0.75rem;color:#94A3B8;margin-top:4px;'>
        Prédictions appliquées ({pct_pred:.1f} % des {gstats['n_strikes']:,} éclairs)
      </div>
    </div>
    <div style='text-align:center;flex:1;min-width:140px;'>
      <div style='font-size:2rem;font-weight:900;color:{"#10B981" if risk_ok else "#EF4444"};line-height:1;'>
        {gstats['risk_pct']:.2f}<span style='font-size:1rem;color:#94A3B8;'> %</span>
      </div>
      <div style='font-size:0.75rem;color:#94A3B8;margin-top:4px;'>
        Risque · {gstats['n_manques']} / {gstats['n_L3']} ({'✓ &lt; 2 %' if risk_ok else '⚠ &gt; 2 %'})
      </div>
    </div>
    <div style='text-align:center;flex:1;min-width:140px;'>
      <div style='font-size:2rem;font-weight:900;color:#A78BFA;line-height:1;'>
        {gstats['gain_moy_min']:.1f}<span style='font-size:1rem;color:#94A3B8;'> min</span>
      </div>
      <div style='font-size:0.75rem;color:#94A3B8;margin-top:4px;'>
        Gain moyen / alerte levée
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════
# ROW 1 : AIRPORT CARD  +  3 BIG NUMBERS
# ═════════════════════════════════════════════════════════════════
info = AIRPORT_INFO.get(airport, {"icao":"---","country":"","coords":""})
g_start, g_end = SAISON_GRADIENT[saison_id]

col_air, col_kpi1, col_kpi2, col_kpi3 = st.columns([1.1, 1, 1, 1])

amp_cur = float(abs(current.get("amplitude", 0))) if "amplitude" in current.index else 0.0
with col_air:
    st.markdown(f"""
    <div class="airport-card" style="background:linear-gradient(135deg,{g_start}26 0%,{g_end}18 100%);
                                     border-color:{g_end}66;">
      <div class="header-row">
        <div class="plane">✈️</div>
        <div class="name-block">
          <div class="name">{airport}</div>
          <div class="icao">{info['icao']} · {SAISON_LABELS.get(saison_id,'')}</div>
        </div>
        <div class="alert-tag">#{alert_id}</div>
      </div>
      <div class="details">
        <div class="detail-row"><span class="k">Éclair</span>
          <span class="v">{idx+1} / {n_strikes}</span></div>
        <div class="detail-row"><span class="k">Heure</span>
          <span class="v">{current['date'].strftime('%H:%M:%S')}</span></div>
        <div class="detail-row"><span class="k">Distance</span>
          <span class="v" style="color:{'#EF4444' if current['dist']<DIST_3KM else '#F1F5F9'}">
            {current['dist']:.1f} km</span></div>
        <div class="detail-row"><span class="k">Amplitude</span>
          <span class="v">{amp_cur:.0f} kA</span></div>
        <div class="detail-row"><span class="k">T*_i</span>
          <span class="v">{current['horizon_min']:.1f} min</span></div>
        <div class="detail-row"><span class="k">Fin si actif</span>
          <span class="v">{current['fin_predite'].strftime('%H:%M:%S')}</span></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# Décision globale en direct
seen["gap_next_min"] = ((seen["date"].shift(-1) - seen["date"]).dt.total_seconds() / 60)
seen["dist_next"]    = seen["dist"].shift(-1)
seen["model_active"] = seen["confiance"] > theta
seen["pred_validable"] = seen["gap_next_min"].notna() & seen["model_active"]
seen["pred_violated"]  = (seen["pred_validable"]
                          & (seen["gap_next_min"] <= seen["horizon_min"])
                          & (seen["dist_next"] < DIST_3KM))
seen["pred_ok"]        = seen["pred_validable"] & ~seen["pred_violated"]
above_seen = seen[seen["model_active"]]

if len(above_seen):
    t_modele = above_seen["fin_predite"].min()
    gain_min = max((t_regle_demo - t_modele).total_seconds() / 60, 0)
    is_active = True
else:
    t_modele = t_regle_demo
    gain_min = 0.0
    is_active = False

# KPI 1 — Confiance courante (avec live indicator)
conf_color = "green" if current["confiance"] > theta else "gray"
conf_status = ('✓ &gt; θ = '+f'{theta:.2f}') if current['confiance']>theta else ('⊘ ≤ θ = '+f'{theta:.2f}')
with col_kpi1:
    st.markdown(f"""
    <div class="bignum {conf_color}">
      <div class="label"><span class="live-dot"></span>Confiance S(30) · éclair #{idx+1}</div>
      <div class="value" key="{idx}_{theta}">{current['confiance']:.3f}</div>
      <div class="extra">{conf_status}</div>
    </div>
    """, unsafe_allow_html=True)

# KPI 2 — Gain
with col_kpi2:
    color2 = "green" if gain_min > 0 else "gray"
    st.markdown(f"""
    <div class="bignum {color2}">
      <div class="label">Temps économisé</div>
      <div class="value">{gain_min:.0f}<span class="unit">min</span></div>
      <div class="extra">vs règle Météorage 30 min</div>
    </div>
    """, unsafe_allow_html=True)

# KPI 3 — Fin prédite
with col_kpi3:
    color3 = "green" if is_active else "amber"
    fin_h  = t_modele.strftime("%H:%M:%S")
    label3 = "Levée prédite" if is_active else "Règle 30 min"
    st.markdown(f"""
    <div class="bignum {color3}">
      <div class="label">{label3}</div>
      <div class="value">{fin_h.split(':')[0]}:{fin_h.split(':')[1]}<span class="unit">:{fin_h.split(':')[2]}</span></div>
      <div class="extra">{'Modèle ACTIF' if is_active else 'Modèle en attente'}</div>
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════
# STATUS BANNER (anim pulse) — gère la fin de simulation
# ═════════════════════════════════════════════════════════════════
simulation_done = (idx >= n_strikes - 1)

if simulation_done and is_active:
    st.markdown(f"""
    <div class="status-banner active">
      <div class="dot green"></div>
      <div>
        <b>🏁 SIMULATION TERMINÉE — ALERTE LEVÉE</b> &nbsp;·&nbsp;
        Tous les éclairs ont été observés. Levée effective à
        <b>{t_modele.strftime('%H:%M:%S UTC')}</b>
        &nbsp;·&nbsp; Gain final : <b>{gain_min:.0f} min</b>
      </div>
    </div>
    """, unsafe_allow_html=True)
elif simulation_done and not is_active:
    st.markdown(f"""
    <div class="status-banner wait">
      <div class="dot amber"></div>
      <div>
        <b>🏁 SIMULATION TERMINÉE — RÈGLE 30 MIN APPLIQUÉE</b> &nbsp;·&nbsp;
        Aucun éclair n'a dépassé θ = {theta:.2f}.
        Alerte fermée à <b>{t_regle_demo.strftime('%H:%M:%S')}</b>
      </div>
    </div>
    """, unsafe_allow_html=True)
elif is_active:
    st.markdown(f"""
    <div class="status-banner active">
      <div class="dot green"></div>
      <div>
        <b>🟢 MODÈLE ACTIF</b> &nbsp;·&nbsp;
        Levée prédite à <b>{t_modele.strftime('%H:%M:%S UTC')}</b>
        &nbsp;·&nbsp; Gain : <b>{gain_min:.0f} min</b>
      </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="status-banner wait">
      <div class="dot amber"></div>
      <div>
        <b>🟡 MODÈLE EN ATTENTE</b> &nbsp;·&nbsp;
        Aucun éclair n'a dépassé θ = {theta:.2f}
        &nbsp;·&nbsp; Fallback : règle 30 min à <b>{t_regle_demo.strftime('%H:%M:%S')}</b>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════
# CONTRÔLES
# ═════════════════════════════════════════════════════════════════
c1, c2, c3, c4 = st.columns([1, 1, 1, 4])
with c1:
    if st.button("⏸ PAUSE" if playing else "▶ PLAY",
                 key="play_btn", use_container_width=True, type="primary"):
        st.session_state["playing"] = not playing
        st.rerun()
with c2:
    if st.button("⏮ RESET", key="reset_btn", use_container_width=True):
        st.session_state["idx"]     = 0
        st.session_state["playing"] = False
        st.rerun()
with c3:
    if st.button("⏭ FIN", key="end_btn", use_container_width=True):
        st.session_state["idx"]     = n_strikes - 1
        st.session_state["playing"] = False
        st.rerun()
with c4:
    st.progress(
        (idx + 1) / n_strikes,
        text=f"⚡ Éclair {idx + 1} / {n_strikes}  ·  "
             f"{current['date'].strftime('%H:%M:%S UTC')}",
    )


# ═════════════════════════════════════════════════════════════════
# FLASH ECLAIR COURANT
# ═════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="lightning-flash">
  <span class="icon">⚡</span>
  <span class="text">NOUVEL ÉCLAIR DÉTECTÉ
    <span class="meta">
      {current['date'].strftime('%H:%M:%S UTC')} · {current['dist']:.1f} km de {airport}
      · amplitude {abs(current.get('amplitude', 0)):.0f} kA
    </span>
  </span>
</div>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════
# VÉRIFICATION DE LA PRÉDICTION PRÉCÉDENTE
# ═════════════════════════════════════════════════════════════════
if idx >= 1:
    prev          = alert_df.iloc[idx - 1]
    prev_active   = float(prev["confiance"]) > theta
    gap_real      = (current["date"] - prev["date"]).total_seconds() / 60
    t_pred        = prev["horizon_min"]
    dist_cur      = float(current["dist"])
    in_window     = gap_real <= t_pred
    danger        = dist_cur < DIST_3KM

    if not prev_active:
        cls = "idle"; badge = "⊘ INACTIF"
        text = (f"Conf. précédente {prev['confiance']:.3f} ≤ θ = {theta:.2f}. "
                f"Aucune prédiction appliquée — règle 30 min en vigueur.")
    elif not in_window:
        cls = "ok"; badge = "✓ VALIDÉE"
        text = (f"Silence prédit ≥ {t_pred:.1f} min respecté : "
                f"prochain éclair après {gap_real:.1f} min "
                f"(écart {gap_real-t_pred:+.1f} min).")
    elif in_window and not danger:
        cls = "ok"; badge = "✓ SAFE"
        text = (f"Éclair en {gap_real:.1f} min (&lt; {t_pred:.1f} prédits) "
                f"mais à {dist_cur:.1f} km — hors zone danger (&ge; 3 km). "
                f"Sécurité piste maintenue.")
    else:
        cls = "fail"; badge = "✗ VIOLÉE"
        text = (f"Éclair en {gap_real:.1f} min (&lt; {t_pred:.1f} prédits) "
                f"ET à {dist_cur:.1f} km &lt; 3 km. Vrai échec.")

    st.markdown(
        f"<div class='verif-card {cls}'>"
        f"<span class='badge'>{badge}</span>"
        f"<span>{text}</span></div>",
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════════
# GRAPHE PLOTLY DARK
# ═════════════════════════════════════════════════════════════════
fig = make_subplots(
    rows=3, cols=1, row_heights=[0.42, 0.30, 0.28],
    shared_xaxes=True, vertical_spacing=0.07,
    subplot_titles=[
        "Distance des éclairs et fin d'alerte prédite",
        "⏳ T*_i prédit (barre) vs délai réel (◆)",
        f"Confiance S(30) — seuil θ = {theta:.2f}",
    ],
)

# Panneau 1 : distance
future = alert_df.iloc[idx + 1:]
if len(future):
    fig.add_trace(go.Scatter(
        x=future["date"], y=future["dist"], mode="markers",
        name="À venir", marker=dict(color="#475569", size=6, opacity=0.5),
        hoverinfo="skip",
    ), row=1, col=1)

seen_low = seen[~seen["model_active"]]
if len(seen_low):
    fig.add_trace(go.Scatter(
        x=seen_low["date"], y=seen_low["dist"], mode="markers",
        name="Éclair (conf ≤ θ)",
        marker=dict(color="#94A3B8", size=10, line=dict(width=1, color="#CBD5E1")),
        hovertemplate="%{x|%H:%M:%S} — %{y:.1f} km<extra></extra>",
    ), row=1, col=1)

seen_hi = seen[seen["model_active"]]
if len(seen_hi):
    fig.add_trace(go.Scatter(
        x=seen_hi["date"], y=seen_hi["dist"], mode="markers",
        name="Éclair confiant ✓",
        marker=dict(color="#10B981", size=13,
                    line=dict(width=1.5, color="#065F46")),
        hovertemplate=("%{x|%H:%M:%S} — %{y:.1f} km<br>"
                       "S(30) = %{customdata:.3f}<extra></extra>"),
        customdata=seen_hi["confiance"],
    ), row=1, col=1)

# Halo + étoile éclair courant
for hs, ho in [(70, 0.18), (48, 0.28), (30, 0.45)]:
    fig.add_trace(go.Scatter(
        x=[current["date"]], y=[current["dist"]],
        mode="markers", showlegend=False,
        marker=dict(color="#FBBF24", size=hs, opacity=ho),
        hoverinfo="skip",
    ), row=1, col=1)

y_top = max(alert_df["dist"].max() * 1.15, 15)
fig.add_trace(go.Scatter(
    x=[current["date"], current["date"]],
    y=[y_top, current["dist"]], mode="lines", showlegend=False,
    line=dict(color="#FBBF24", width=4), opacity=0.8, hoverinfo="skip",
), row=1, col=1)
fig.add_trace(go.Scatter(
    x=[current["date"]], y=[current["dist"]],
    mode="markers+text", name="⚡ Éclair courant",
    marker=dict(color="#DC2626", size=28, symbol="star",
                line=dict(width=2.5, color="#FBBF24")),
    text=["⚡"], textposition="top center",
    textfont=dict(size=24, color="#FBBF24"),
    hovertemplate="<b>ÉCLAIR ACTUEL</b><br>%{x|%H:%M:%S} — %{y:.1f} km<extra></extra>",
), row=1, col=1)

if gain_min > 0:
    fig.add_vrect(
        x0=t_modele.timestamp() * 1000, x1=t_regle_demo.timestamp() * 1000,
        fillcolor="#10B981", opacity=0.15, line_width=0, row=1, col=1,
    )
    fig.add_vline(
        x=t_modele.timestamp() * 1000, line_color="#10B981", line_width=3,
        annotation_text=f"<b>LEVÉE {t_modele.strftime('%H:%M')}</b>",
        annotation_font_color="#10B981", annotation_position="top left",
        row=1, col=1,
    )
fig.add_vline(
    x=t_regle_demo.timestamp() * 1000,
    line_color="#F59E0B", line_dash="dash", line_width=2,
    annotation_text=f"Règle 30 min ({t_regle_demo.strftime('%H:%M')})",
    annotation_font_color="#F59E0B", annotation_position="top right",
    row=1, col=1,
)
fig.add_hline(y=DIST_3KM, line_dash="dot", line_color="#EF4444", line_width=2,
              annotation_text="3 km", annotation_position="right",
              annotation_font_color="#EF4444", row=1, col=1)

# Panneau 2 : T*_i vs gap réel
seen_no_last = seen.iloc[:-1] if len(seen) >= 2 else seen.iloc[0:0]
inact = seen_no_last[~seen_no_last["model_active"]]
if len(inact):
    fig.add_trace(go.Bar(
        x=inact["date"], y=inact["horizon_min"],
        marker_color="#475569", opacity=0.55, name="Inactif",
        hovertemplate=("<b>%{x|%H:%M:%S}</b><br>Conf ≤ θ<br>"
                       "T*_i (ignoré) : %{y:.1f} min<extra></extra>"),
    ), row=2, col=1)

valid = seen[seen["pred_validable"]]
if len(valid):
    bar_colors = ["#10B981" if ok else "#EF4444" for ok in valid["pred_ok"]]
    fig.add_trace(go.Bar(
        x=valid["date"], y=valid["horizon_min"],
        marker_color=bar_colors, opacity=0.8,
        name="T*_i actif", marker_line=dict(width=1, color="#0F172A"),
        hovertemplate=("<b>%{x|%H:%M:%S}</b><br>Modèle actif<br>"
                       "T*_i : %{y:.1f} min<extra></extra>"),
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=valid["date"], y=valid["gap_next_min"],
        mode="markers", name="Délai réel ◆",
        marker=dict(color="#3B82F6", size=12, symbol="diamond",
                    line=dict(width=2, color="white")),
        hovertemplate=("<b>%{x|%H:%M:%S}</b><br>"
                       "Prochain éclair après : %{y:.1f} min<extra></extra>"),
    ), row=2, col=1)
    for _, r in valid.iterrows():
        cseg = "#10B981" if r["pred_ok"] else "#EF4444"
        fig.add_trace(go.Scatter(
            x=[r["date"], r["date"]],
            y=[r["horizon_min"], r["gap_next_min"]],
            mode="lines", showlegend=False,
            line=dict(color=cseg, width=2, dash="dot"),
            hoverinfo="skip",
        ), row=2, col=1)

cur_active = float(current["confiance"]) > theta
cur_color = "#FBBF24" if cur_active else "#475569"
fig.add_trace(go.Bar(
    x=[current["date"]], y=[current["horizon_min"]],
    marker_color=cur_color, opacity=0.85,
    name=("T*_i en attente" if cur_active else "Inactif (règle 30)"),
    marker_line=dict(width=2, color="#92400E" if cur_active else "#94A3B8"),
    hovertemplate=(
        f"<b>EN ATTENTE</b><br>"
        f"Modèle {'ACTIF' if cur_active else 'INACTIF'}<br>"
        f"conf = {current['confiance']:.3f} "
        f"{'>' if cur_active else '≤'} θ = {theta:.2f}<br>"
        f"T*_i = %{{y:.1f}} min<extra></extra>"
    ),
), row=2, col=1)

# Panneau 3 : confiance
fig.add_trace(go.Scatter(
    x=seen["date"], y=seen["confiance"],
    mode="lines+markers", name="S(30)",
    line=dict(color="#3B82F6", width=3),
    marker=dict(size=8, color="#3B82F6", line=dict(width=1, color="#1E40AF")),
    hovertemplate="%{x|%H:%M:%S} — S(30) = %{y:.3f}<extra></extra>",
), row=3, col=1)
for hs, ho in [(40, 0.22), (28, 0.4)]:
    fig.add_trace(go.Scatter(
        x=[current["date"]], y=[current["confiance"]],
        mode="markers", showlegend=False,
        marker=dict(color="#FBBF24", size=hs, opacity=ho),
        hoverinfo="skip",
    ), row=3, col=1)
fig.add_trace(go.Scatter(
    x=[current["date"]], y=[current["confiance"]],
    mode="markers", showlegend=False,
    marker=dict(color="#DC2626", size=18, symbol="star",
                line=dict(width=2, color="#FBBF24")),
    hoverinfo="skip",
), row=3, col=1)
fig.add_hline(y=theta, line_dash="dot", line_color="#10B981", line_width=2,
              annotation_text=f"θ = {theta:.2f}",
              annotation_font_color="#10B981",
              annotation_position="right", row=3, col=1)

# X range
x_min = alert_df["date"].min() - pd.Timedelta(minutes=2)
x_max = t_regle_demo + pd.Timedelta(minutes=3)
for r in (1, 2, 3):
    fig.update_xaxes(range=[x_min, x_max], row=r, col=1,
                     gridcolor="rgba(148,163,184,0.15)", color="#CBD5E1")
fig.update_xaxes(title_text="Heure (UTC)", row=3, col=1, color="#CBD5E1")
fig.update_yaxes(title_text="Distance (km)", row=1, col=1, rangemode="tozero",
                 gridcolor="rgba(148,163,184,0.15)", color="#CBD5E1")
fig.update_yaxes(title_text="Minutes", row=2, col=1, rangemode="tozero",
                 gridcolor="rgba(148,163,184,0.15)", color="#CBD5E1")
fig.update_yaxes(title_text="S(30)", row=3, col=1, range=[0, 1.05],
                 gridcolor="rgba(148,163,184,0.15)", color="#CBD5E1")

fig.update_layout(
    height=720,
    paper_bgcolor="rgba(15,23,42,0)", plot_bgcolor="rgba(15,23,42,0.4)",
    font=dict(color="#E2E8F0"),
    margin=dict(t=80, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.06, x=0,
                font=dict(color="#CBD5E1"), bgcolor="rgba(0,0,0,0)"),
    hovermode="x unified",
    bargap=0.15,
)
# Sous-titres en clair
for i, ann in enumerate(fig.layout.annotations[:3]):
    ann.font.color = "#F8FAFC"
    ann.font.size  = 14

st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════════
# TABLEAU PAR ALERTE
# ═════════════════════════════════════════════════════════════════
st.markdown("### 📋  Résumé de l'alerte en cours")
above_all  = alert_df[alert_df["confiance"] > theta]
n_actif    = len(above_all)
n_3km      = int((alert_df["dist"] < DIST_3KM).sum())
amp_max    = alert_df.get("amplitude", pd.Series([0])).abs().max()
dur_min    = (alert_df["date"].max() - alert_df["date"].min()).total_seconds() / 60

resume = pd.DataFrame({
    "Métrique": [
        "Aéroport", "Saison", "Alerte ID",
        "Éclairs total", "Éclairs < 3 km", "Durée orage",
        "Amplitude max", "Éclairs confiants (> θ)",
        "Règle Météorage", "Modèle GBS v7", "Gain temps",
    ],
    "Valeur": [
        f"✈️ {airport} ({info['icao']})",
        SAISON_LABELS.get(saison_id, "—"),
        f"#{alert_id}",
        f"{n_strikes} ⚡",
        f"{n_3km}" + (" ⚠️" if n_3km > 0 else ""),
        f"{dur_min:.0f} min",
        f"{amp_max:.0f} kA",
        f"{n_actif} / {n_strikes}",
        f"{t_regle_demo.strftime('%H:%M:%S')}",
        f"{t_modele.strftime('%H:%M:%S')}" if is_active else "— (inactif)",
        f"**{gain_min:.0f} min**" if is_active else "0 min",
    ],
})
st.dataframe(resume, hide_index=True, use_container_width=True, height=420)


# ═════════════════════════════════════════════════════════════════
# AUTO-ADVANCE
# ═════════════════════════════════════════════════════════════════
if playing:
    if idx < n_strikes - 1:
        time.sleep(delay_s)
        st.session_state["idx"] = idx + 1
        st.rerun()
    else:
        st.session_state["playing"] = False
        if gain_min > 0:
            st.success(
                f"✅ Simulation terminée — alerte levée à "
                f"**{t_modele.strftime('%H:%M:%S')}** "
                f"(gain final : **{gain_min:.0f} min**)"
            )
        else:
            st.info(f"ℹ️ Aucun éclair n'a dépassé θ = {theta:.2f}. Règle 30 min appliquée.")


# ═════════════════════════════════════════════════════════════════
# FOOTER
# ═════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="footer-strip">
  GBS v7 · θ = {theta:.2f} {'(défaut)' if abs(theta-THETA_DEFAULT)<1e-6 else '(modifié)'}
  · Données jury 2023-2025 · Battle Météorage 2026
</div>
""", unsafe_allow_html=True)
