"""Démo Data Battle Météorage 2026 — Optimisation des alertes foudre par GBS."""
import json
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import streamlit as st
from pathlib import Path

# ─────────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent
DATA_PATH   = ROOT / "data" / "data_df_with_alert.parquet"
MODELS_DIR  = ROOT / "models"
RESULTS_DIR = ROOT / "results"
MAX_GAP     = 30
DIST_3KM    = 3.0

SAISON_LABELS = {
    1: "❄️ Hiver",
    2: "🌸 Printemps",
    3: "☀️ Été",
    4: "🍂 Automne",
}

# Noms affichés dans l'interface
MODEL_DISPLAY = {
    "GBS v6": "GBS v6 — 12 variables",
    "GBS v7": "GBS v7 — 13 variables (+ aéroport)",
}

FEATURES_V6 = [
    "h_cos", "h_sin", "doy_cos", "doy_sin", "saison",
    "dist_centre", "dist_avg_5", "dist_min_so_far",
    "silence_min", "freq_5min", "rang", "rang_norm",
]
FEATURES_V7 = FEATURES_V6 + ["airport_enc"]

MODEL_CONFIGS = {
    "GBS v6": dict(model=MODELS_DIR/"gbs_v6_model.pkl", scaler=MODELS_DIR/"gbs_v6_scaler.pkl",
                   meta=MODELS_DIR/"gbs_v6_metadata.json", features=FEATURES_V6, le=None),
    "GBS v7": dict(model=MODELS_DIR/"gbs_v7_model.pkl", scaler=MODELS_DIR/"gbs_v7_scaler.pkl",
                   meta=MODELS_DIR/"gbs_v7_metadata.json", features=FEATURES_V7,
                   le=MODELS_DIR/"gbs_v7_label_encoder.pkl"),
}
MODEL_CONFIGS = {k: v for k, v in MODEL_CONFIGS.items() if v["model"].exists()}

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Météorage — Optimisation Alertes Foudre",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS léger pour les métriques
st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 1.4rem; font-weight: 700; }
[data-testid="stMetricLabel"] { font-size: 0.8rem; color: #6B7280; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# CHARGEMENT (cache)
# ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Chargement du modèle...")
def load_model(version: str):
    cfg = MODEL_CONFIGS[version]
    gbs    = joblib.load(cfg["model"])
    scaler = joblib.load(cfg["scaler"])
    with open(cfg["meta"]) as f:
        meta = json.load(f)
    le_path = cfg["le"]
    le = joblib.load(le_path) if le_path and Path(le_path).exists() else None
    return gbs, scaler, meta, le


@st.cache_data(show_spinner="Chargement des données...")
def load_data():
    df = pd.read_parquet(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df[df["date"].dt.year >= 2021].copy()
    df = df.sort_values(["airport", "airport_alert_id", "date"]).reset_index(drop=True)

    g = df.groupby(["airport", "airport_alert_id"])
    h   = df["date"].dt.hour + df["date"].dt.minute / 60
    doy = df["date"].dt.dayofyear
    df["h_cos"]           = np.cos(2 * np.pi * h / 24)
    df["h_sin"]           = np.sin(2 * np.pi * h / 24)
    df["doy_cos"]         = np.cos(2 * np.pi * doy / 365)
    df["doy_sin"]         = np.sin(2 * np.pi * doy / 365)
    df["saison"]          = ((df["date"].dt.month % 12) // 3) + 1
    prev = g["date"].shift(1)
    df["silence_min"]     = ((df["date"] - prev).dt.total_seconds() / 60).fillna(30).clip(0, 60)
    df["freq_5min"]       = (1 / df["silence_min"].clip(lower=0.5)).clip(upper=10)
    df["dist_centre"]     = df["dist"]
    df["dist_avg_5"]      = g["dist"].transform(lambda x: x.rolling(5, min_periods=1).mean())
    df["dist_min_so_far"] = g["dist"].cummin()
    df["rang"]            = g.cumcount()
    df["rang_norm"]       = df["rang"] / g["rang"].transform("max").clip(lower=1)

    le_path = MODELS_DIR / "gbs_v7_label_encoder.pkl"
    if le_path.exists():
        le_enc = joblib.load(le_path)
        df["airport_enc"] = le_enc.transform(df["airport"])
    else:
        from sklearn.preprocessing import LabelEncoder
        df["airport_enc"] = LabelEncoder().fit_transform(df["airport"])

    for col in FEATURES_V7:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    df["alert_saison"] = g["saison"].transform("first")
    return df


@st.cache_data(show_spinner="Calcul des probabilités de survie (mise en cache)...")
def precompute_all_scores(_gbs, _scaler, df, features):
    X        = _scaler.transform(df[features].values.astype(float))
    surv_fns = _gbs.predict_survival_function(X)
    s30      = np.array([float(fn(30)) for fn in surv_fns])
    t_stars  = np.full(len(surv_fns), 30.0)
    for i, (fn, s) in enumerate(zip(surv_fns, s30)):
        if s <= 0:
            continue
        idx = np.searchsorted(-fn.y, -(s / 0.98), side="left")
        if idx < len(fn.x):
            t_stars[i] = min(float(fn.x[idx]), 30.0)
    result = df.copy()
    result["confiance"]     = s30
    result["horizon_min"]   = t_stars
    result["fin_predite"]   = result["date"] + pd.to_timedelta(t_stars, unit="m")
    return result


@st.cache_data
def load_jury_data():
    jury = {}
    for vk, jname, cname in [
        ("GBS v6", "jury_summary_eval_2023.json",    "predictions_eval_2023.csv"),
        ("GBS v7", "jury_summary_eval_2023_v7.json", "predictions_eval_2023_v7.csv"),
    ]:
        jpath = RESULTS_DIR / jname
        if jpath.exists():
            with open(jpath) as f:
                summary = json.load(f)
            cpath = RESULTS_DIR / cname
            jury[vk] = {"summary": summary,
                        "predictions": pd.read_csv(cpath) if cpath.exists() else None}
    return jury


@st.cache_data
def load_jury_strikes(version: str):
    paths = {"GBS v7": RESULTS_DIR / "strikes_eval_2023_v7.csv",
             "GBS v6": RESULTS_DIR / "strikes_eval_2023_v6.csv"}
    path = paths.get(version)
    if path is None or not path.exists():
        return None
    df = pd.read_csv(path)
    # format="mixed" gère les timestamps ISO8601 avec nanosecondes
    df["date"]        = pd.to_datetime(df["date"],  format="mixed", utc=True)
    df["fin_predite"] = pd.to_datetime(df["t_hat"], format="mixed", utc=True)
    if "score_conf" in df.columns:
        df = df.rename(columns={"score_conf": "confiance"})
    df["saison"]       = ((df["date"].dt.month % 12) // 3) + 1
    df["alert_saison"] = (df.groupby(["airport", "airport_alert_id"])["saison"]
                            .transform("first"))
    return df


# ─────────────────────────────────────────────────────────────────
# CALCUL MÉTRIQUES (noyau partagé)
# ─────────────────────────────────────────────────────────────────
def _compute_stats(df, theta, only_l3=False):
    """
    Protocole jury :
      - t_règle    = dernier éclair de l'alerte + 30 min
      - t_modèle   = min(fin_predite) pour éclairs avec confiance > θ
      - Gain       = Σ max(0, t_règle − t_modèle) / 3600
      - Taux risque = éclairs < 3 km après t_modèle / total éclairs < 3 km
    """
    if len(df) == 0:
        return pd.DataFrame(), {}

    g = df.groupby(["airport", "airport_alert_id"])
    t_regles = g["date"].max() + pd.Timedelta(minutes=MAX_GAP)

    above = df[df["confiance"] > theta]
    merged = t_regles.rename("t_regle").to_frame()
    if len(above) > 0:
        t_modeles = (above.groupby(["airport", "airport_alert_id"])["fin_predite"]
                     .min().rename("fin_predite"))
        merged = merged.join(t_modeles, how="left")
    else:
        merged["fin_predite"] = pd.Series(
            pd.NaT, index=merged.index, dtype=df["fin_predite"].dtype
        )
    merged["t_modele"]    = merged["fin_predite"].combine_first(merged["t_regle"])
    merged["gain_min"]    = ((merged["t_regle"] - merged["t_modele"])
                             .dt.total_seconds().clip(lower=0) / 60)
    merged["modele_actif"] = merged["fin_predite"].notna()

    # Éclairs manqués par zone de distance
    t_pred_map = merged[["t_modele"]].reset_index()

    def missed_zone(dist_max):
        z = df[df["dist"] < dist_max][["airport", "airport_alert_id", "date"]].copy()
        if len(z) == 0:
            return pd.DataFrame(columns=["airport", "airport_alert_id",
                                         f"n_{int(dist_max)}km", f"n_mq_{int(dist_max)}km"])
        z = z.merge(t_pred_map, on=["airport", "airport_alert_id"], how="left")
        z["manque"] = z["date"] >= z["t_modele"]
        return z.groupby(["airport", "airport_alert_id"]).agg(
            **{f"n_{int(dist_max)}km": ("date", "count"),
               f"n_mq_{int(dist_max)}km": ("manque", "sum")}
        ).reset_index()

    l3_stats  = missed_zone(3)
    l10_stats = missed_zone(10)
    l20_stats = missed_zone(20)
    # renommage pour garder les noms lisibles
    l3_stats  = l3_stats.rename(columns={"n_3km": "n_danger",   "n_mq_3km":  "n_manques"})
    l10_stats = l10_stats.rename(columns={"n_10km": "n_10km",   "n_mq_10km": "n_mq_10km"})
    l20_stats = l20_stats.rename(columns={"n_20km": "n_20km",   "n_mq_20km": "n_mq_20km"})

    info = g.agg(n_eclairs=("date", "count"), dist_min=("dist", "min")).reset_index()
    result = (info
              .merge(merged[["t_regle", "t_modele", "gain_min", "modele_actif"]].reset_index(),
                     on=["airport", "airport_alert_id"], how="left")
              .merge(l3_stats,  on=["airport", "airport_alert_id"], how="left")
              .merge(l10_stats, on=["airport", "airport_alert_id"], how="left")
              .merge(l20_stats, on=["airport", "airport_alert_id"], how="left"))

    for col in ["n_danger", "n_manques", "n_10km", "n_mq_10km", "n_20km", "n_mq_20km"]:
        result[col] = result[col].fillna(0).astype(int)
    result["est_dangereux"] = result["n_danger"] > 0

    if only_l3:
        result = result[result["est_dangereux"]].copy()
    if len(result) == 0:
        return pd.DataFrame(), {}

    tot_3   = int(result["n_danger"].sum())
    mq_3    = int(result["n_manques"].sum())
    tot_10  = int(result["n_10km"].sum())
    mq_10   = int(result["n_mq_10km"].sum())
    tot_20  = int(result["n_20km"].sum())
    mq_20   = int(result["n_mq_20km"].sum())
    gain_h  = result["gain_min"].sum() / 60
    n_actif = int(result["modele_actif"].sum())

    agg = {
        "n_alertes":      len(result),
        "n_dangereuses":  int(result["est_dangereux"].sum()),
        # zone < 3 km
        "n_danger":       tot_3,
        "n_manques":      mq_3,
        "taux_risque":    mq_3  / tot_3  * 100 if tot_3  > 0 else 0.0,
        # zone < 10 km
        "n_10km":         tot_10,
        "n_mq_10km":      mq_10,
        "taux_10km":      mq_10 / tot_10 * 100 if tot_10 > 0 else 0.0,
        # zone < 20 km
        "n_20km":         tot_20,
        "n_mq_20km":      mq_20,
        "taux_20km":      mq_20 / tot_20 * 100 if tot_20 > 0 else 0.0,
        # gain
        "gain_h":         gain_h,
        "gain_moy_min":   float(result["gain_min"].mean()),
        "n_actif":        n_actif,
        "pct_actif":      n_actif / len(result) * 100,
    }
    return result, agg


def compute_test_stats(df_scored, theta, airport=None, saison=None, only_l3=False):
    df = df_scored.copy()
    if airport and airport != "Tous":
        df = df[df["airport"] == airport]
    if saison is not None:
        df = df[df["alert_saison"] == saison]
    return _compute_stats(df, theta, only_l3=only_l3)


def compute_jury_stats(df_strikes, theta, airport=None, saison=None, only_l3=False):
    df = df_strikes.copy()
    if airport and airport != "Tous":
        df = df[df["airport"] == airport]
    if saison is not None:
        df = df[df["alert_saison"] == saison]
    return _compute_stats(df, theta, only_l3=only_l3)


# ─────────────────────────────────────────────────────────────────
# GRAPHES — PERFORMANCE GLOBALE
# ─────────────────────────────────────────────────────────────────
def plot_gain_par_aeroport(df_alerts):
    by_ap = (df_alerts.groupby("airport")
             .agg(gain_h=("gain_min", lambda x: x.sum() / 60),
                  n=("airport_alert_id", "count"))
             .reset_index().sort_values("gain_h", ascending=True))
    colors = ["#10B981" if g > 0 else "#E5E7EB" for g in by_ap["gain_h"]]
    fig = go.Figure(go.Bar(
        y=by_ap["airport"], x=by_ap["gain_h"], orientation="h",
        marker_color=colors,
        text=[f"  {v:.1f} h ({n} alertes)" for v, n in zip(by_ap["gain_h"], by_ap["n"])],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Temps économisé : %{x:.2f} h<extra></extra>",
    ))
    fig.update_layout(
        title="<b>Temps économisé par aéroport</b>",
        xaxis_title="Gain (heures)", yaxis_title="",
        height=max(280, len(by_ap) * 44),
        template="plotly_white", margin=dict(l=140, r=100, t=50),
    )
    return fig


def plot_distribution_gains(df_alerts):
    vals = df_alerts["gain_min"].clip(lower=0)
    moy  = vals.mean()
    fig  = go.Figure(go.Histogram(
        x=vals, nbinsx=35,
        marker_color="#3B82F6", opacity=0.75,
        hovertemplate="Gain : %{x:.0f} min — %{y} alertes<extra></extra>",
    ))
    fig.add_vline(x=moy, line_dash="dash", line_color="#F59E0B",
                  annotation_text=f"Moy. : {moy:.0f} min",
                  annotation_font_color="#F59E0B", annotation_position="top right")
    fig.update_layout(
        title="<b>Distribution du gain par alerte</b>",
        xaxis_title="Gain (minutes)", yaxis_title="Nombre d'alertes",
        height=310, template="plotly_white",
    )
    return fig


def plot_scatter_alertes(df_alerts):
    sures    = df_alerts[df_alerts["n_manques"] == 0]
    risquees = df_alerts[df_alerts["n_manques"] >  0]
    fig = go.Figure()
    if len(sures):
        fig.add_trace(go.Scatter(
            x=sures["n_danger"], y=sures["gain_min"] / 60,
            mode="markers", name="Alerte sans risque",
            marker=dict(color="#10B981", size=7, opacity=0.6),
            hovertemplate="<b>%{customdata[0]} #%{customdata[1]}</b><br>"
                          "Éclairs < 3 km : %{x} | Gain : %{y:.2f} h<extra></extra>",
            customdata=list(zip(sures["airport"], sures["airport_alert_id"])),
        ))
    if len(risquees):
        fig.add_trace(go.Scatter(
            x=risquees["n_danger"], y=risquees["gain_min"] / 60,
            mode="markers", name="Alerte risquée ⚠️",
            marker=dict(color="#EF4444", size=11, symbol="diamond", opacity=0.9),
            hovertemplate="<b>%{customdata[0]} #%{customdata[1]}</b><br>"
                          "Éclairs < 3 km : %{x} | Gain : %{y:.2f} h | "
                          "Manqués : %{customdata[2]}<extra></extra>",
            customdata=list(zip(risquees["airport"],
                                risquees["airport_alert_id"],
                                risquees["n_manques"])),
        ))
    fig.update_layout(
        title="<b>Gain vs dangerosité de l'alerte</b>",
        xaxis_title="Éclairs < 3 km", yaxis_title="Gain (heures)",
        height=340, template="plotly_white",
        legend=dict(orientation="h", y=1.12),
    )
    return fig


# ─────────────────────────────────────────────────────────────────
# GRAPHE — ANALYSE D'UN ORAGE (visuel principal démo)
# ─────────────────────────────────────────────────────────────────
def plot_analyse_orage(alert_df, t_modele, t_regle, gain_min, theta):
    """
    Graphe triple panneau :
      Haut    : timeline distance des éclairs, coloré PRÉDICTION OK / KO
      Milieu  : barres = horizon prédit T*_i par éclair (vert/rouge selon validation)
      Bas     : évolution du score de confiance S(30) avec seuil θ
    """
    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.42, 0.30, 0.28],
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[
            "Distance des éclairs à l'aéroport — coloré selon la prédiction du modèle",
            "Prédiction par éclair : T*_i (min) — vert ✓ si prochain éclair arrive après, rouge ✗ sinon",
            "Score de confiance S(30) — probabilité que l'orage se calme",
        ],
    )

    # ── Panneau 1 : distances colorées par validation prédiction ───
    df_ok  = alert_df[alert_df["prediction_ok"]]
    df_ko  = alert_df[~alert_df["prediction_ok"]]
    df_pres = alert_df[alert_df["dist"] < DIST_3KM]

    if len(df_ok):
        fig.add_trace(go.Scatter(
            x=df_ok["date"], y=df_ok["dist"],
            mode="markers",
            name="Prédiction validée ✓ (aucun éclair avant T*_i)",
            marker=dict(color="#10B981", size=10, symbol="circle",
                        line=dict(width=1.2, color="#065F46")),
            customdata=list(zip(df_ok["confiance"],
                                df_ok["horizon_min"],
                                df_ok["gap_next_min"].fillna(-1))),
            hovertemplate=(
                "<b>%{x|%H:%M:%S UTC}</b><br>"
                "Distance : %{y:.1f} km<br>"
                "Confiance S(30) : %{customdata[0]:.3f}<br>"
                "T*_i prédit : %{customdata[1]:.1f} min<br>"
                "Prochain éclair : %{customdata[2]:.1f} min (ou aucun)"
                "<extra>✓ Prédiction validée</extra>"
            ),
        ), row=1, col=1)

    if len(df_ko):
        fig.add_trace(go.Scatter(
            x=df_ko["date"], y=df_ko["dist"],
            mode="markers",
            name="Prédiction violée ✗ (éclair arrivé avant T*_i)",
            marker=dict(color="#EF4444", size=11, symbol="x-thin",
                        line=dict(width=3, color="#991B1B")),
            customdata=list(zip(df_ko["confiance"],
                                df_ko["horizon_min"],
                                df_ko["gap_next_min"].fillna(-1))),
            hovertemplate=(
                "<b>%{x|%H:%M:%S UTC}</b><br>"
                "Distance : %{y:.1f} km<br>"
                "Confiance S(30) : %{customdata[0]:.3f}<br>"
                "T*_i prédit : %{customdata[1]:.1f} min<br>"
                "Prochain éclair après : %{customdata[2]:.1f} min ← AVANT T*"
                "<extra>✗ Prédiction violée</extra>"
            ),
        ), row=1, col=1)

    if len(df_pres):
        fig.add_trace(go.Scatter(
            x=df_pres["date"], y=df_pres["dist"],
            mode="markers",
            name="<b>Éclair en zone danger (< 3 km) ⚠️</b>",
            marker=dict(color="#F59E0B", size=18, symbol="star",
                        line=dict(width=1.5, color="#92400E")),
            hovertemplate=(
                "<b>%{x|%H:%M:%S UTC}</b><br>"
                "Distance : %{y:.2f} km — ZONE DANGER<extra></extra>"
            ),
        ), row=1, col=1)

    # ── Panneau 2 : barres T*_i + segments gap réel ────────────────
    bar_colors = ["#10B981" if ok else "#EF4444"
                  for ok in alert_df["prediction_ok"]]
    fig.add_trace(go.Bar(
        x=alert_df["date"], y=alert_df["horizon_min"],
        marker_color=bar_colors, opacity=0.75,
        name="T*_i prédit (min)",
        showlegend=False,
        hovertemplate=(
            "<b>%{x|%H:%M:%S UTC}</b><br>"
            "T*_i prédit : %{y:.1f} min<extra></extra>"
        ),
    ), row=2, col=1)
    # Points : gap réel
    gap_pts = alert_df[alert_df["gap_next_min"].notna()]
    if len(gap_pts):
        fig.add_trace(go.Scatter(
            x=gap_pts["date"], y=gap_pts["gap_next_min"],
            mode="markers",
            name="Délai réel jusqu'au prochain éclair (min)",
            marker=dict(color="#1E40AF", size=7, symbol="diamond",
                        line=dict(width=1, color="white")),
            hovertemplate=(
                "<b>%{x|%H:%M:%S UTC}</b><br>"
                "Prochain éclair arrivé après : %{y:.1f} min<extra></extra>"
            ),
        ), row=2, col=1)

    # Zone de gain
    if gain_min > 0:
        fig.add_vrect(
            x0=t_modele.timestamp() * 1000,
            x1=t_regle.timestamp() * 1000,
            fillcolor="#10B981", opacity=0.10, line_width=0,
            row=1, col=1,
        )

    # Lignes de fin d'alerte
    fig.add_vline(
        x=t_modele.timestamp() * 1000, line_dash="solid",
        line_color="#10B981", line_width=2.5,
        annotation_text=f"<b>Fin prédite : {t_modele.strftime('%H:%M')}</b>",
        annotation_font_color="#10B981", annotation_font_size=13,
        annotation_position="top left",
        row=1, col=1,
    )
    fig.add_vline(
        x=t_regle.timestamp() * 1000, line_dash="dash",
        line_color="#F59E0B", line_width=2,
        annotation_text=f"Règle +30 min : {t_regle.strftime('%H:%M')}",
        annotation_font_color="#F59E0B", annotation_font_size=12,
        annotation_position="top right",
        row=1, col=1,
    )
    fig.add_hline(y=DIST_3KM, line_dash="dot", line_color="#EF4444", line_width=1.5,
                  annotation_text="Seuil 3 km", annotation_position="right",
                  annotation_font_color="#EF4444",
                  row=1, col=1)

    # ── Panneau 3 : confiance ──────────────────────────────────────
    above_mask = alert_df["confiance"] > theta
    below_mask = ~above_mask

    if above_mask.any():
        fig.add_trace(go.Scatter(
            x=alert_df[above_mask]["date"],
            y=alert_df[above_mask]["confiance"],
            mode="lines+markers",
            name=f"Confiance > θ ({theta:.2f}) → prédiction active",
            line=dict(color="#10B981", width=2),
            marker=dict(size=6),
            hovertemplate="%{x|%H:%M:%S} — S(30) = %{y:.3f}<extra></extra>",
        ), row=3, col=1)

    if below_mask.any():
        fig.add_trace(go.Scatter(
            x=alert_df[below_mask]["date"],
            y=alert_df[below_mask]["confiance"],
            mode="markers",
            name=f"Confiance ≤ θ ({theta:.2f}) → ignoré",
            marker=dict(color="#D1D5DB", size=5),
            hovertemplate="%{x|%H:%M:%S} — S(30) = %{y:.3f}<extra></extra>",
        ), row=3, col=1)

    fig.add_hline(y=theta, line_dash="dot", line_color="#10B981", line_width=1.5,
                  annotation_text=f"Seuil θ = {theta:.2f}",
                  annotation_position="right",
                  annotation_font_color="#10B981",
                  row=3, col=1)

    # ── Lignes verticales communes : fin prédite + règle 30 min ────
    for r in (1, 2, 3):
        fig.add_vline(
            x=t_modele.timestamp() * 1000, line_dash="solid",
            line_color="#10B981", line_width=2.5, row=r, col=1,
        )
        fig.add_vline(
            x=t_regle.timestamp() * 1000, line_dash="dash",
            line_color="#F59E0B", line_width=2, row=r, col=1,
        )

    # Annotations sur le panneau 1 uniquement
    fig.add_annotation(
        x=t_modele, y=1, xref="x", yref="paper",
        text=f"<b>Fin prédite : {t_modele.strftime('%H:%M')}</b>",
        showarrow=False, font=dict(color="#10B981", size=12),
        xanchor="right", yanchor="bottom",
    )
    fig.add_annotation(
        x=t_regle, y=1, xref="x", yref="paper",
        text=f"Règle +30 min : {t_regle.strftime('%H:%M')}",
        showarrow=False, font=dict(color="#F59E0B", size=11),
        xanchor="left", yanchor="bottom",
    )

    # Zone de gain (panneau 1)
    if gain_min > 0:
        fig.add_vrect(
            x0=t_modele.timestamp() * 1000,
            x1=t_regle.timestamp() * 1000,
            fillcolor="#10B981", opacity=0.08, line_width=0, row=1, col=1,
        )

    # Ligne 3 km
    fig.add_hline(y=DIST_3KM, line_dash="dot", line_color="#EF4444",
                  line_width=1.3, row=1, col=1,
                  annotation_text="Seuil 3 km", annotation_position="right",
                  annotation_font_color="#EF4444")

    # ── Mise en forme axes ─────────────────────────────────────────
    fig.update_yaxes(title_text="Distance (km)", row=1, col=1, rangemode="tozero")
    fig.update_yaxes(title_text="T*_i (min)",    row=2, col=1, rangemode="tozero")
    fig.update_yaxes(title_text="S(30)",         row=3, col=1, range=[0, 1.05])
    fig.update_xaxes(title_text="Heure (UTC)",   row=3, col=1)

    n_ok = int(alert_df["prediction_ok"].sum())
    n_tot = len(alert_df)
    n_ko = n_tot - n_ok
    gain_txt = (f"<b>Gain : {gain_min:.0f} min économisées</b>"
                if gain_min > 0 else "Pas de gain (θ trop élevé)")
    val_txt = (f"Prédictions validées : <b>{n_ok}/{n_tot}</b> "
               f"({n_ok/n_tot*100:.0f} %) — violées : {n_ko}")
    fig.update_layout(
        title=dict(
            text=f"Analyse de l'orage — {gain_txt}<br>"
                 f"<span style='font-size:12px;color:#374151'>{val_txt}</span>",
            font=dict(size=15),
        ),
        height=760, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.05, x=0),
        hovermode="x unified",
        margin=dict(t=110),
        bargap=0.05,
    )
    return fig


def plot_courbe_survie(surv_fn, t_star, confiance, date_eclair, dist_eclair):
    t_grid = np.linspace(0, 30, 300)
    s_vals = [float(surv_fn(t)) for t in t_grid]
    s30    = float(surv_fn(30))

    fig = go.Figure()
    # Zone sous la courbe jusqu'à T*
    t_fill = t_grid[t_grid <= t_star]
    s_fill = [float(surv_fn(t)) for t in t_fill]
    fig.add_trace(go.Scatter(
        x=np.concatenate([t_fill, t_fill[::-1]]),
        y=np.concatenate([s_fill, np.zeros(len(s_fill))]),
        fill="toself", fillcolor="rgba(16,185,129,0.12)",
        line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=t_grid, y=s_vals, mode="lines", name="S(t|x)",
        line=dict(color="#3B82F6", width=3),
        hovertemplate="t = %{x:.1f} min → S(t) = %{y:.3f}<extra></extra>",
    ))
    fig.add_vline(x=t_star, line_dash="dash", line_color="#10B981", line_width=2,
                  annotation_text=f"<b>T* = {t_star:.0f} min</b>",
                  annotation_font_color="#10B981", annotation_position="top right")
    fig.add_hline(y=s30, line_dash="dot", line_color="#F59E0B", line_width=1,
                  annotation_text=f"S(30) = {s30:.2f}",
                  annotation_position="right", annotation_font_color="#F59E0B")
    fig.update_layout(
        title=(
            f"Fonction de survie — éclair à {date_eclair.strftime('%H:%M:%S UTC')} "
            f"({dist_eclair:.1f} km)"
        ),
        xaxis_title="Temps depuis cet éclair (minutes)",
        yaxis_title="Probabilité qu'aucun nouvel éclair ne survienne",
        yaxis=dict(range=[0, 1.05], tickformat=".0%"),
        height=370, template="plotly_white",
    )
    return fig


def plot_jury_comparaison(jury):
    models = list(jury.keys())
    gains  = [jury[m]["summary"]["Gain_h"]   for m in models]
    risks  = [jury[m]["summary"]["Risk_pct"] for m in models]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Temps économisé (heures)", "Taux de risque (%)"],
    )
    colors = ["#3B82F6", "#10B981"]
    fig.add_trace(go.Bar(x=models, y=gains, marker_color=colors,
                         text=[f"{v:.2f} h" for v in gains], textposition="outside",
                         showlegend=False), row=1, col=1)
    fig.add_trace(go.Bar(x=models, y=risks,
                         marker_color=["#EF4444" if r > 0 else "#6EE7B7" for r in risks],
                         text=[f"{v:.4f}%" for v in risks], textposition="outside",
                         showlegend=False), row=1, col=2)
    fig.add_hline(y=2.0, line_dash="dot", line_color="red", line_width=1.5,
                  annotation_text="Limite 2%", row=1, col=2)
    fig.update_layout(
        title="<b>Comparaison des modèles — Données jury 2023</b>",
        height=370, template="plotly_white",
    )
    return fig


# ─────────────────────────────────────────────────────────────────
# HELPER AFFICHAGE MÉTRIQUES
# ─────────────────────────────────────────────────────────────────
def afficher_metriques(stats_df, agg, c_idx, theta):
    if not agg:
        st.warning("Aucune alerte ne correspond aux filtres.")
        return

    # Ligne 1 — métriques clés
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Alertes analysées",     f"{agg['n_alertes']:,}")
    col2.metric("Alertes avec éclairs < 3 km", f"{agg['n_dangereuses']:,}")
    col3.metric("⏱ Temps économisé",    f"{agg['gain_h']:.2f} h",
                help="vs règle fixe +30 min après dernier éclair")
    col4.metric("Gain moyen / alerte",   f"{agg['gain_moy_min']:.1f} min")
    col5.metric("C-index (validation)",  f"{c_idx:.4f}",
                help="Pouvoir discriminant du modèle de survie")

    st.divider()

    # Ligne 2 — éclairs manqués par zone de proximité
    st.markdown("**Éclairs manqués après la fin prédite — par zone de danger**")
    z1, z2, z3, z4 = st.columns(4)
    z1.metric(
        "Zone critique < 3 km",
        f"{agg['n_manques']} / {agg['n_danger']}",
        delta="✓ Aucun" if agg["n_manques"] == 0 else f"⚠️ {agg['taux_risque']:.3f}%",
        delta_color="normal" if agg["n_manques"] == 0 else "inverse",
        help="Limite jury : < 2%",
    )
    z2.metric(
        "Zone alerte < 10 km",
        f"{agg['n_mq_10km']} / {agg['n_10km']}",
        delta=f"{agg['taux_10km']:.2f}%" if agg["n_mq_10km"] > 0 else "✓ Aucun",
        delta_color="inverse" if agg["n_mq_10km"] > 0 else "normal",
    )
    z3.metric(
        "Zone surveillance < 20 km",
        f"{agg['n_mq_20km']} / {agg['n_20km']}",
        delta=f"{agg['taux_20km']:.2f}%" if agg["n_mq_20km"] > 0 else "✓ Aucun",
        delta_color="inverse" if agg["n_mq_20km"] > 0 else "normal",
    )
    z4.metric(
        "Modèle actif",
        f"{agg['pct_actif']:.0f}% des alertes",
        help=f"{agg['n_actif']} alertes raccourcies",
    )

    st.divider()

    # Graphes
    gcol1, gcol2 = st.columns(2)
    with gcol1:
        st.plotly_chart(plot_gain_par_aeroport(stats_df), use_container_width=True)
    with gcol2:
        st.plotly_chart(plot_distribution_gains(stats_df), use_container_width=True)
    st.plotly_chart(plot_scatter_alertes(stats_df), use_container_width=True)

    # Tableau
    st.subheader("Résultats par alerte")
    cols_table = ["airport", "airport_alert_id", "n_eclairs",
                  "n_danger", "n_manques",
                  "n_10km",  "n_mq_10km",
                  "n_20km",  "n_mq_20km",
                  "gain_min", "modele_actif", "dist_min"]
    cols_table = [c for c in cols_table if c in stats_df.columns]
    df_show = stats_df[cols_table].copy()
    df_show["gain_min"]     = df_show["gain_min"].round(1)
    df_show["dist_min"]     = df_show["dist_min"].round(2)
    df_show["modele_actif"] = df_show["modele_actif"].map(
        {True: "✓ Modèle", False: "Règle +30 min"}
    )
    st.dataframe(
        df_show.rename(columns={
            "airport":          "Aéroport",
            "airport_alert_id": "ID Alerte",
            "n_eclairs":        "Éclairs",
            "n_danger":         "< 3 km",
            "n_manques":        "Mq. < 3 km",
            "n_10km":           "< 10 km",
            "n_mq_10km":        "Mq. < 10 km",
            "n_20km":           "< 20 km",
            "n_mq_20km":        "Mq. < 20 km",
            "gain_min":         "Gain (min)",
            "modele_actif":     "Décision",
            "dist_min":         "Dist. min (km)",
        }).sort_values("Gain (min)", ascending=False),
        use_container_width=True, hide_index=True, height=420,
    )

    # Synthèse par aéroport
    st.subheader("Synthèse par aéroport")
    by_ap = stats_df.groupby("airport").agg(
        Alertes     =("airport_alert_id", "count"),
        Dangereuses =("est_dangereux", "sum"),
        Gain_h      =("gain_min", lambda x: round(x.sum() / 60, 2)),
        L3_total    =("n_danger", "sum"),
        L3_manques  =("n_manques", "sum"),
    ).reset_index()
    by_ap["Taux risque (%)"] = (
        by_ap["L3_manques"] / by_ap["L3_total"].clip(lower=1) * 100
    ).round(3)
    st.dataframe(
        by_ap.rename(columns={"airport": "Aéroport"}),
        use_container_width=True, hide_index=True,
    )


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
df       = load_data()
jury_all = load_jury_data()

# ── SIDEBAR ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ Météorage")
    st.markdown("##### Optimisation des alertes foudre aéroportuaires")
    st.markdown("*Data Battle 2026*")
    st.divider()

    st.markdown("### Modèle")
    if not MODEL_CONFIGS:
        st.error("Aucun modèle trouvé.")
        st.stop()
    version = st.radio(
        "Choisir le modèle",
        list(MODEL_CONFIGS.keys()),
        format_func=lambda x: MODEL_DISPLAY.get(x, x),
        horizontal=False,
    )
    gbs, scaler, meta, _ = load_model(version)
    features = MODEL_CONFIGS[version]["features"]
    c_idx    = meta.get("c_index_test", meta.get("c_index_v6_test", 0))

    st.metric("C-index (validation 2021-22)", f"{c_idx:.4f}",
              help="1.0 = parfait, 0.5 = aléatoire")
    if version != "GBS v6":
        dv6 = meta.get("delta_test", meta.get("delta_vs_v6", 0))
        st.metric("Δ vs v6", f"{dv6:+.4f}", delta_color="normal" if dv6 > 0 else "inverse")
    st.caption(f"{len(features)} variables d'entrée")

    st.divider()
    st.markdown("### Seuil de confiance θ")
    theta = st.slider(
        "θ (S(30) minimum pour agir)",
        0.0, 1.0, 0.30, 0.01,
        help="Plus θ est élevé, plus le modèle est conservateur.\nθ = 0.30 est optimal sur les données jury.",
    )
    tref = jury_all.get(version, {}).get("summary", {})
    if tref:
        gain_ref = tref.get("Gain_h", 0)
        st.caption(f"Référence jury 2023 (θ=0.30) : **{gain_ref:.1f} h** économisées")

    st.divider()
    st.markdown("### Filtres")
    airports_list  = ["Tous"] + sorted(df["airport"].unique().tolist())
    airport_sel    = st.selectbox("Aéroport", airports_list)
    saison_opts    = ["Toutes"] + [SAISON_LABELS[k] for k in sorted(SAISON_LABELS)]
    saison_sel     = st.selectbox("Saison", saison_opts)
    saison_num     = {v: k for k, v in SAISON_LABELS.items()}.get(saison_sel, None)
    only_l3        = st.checkbox("Alertes avec éclairs < 3 km seulement", value=False)

    st.divider()
    st.caption("Entraînement : 2016–2020 | Validation : 2021–2022 | Jury : 2023–2025")
    with st.expander("ℹ️ À propos"):
        st.markdown(
            "**GBS Survival Analysis** — Gradient Boosting sur la perte de Cox "
            "implémenté via `scikit-survival`. \n\n"
            "Pour chaque éclair on prédit `S(t|xi)` et on en déduit l'horizon "
            "individuel `T*_i`. La fin d'alerte = `min(T*_i)` sur les éclairs "
            "confiants (S(30) > θ). \n\n"
            "**Résultat jury 2023-2025** : 103,5 h économisées, "
            "risque 0,10 % (limite : 2 %)."
        )


# ── HEADER ───────────────────────────────────────────────────────
st.title("⚡ Optimisation des alertes foudre aéroportuaires")
st.markdown(
    f"**{MODEL_DISPLAY.get(version, version)}** — "
    f"C-index `{c_idx:.4f}` — "
    f"Seuil θ = `{theta:.2f}` — "
    "Objectif : **réduire la durée des alertes** sans jamais manquer un éclair dangereux"
)
st.divider()

# ── PRÉ-CALCUL (cache par modèle) ────────────────────────────────
with st.spinner("Calcul en cours..."):
    df_scored = precompute_all_scores(gbs, scaler, df, features)

stats_test, agg_test = compute_test_stats(
    df_scored, theta, airport=airport_sel, saison=saison_num, only_l3=only_l3
)

# ── ONGLETS ───────────────────────────────────────────────────────
tab_perf, tab_jury, tab_orage, tab_demo = st.tabs([
    "📊 Performance globale",
    "🏆 Évaluation jury 2023",
    "🔍 Analyse d'un orage",
    "🎬 Démo en direct",
])


# ═════════════════════════════════════════════════════════════════
# TAB 1 — PERFORMANCE GLOBALE
# ═════════════════════════════════════════════════════════════════
with tab_perf:
    jeu = st.radio(
        "Jeu de données",
        ["Validation 2021-2022", "Jury 2023 (données indépendantes)"],
        horizontal=True,
    )
    st.divider()

    if jeu == "Validation 2021-2022":
        st.subheader(f"Résultats sur données de validation — {MODEL_DISPLAY.get(version, version)} — θ = {theta:.2f}")
        afficher_metriques(stats_test, agg_test, c_idx, theta)

    else:
        try:
            jury_strikes = load_jury_strikes(version)
            if jury_strikes is None:
                st.info(
                    f"Données strike-niveau non disponibles pour **{MODEL_DISPLAY.get(version, version)}** "
                    "(fichier `strikes_eval_2023_v6.csv` absent). "
                    "Affichage des résultats officiels fixes — θ = 0.30 (valeur optimale)."
                )
                s = jury_all.get(version, {}).get("summary", {})
                df_pred = jury_all.get(version, {}).get("predictions")
                if s:
                    st.subheader(f"Résultats sur données jury 2023 — {MODEL_DISPLAY.get(version, version)} — θ = 0.30")
                    # Métriques principales
                    c1, c2, c3, c4, c5 = st.columns(5)
                    c1.metric("Alertes analysées",      f"{s['n_alerts']:,}")
                    c2.metric("⏱ Temps économisé",      f"{s['Gain_h']:.2f} h",
                              help="vs règle fixe +30 min après dernier éclair")
                    c3.metric("Gain moyen / alerte",    f"{s.get('Gain_moy_min', 0):.1f} min")
                    c4.metric("⚠️ Taux de risque",      f"{s['Risk_pct']:.4f} %",
                              delta="✓ Sous 2%" if s["Risk_pct"] < 2 else "⚠️",
                              delta_color="normal" if s["Risk_pct"] < 2 else "inverse",
                              help="Limite jury : < 2%")
                    c5.metric("C-index (validation)",   f"{s['c_index_test']:.4f}")
                    st.divider()
                    z1, z2 = st.columns(2)
                    z1.metric("Zone critique < 3 km", f"{s['M_L3']} / {s['N_L3']} éclairs manqués",
                              delta="✓ Sous 2%" if s["Risk_pct"] < 2 else "⚠️ Dépassement",
                              delta_color="normal" if s["Risk_pct"] < 2 else "inverse")
                    z2.metric("Seuil θ optimal", f"{s['theta_opt']:.2f}",
                              help="Optimisé sur les données de validation 2021-2022")
                    if df_pred is not None:
                        st.divider()
                        # Graphes à partir des prédictions par alerte
                        df_plot = df_pred.rename(columns={
                            "n_L3":    "n_danger",
                            "n_missed": "n_manques",
                        })
                        gcol1, gcol2 = st.columns(2)
                        with gcol1:
                            st.plotly_chart(plot_gain_par_aeroport(df_plot), use_container_width=True)
                        with gcol2:
                            st.plotly_chart(plot_distribution_gains(df_plot), use_container_width=True)
                        # Tableau par alerte
                        st.subheader("Résultats par alerte")
                        cols_pred = [c for c in ["airport", "airport_alert_id", "n_strikes",
                                                 "n_L3", "n_missed", "gain_min", "dist_min"]
                                     if c in df_pred.columns]
                        st.caption(f"Résultat officiel sur {len(df_pred):,} alertes jury — θ = 0.30 (optimal)")
                        st.dataframe(
                            df_pred[cols_pred]
                            .rename(columns={
                                "airport":          "Aéroport",
                                "airport_alert_id": "ID Alerte",
                                "n_strikes":        "Éclairs",
                                "n_L3":             "< 3 km",
                                "n_missed":         "Manqués",
                                "gain_min":         "Gain (min)",
                                "dist_min":         "Dist. min (km)",
                            }).sort_values("Gain (min)", ascending=False),
                            use_container_width=True, hide_index=True, height=450,
                        )
            else:
                with st.spinner("Calcul des statistiques jury en cours..."):
                    jury_stats, jury_agg = compute_jury_stats(
                        jury_strikes, theta,
                        airport=airport_sel, saison=saison_num, only_l3=only_l3,
                    )
                sref = jury_all.get(version, {}).get("summary", {})
                if sref:
                    dgain = float(jury_agg.get("gain_h", 0)) - float(sref.get("Gain_h", 0))
                    st.caption(
                        f"Référence officielle θ=0.30 : Gain = {sref['Gain_h']:.2f} h | "
                        f"Risque = {sref['Risk_pct']:.4f}% — "
                        f"Avec θ={theta:.2f} : Δ Gain = **{dgain:+.2f} h**"
                    )
                st.subheader(f"Résultats sur données jury 2023 — {MODEL_DISPLAY.get(version, version)} — θ = {theta:.2f}")
                afficher_metriques(jury_stats, jury_agg, c_idx, theta)
        except Exception as _jury_err:
            st.error(f"Erreur lors du calcul des métriques jury : {_jury_err}")
            st.exception(_jury_err)


# ═════════════════════════════════════════════════════════════════
# TAB 2 — JURY 2023
# ═════════════════════════════════════════════════════════════════
with tab_jury:
    if not jury_all:
        st.warning("Aucun résultat jury trouvé dans results/")
    else:
        st.subheader("Évaluation officielle — Données jury indépendantes 2023-2025")
        st.markdown(
            "Ces résultats sont calculés sur un jeu de données **jamais vu à l'entraînement**, "
            "avec le protocole officiel du jury Météorage. Le seuil θ = 0.30 a été optimisé "
            "sur les données de validation (2021-2022)."
        )
        st.divider()

        # KPIs côte à côte
        cols = st.columns(len(jury_all))
        for col, (vname, vdata) in zip(cols, jury_all.items()):
            s = vdata["summary"]
            col.markdown(f"#### {MODEL_DISPLAY.get(vname, vname)}")
            col.metric("⏱ Temps économisé",      f"{s['Gain_h']:.2f} h")
            col.metric("Gain moyen / alerte",     f"{s.get('Gain_moy_min', 0):.1f} min")
            col.metric("⚠️ Taux de risque",       f"{s['Risk_pct']:.4f} %",
                       delta="✓ Sous 2%" if s["Risk_pct"] < 2 else "⚠️",
                       delta_color="normal" if s["Risk_pct"] < 2 else "inverse")
            col.metric("Éclairs < 3 km manqués", f"{s['M_L3']} / {s['N_L3']}")
            col.metric("C-index (validation)",    f"{s['c_index_test']:.4f}")
            col.metric("Seuil θ optimal",         f"{s['theta_opt']:.2f}")
            col.caption(f"Évalué sur {s['n_alerts']} alertes | {s['n_strikes']:,} éclairs")

        st.divider()

        if len(jury_all) >= 2:
            st.plotly_chart(plot_jury_comparaison(jury_all), use_container_width=True)
            keys = list(jury_all.keys())
            s6   = jury_all[keys[0]]["summary"]
            s7   = jury_all[keys[1]]["summary"]
            dg   = s7["Gain_h"]    - s6["Gain_h"]
            dr   = s7["Risk_pct"]  - s6["Risk_pct"]
            dc   = s7["c_index_test"] - s6["c_index_test"]
            st.info(
                f"**v7 vs v6 :** C-index `{dc:+.4f}` | "
                f"Temps économisé `{dg:+.2f} h` | Taux risque `{dr:+.4f}%` — "
                "Un meilleur C-index ne garantit pas un meilleur gain jury "
                "(classement ≠ calibration temporelle)."
            )
            st.divider()

        # Tableau jury par alerte
        vsel = st.selectbox("Voir les alertes de :", list(jury_all.keys()),
                            format_func=lambda x: MODEL_DISPLAY.get(x, x))
        dp   = jury_all[vsel].get("predictions")
        if dp is not None:
            show_risk = st.checkbox("Alertes risquées seulement", value=False)
            if show_risk:
                dp = dp[dp["is_risky"] == True]
            cols_show = [c for c in
                         ["airport", "airport_alert_id", "n_strikes", "n_L3",
                          "n_missed", "gain_min", "dist_min"] if c in dp.columns]
            st.dataframe(
                dp[cols_show].rename(columns={
                    "airport": "Aéroport", "airport_alert_id": "ID Alerte",
                    "n_strikes": "Éclairs", "n_L3": "< 3 km",
                    "n_missed": "Manqués", "gain_min": "Gain (min)",
                    "dist_min": "Dist. min (km)",
                }).sort_values("Gain (min)", ascending=False),
                use_container_width=True, hide_index=True, height=450,
            )


# ═════════════════════════════════════════════════════════════════
# TAB 3 — ANALYSE D'UN ORAGE
# ═════════════════════════════════════════════════════════════════
with tab_orage:
    # ── Sélection de l'alerte ─────────────────────────────────────
    df_filt = df_scored.copy()
    if airport_sel != "Tous":
        df_filt = df_filt[df_filt["airport"] == airport_sel]
    if saison_num is not None:
        df_filt = df_filt[df_filt["alert_saison"] == saison_num]
    if only_l3:
        ids_l3  = (df_filt[df_filt["dist"] < DIST_3KM][["airport", "airport_alert_id"]]
                   .drop_duplicates())
        df_filt = df_filt.merge(ids_l3, on=["airport", "airport_alert_id"])

    dispo = (df_filt[["airport", "airport_alert_id"]]
             .drop_duplicates()
             .sort_values(["airport", "airport_alert_id"]))

    if len(dispo) == 0:
        st.warning("Aucune alerte disponible avec ces filtres.")
    else:
        # Sélecteur compact en deux colonnes
        sel_col1, sel_col2 = st.columns([2, 1])
        with sel_col1:
            opts = [f"{r['airport']} — Alerte #{r['airport_alert_id']}"
                    for _, r in dispo.iterrows()]
            choix = st.selectbox(f"Sélectionner un orage ({len(opts)} disponibles)", opts)
        chosen_airport  = choix.split(" — ")[0]
        chosen_alert_id = int(choix.split("#")[1])

        alert_df = (df_scored[
            (df_scored["airport"] == chosen_airport) &
            (df_scored["airport_alert_id"] == chosen_alert_id)
        ].sort_values("date").reset_index(drop=True).copy())

        # ── Validation par éclair : prédiction vs réalité ─────────
        # Pour chaque éclair i, on compare horizon_min (T*_i) au délai
        # réel jusqu'au prochain éclair de la même alerte.
        next_date = alert_df["date"].shift(-1)
        alert_df["gap_next_min"] = (
            (next_date - alert_df["date"]).dt.total_seconds() / 60
        )
        alert_df["dist_next"] = alert_df["dist"].shift(-1)
        # Prédiction validée si :
        #   - pas de prochain éclair (dernière observation), OU
        #   - le prochain éclair arrive APRÈS l'horizon prédit T*_i
        alert_df["prediction_ok"] = (
            alert_df["gap_next_min"].isna()
            | (alert_df["gap_next_min"] > alert_df["horizon_min"])
        )

        # ── Métriques de l'orage ──────────────────────────────────
        dur_min   = (alert_df["date"].max() - alert_df["date"].min()).total_seconds() / 60
        n_3km     = int((alert_df["dist"] < DIST_3KM).sum())
        saison_v  = int(alert_df["saison"].iloc[0])
        amp_max   = alert_df["amplitude"].abs().max() if "amplitude" in alert_df.columns else None

        t_regle = alert_df["date"].max() + pd.Timedelta(minutes=MAX_GAP)
        above   = alert_df[alert_df["confiance"] > theta]

        if len(above) == 0:
            t_modele  = t_regle
            gain_min  = 0.0
            idx_min   = None
        else:
            idx_min  = above["fin_predite"].idxmin()
            t_modele = above.loc[idx_min, "fin_predite"]
            gain_min = max((t_regle - t_modele).total_seconds() / 60, 0.0)

        def missed_in_alert(dist_max):
            return int((alert_df[alert_df["dist"] < dist_max]["date"] >= t_modele).sum())

        n_mq_3  = missed_in_alert(3)
        n_mq_10 = missed_in_alert(10)
        n_mq_20 = missed_in_alert(20)
        n_10km  = int((alert_df["dist"] < 10).sum())
        n_20km  = int((alert_df["dist"] < 20).sum())

        # ── Bandeau de métriques ──────────────────────────────────
        st.subheader(f"Orage — {chosen_airport}  |  Alerte #{chosen_alert_id}")

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Saison",         SAISON_LABELS.get(saison_v, "—"))
        m2.metric("Début",          alert_df["date"].min().strftime("%H:%M UTC"))
        m3.metric("Durée",          f"{dur_min:.0f} min")
        m4.metric("Éclairs total",  len(alert_df))
        m5.metric("Éclairs < 3 km", n_3km,
                  delta="Zone danger" if n_3km > 0 else "Hors zone",
                  delta_color="inverse" if n_3km > 0 else "normal")
        m6.metric("Amplitude max",  f"{amp_max:.0f} kA" if amp_max else "—")

        # Résultat prédiction
        r1, r2, r3 = st.columns(3)
        r1.metric("Fin prédite par le modèle", t_modele.strftime("%H:%M:%S UTC"),
                  delta=f"Gain : {gain_min:.0f} min" if gain_min > 0 else "Fallback règle",
                  delta_color="normal" if gain_min > 0 else "off")
        r2.metric("Règle +30 min", t_regle.strftime("%H:%M:%S UTC"))
        r3.metric("Éclairs confiants (> θ)", f"{len(above)} / {len(alert_df)}")

        # Éclairs manqués par zone
        st.markdown("**Éclairs manqués après la fin prédite — par zone**")
        z1, z2, z3 = st.columns(3)
        z1.metric(
            "Zone critique < 3 km",
            f"{n_mq_3} / {n_3km}" if n_3km > 0 else "0 éclair",
            delta="✓ Aucun manqué" if n_mq_3 == 0 else f"⚠️ {n_mq_3} manqué(s)",
            delta_color="normal" if n_mq_3 == 0 else "inverse",
        )
        z2.metric(
            "Zone alerte < 10 km",
            f"{n_mq_10} / {n_10km}" if n_10km > 0 else "0 éclair",
            delta="✓ Aucun manqué" if n_mq_10 == 0 else f"⚠️ {n_mq_10} manqué(s)",
            delta_color="normal" if n_mq_10 == 0 else "inverse",
        )
        z3.metric(
            "Zone surveillance < 20 km",
            f"{n_mq_20} / {n_20km}" if n_20km > 0 else "0 éclair",
            delta="✓ Aucun manqué" if n_mq_20 == 0 else f"⚠️ {n_mq_20} manqué(s)",
            delta_color="normal" if n_mq_20 == 0 else "inverse",
        )

        st.divider()

        # ── Visualisation principale ──────────────────────────────
        st.plotly_chart(
            plot_analyse_orage(alert_df, t_modele, t_regle, gain_min, theta),
            use_container_width=True,
        )

        # ── Tableau de validation par éclair ──────────────────────
        st.markdown("### 🎯 Validation prédiction éclair par éclair")
        n_ok = int(alert_df["prediction_ok"].sum())
        n_tot = len(alert_df)
        v1, v2, v3 = st.columns(3)
        v1.metric("Prédictions validées ✓", f"{n_ok} / {n_tot}",
                  delta=f"{n_ok/n_tot*100:.0f} %")
        v2.metric("Prédictions violées ✗", f"{n_tot - n_ok}",
                  delta="éclair arrivé avant T*_i" if n_tot - n_ok > 0 else "Aucune",
                  delta_color="inverse" if n_tot - n_ok > 0 else "normal")
        v3.metric("T*_i moyen prédit",
                  f"{alert_df['horizon_min'].mean():.1f} min")

        st.caption(
            "Pour chaque éclair $i$, le modèle prédit qu'aucun nouvel éclair "
            "n'arrivera dans les $T^*_i$ prochaines minutes. "
            "**Vert ✓** = prédiction validée (le prochain éclair arrive après $T^*_i$, "
            "ou aucun ne suit). **Rouge ✗** = prédiction violée."
        )

        table = alert_df[[
            "date", "dist", "confiance", "horizon_min",
            "gap_next_min", "dist_next", "prediction_ok",
        ]].copy()
        table["statut"] = np.where(table["prediction_ok"], "✓ Validée", "✗ Violée")
        table["date"] = table["date"].dt.strftime("%H:%M:%S")
        table["dist"] = table["dist"].round(2)
        table["confiance"] = table["confiance"].round(3)
        table["horizon_min"] = table["horizon_min"].round(1)
        table["gap_next_min"] = table["gap_next_min"].round(1)
        table["dist_next"] = table["dist_next"].round(2)
        table = table.drop(columns=["prediction_ok"])
        table.columns = [
            "Heure (UTC)", "Dist. (km)", "Confiance S(30)",
            "T*_i prédit (min)", "Délai réel (min)",
            "Dist. prochain (km)", "Statut",
        ]

        def _row_style(row):
            ok = row["Statut"].startswith("✓")
            bg = "background-color: #ECFDF5" if ok else "background-color: #FEF2F2"
            return [bg] * len(row)

        st.dataframe(
            table.style.apply(_row_style, axis=1),
            hide_index=True, use_container_width=True, height=320,
        )

        # ── Courbe de survie (éclair déterminant) ─────────────────
        with st.expander("Fonction de survie — éclair qui détermine la prédiction", expanded=False):
            with st.spinner("Calcul de la courbe de survie..."):
                X_s      = scaler.transform(alert_df[features].values.astype(float))
                surv_fns = gbs.predict_survival_function(X_s)

            if idx_min is not None:
                pos = alert_df.index.get_loc(idx_min)
            else:
                pos = len(alert_df) - 1

            fn_sel   = surv_fns[pos]
            t_star_s = float(alert_df.iloc[pos]["horizon_min"])
            ecl_date = alert_df.iloc[pos]["date"]
            ecl_dist = float(alert_df.iloc[pos]["dist"])

            st.plotly_chart(
                plot_courbe_survie(fn_sel, t_star_s, float(alert_df.iloc[pos]["confiance"]),
                                   ecl_date, ecl_dist),
                use_container_width=True,
            )
            st.caption(
                f"Cet éclair (dist. {ecl_dist:.1f} km, "
                f"confiance S(30) = {alert_df.iloc[pos]['confiance']:.3f}) "
                f"détermine la fin prédite de l'alerte via T* = {t_star_s:.0f} min."
            )

        # ── Données brutes ────────────────────────────────────────
        with st.expander("Données de l'alerte (éclairs individuels)", expanded=False):
            cols_show = ["date", "dist", "confiance", "horizon_min"]
            if "amplitude" in alert_df.columns:
                cols_show.insert(2, "amplitude")
            st.dataframe(
                alert_df[cols_show].rename(columns={
                    "date":       "Horodatage (UTC)",
                    "dist":       "Distance (km)",
                    "amplitude":  "Amplitude (kA)",
                    "confiance":  "Confiance S(30)",
                    "horizon_min":"Horizon T* (min)",
                }).round(3),
                use_container_width=True,
            )


# ═════════════════════════════════════════════════════════════════
# TAB 4 — DÉMO EN DIRECT (simulation strike-by-strike)
# ═════════════════════════════════════════════════════════════════
with tab_demo:
    st.subheader("🎬 Simulation d'une alerte foudre — éclair par éclair")
    st.markdown(
        "Les éclairs arrivent **un par un** dans l'ordre chronologique réel. "
        "Le modèle GBS recalcule sa confiance à chaque éclair. "
        "Dès qu'un éclair atteint $S(30) > \\theta$, l'alerte peut être levée."
    )

    # ── Sélection : on propose des alertes "démonstratrices" ─────
    demo_summary = (df_scored.groupby(["airport", "airport_alert_id"])
                    .agg(n_strikes=("date", "count"),
                         max_conf=("confiance", "max"),
                         dist_min=("dist", "min"),
                         t_start=("date", "min"),
                         t_end=("date", "max"))
                    .reset_index())
    demo_summary["duree_min"] = (
        (demo_summary["t_end"] - demo_summary["t_start"]).dt.total_seconds() / 60
    )
    # Critère "bonne démo" : assez d'éclairs, assez de durée, modèle qui se déclenche
    candidats = demo_summary[
        (demo_summary["n_strikes"] >= 8)
        & (demo_summary["duree_min"] >= 20)
        & (demo_summary["max_conf"] > theta)
    ].copy()
    if airport_sel != "Tous":
        candidats = candidats[candidats["airport"] == airport_sel]
    candidats = candidats.sort_values(
        ["max_conf", "n_strikes"], ascending=[False, False]
    ).head(40)

    if len(candidats) == 0:
        st.info(
            "Aucune alerte « démo » disponible avec ces filtres "
            "(il faut ≥ 8 éclairs, ≥ 20 min, et un déclenchement du modèle). "
            "Choisissez « Tous » en aéroport ou baissez θ."
        )
    else:
        sc1, sc2 = st.columns([3, 1])
        with sc1:
            opts_demo = [
                f"{r['airport']} — Alerte #{r['airport_alert_id']} "
                f"({r['n_strikes']} éclairs · {r['duree_min']:.0f} min · "
                f"S(30) max = {r['max_conf']:.2f})"
                for _, r in candidats.iterrows()
            ]
            demo_choix = st.selectbox(
                f"Choisir une alerte de démo  ({len(candidats)} disponibles)",
                opts_demo, key="demo_select",
            )
        with sc2:
            delay_s = st.select_slider(
                "Cadence",
                options=[1.5, 2.0, 2.5, 3.0, 4.0],
                value=2.5,
                format_func=lambda x: f"⏱ {x:.1f} s / éclair",
                key="demo_delay",
                help="Délai entre deux éclairs (effet dramatique : 2 à 3 s).",
            )

        demo_airport  = demo_choix.split(" — ")[0]
        demo_alert_id = int(demo_choix.split("#")[1].split(" ")[0])

        # Reset state si on change d'alerte
        alert_key = (demo_airport, demo_alert_id)
        if st.session_state.get("demo_alert_key") != alert_key:
            st.session_state["demo_alert_key"] = alert_key
            st.session_state["demo_idx"]     = 0
            st.session_state["demo_playing"] = False

        demo_df = (df_scored[
            (df_scored["airport"]          == demo_airport) &
            (df_scored["airport_alert_id"] == demo_alert_id)
        ].sort_values("date").reset_index(drop=True).copy())

        n_strikes    = len(demo_df)
        t_regle_demo = demo_df["date"].max() + pd.Timedelta(minutes=MAX_GAP)
        playing      = st.session_state.get("demo_playing", False)
        idx          = min(st.session_state.get("demo_idx", 0), n_strikes - 1)

        # ── Contrôles ────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns([1, 1, 1, 4])
        with c1:
            if st.button("⏸ Pause" if playing else "▶ Play",
                         key="demo_play_btn", use_container_width=True):
                st.session_state["demo_playing"] = not playing
                st.rerun()
        with c2:
            if st.button("⏮ Reset", key="demo_reset_btn", use_container_width=True):
                st.session_state["demo_idx"]     = 0
                st.session_state["demo_playing"] = False
                st.rerun()
        with c3:
            if st.button("⏭ Fin", key="demo_end_btn", use_container_width=True):
                st.session_state["demo_idx"]     = n_strikes - 1
                st.session_state["demo_playing"] = False
                st.rerun()
        with c4:
            st.progress(
                (idx + 1) / n_strikes,
                text=f"Éclair {idx + 1} / {n_strikes}"
                     f"  ·  {demo_df.iloc[idx]['date'].strftime('%H:%M:%S UTC')}",
            )

        # ── État courant ────────────────────────────────────────
        seen       = demo_df.iloc[: idx + 1].copy()
        # Délai et distance du prochain éclair (connus seulement pour
        # les éclairs qui ne sont pas le dernier vu)
        seen["gap_next_min"] = (
            (seen["date"].shift(-1) - seen["date"]).dt.total_seconds() / 60
        )
        seen["dist_next"]      = seen["dist"].shift(-1)
        seen["pred_validable"] = seen["gap_next_min"].notna()
        # Une prédiction n'est VIOLÉE que si le prochain éclair arrive
        # AVANT T*_i ET en zone dangereuse (< 3 km). Un éclair qui arrive
        # vite mais à 15 km n'est PAS un échec — la zone aéroport reste sûre.
        seen["pred_violated"] = (
            seen["pred_validable"]
            & (seen["gap_next_min"] <= seen["horizon_min"])
            & (seen["dist_next"] < DIST_3KM)
        )
        # Validée = validable ET non violée
        seen["pred_ok"]         = seen["pred_validable"] & ~seen["pred_violated"]
        # Sous-cas "safe" : éclair dans la fenêtre mais hors zone danger
        seen["pred_safe_zone"]  = (
            seen["pred_validable"]
            & (seen["gap_next_min"] <= seen["horizon_min"])
            & (seen["dist_next"] >= DIST_3KM)
        )

        current    = demo_df.iloc[idx]
        seen_above = seen[seen["confiance"] > theta]

        if len(seen_above) > 0:
            t_modele_demo = seen_above["fin_predite"].min()
            gain_demo     = max((t_regle_demo - t_modele_demo).total_seconds() / 60, 0)
            status_color  = "#10B981"
            status_emoji  = "🟢"
            status_text   = (
                f"<b>MODÈLE ACTIF</b> — Levée prédite à "
                f"<b>{t_modele_demo.strftime('%H:%M:%S UTC')}</b> "
                f"(gain : <b>{gain_demo:.0f} min</b> sur la règle des 30 min)"
            )
        else:
            t_modele_demo = t_regle_demo
            gain_demo     = 0.0
            status_color  = "#F59E0B"
            status_emoji  = "🟡"
            status_text   = (
                f"<b>MODÈLE EN ATTENTE</b> — Aucun éclair encore confiant "
                f"(seuil S(30) &gt; {theta:.2f}). Fallback : règle 30 min à "
                f"<b>{t_regle_demo.strftime('%H:%M:%S UTC')}</b>"
            )

        st.markdown(
            f"<div style='padding:14px 18px;background:{status_color}1A;"
            f"border-left:5px solid {status_color};border-radius:6px;"
            f"font-size:1.05rem;margin:8px 0'>{status_emoji}  {status_text}</div>",
            unsafe_allow_html=True,
        )

        # ── Flash dramatique : nouvel éclair ────────────────────
        flash_bg = "linear-gradient(90deg,#FEF3C7 0%,#FDE68A 50%,#FEF3C7 100%)"
        st.markdown(
            f"""
            <div style='padding:10px 16px;background:{flash_bg};
                        border:2px solid #F59E0B;border-radius:8px;
                        text-align:center;margin:6px 0 12px 0;
                        font-size:1.15rem;font-weight:700;color:#92400E;
                        box-shadow:0 0 16px rgba(251,191,36,0.5);'>
                ⚡ NOUVEL ÉCLAIR DÉTECTÉ &nbsp;·&nbsp;
                {current['date'].strftime('%H:%M:%S UTC')} &nbsp;·&nbsp;
                {current['dist']:.1f} km de l'aéroport
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Métriques temps réel de l'éclair courant ────────────
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("⏱ Horodatage", current["date"].strftime("%H:%M:%S"))
        m2.metric("📏 Distance",   f"{current['dist']:.1f} km",
                  delta="< 3 km ⚠️" if current["dist"] < DIST_3KM else "Hors zone",
                  delta_color="inverse" if current["dist"] < DIST_3KM else "off")
        m3.metric("🔋 Confiance S(30)", f"{current['confiance']:.3f}",
                  delta=f"> θ={theta:.2f} ✓" if current["confiance"] > theta
                                              else f"≤ θ={theta:.2f}",
                  delta_color="normal" if current["confiance"] > theta else "off")
        m4.metric("⏳ T*_i prédit (silence attendu)", f"{current['horizon_min']:.1f} min",
                  help="Minutes pendant lesquelles le modèle prédit qu'aucun nouvel éclair n'arrivera.")
        m5.metric("📍 Fin si actif", current["fin_predite"].strftime("%H:%M:%S"))

        # ── Vérification de la prédiction précédente ────────────
        if idx >= 1:
            prev      = demo_df.iloc[idx - 1]
            gap_real  = (current["date"] - prev["date"]).total_seconds() / 60
            t_pred    = prev["horizon_min"]
            ecart     = gap_real - t_pred       # >0 : silence respecté
            dist_cur  = float(current["dist"])
            in_window = gap_real <= t_pred      # arrivé pendant le silence prédit
            danger    = dist_cur < DIST_3KM     # < 3 km = zone critique

            if not in_window:
                v_state  = "ok"
                v_label  = "✓ PRÉDICTION VALIDÉE"
                v_color  = "#10B981"
                v_explain = (
                    f"Silence prédit de <b>{t_pred:.1f} min</b> respecté : "
                    f"le prochain éclair (celui-ci) est arrivé après "
                    f"<b>{gap_real:.1f} min</b> (écart : <b>{ecart:+.1f} min</b>)."
                )
            elif in_window and not danger:
                v_state  = "safe"
                v_label  = "✓ ÉCART ACCEPTABLE (zone safe)"
                v_color  = "#10B981"
                v_explain = (
                    f"L'éclair arrive en <b>{gap_real:.1f} min</b> "
                    f"(prédit ≥ {t_pred:.1f} min), <u>mais à <b>{dist_cur:.1f} km</b> "
                    f"de l'aéroport</u> — hors zone danger (≥ 3 km). "
                    f"La sécurité de la piste n'est <b>pas compromise</b>."
                )
            else:  # in_window AND danger
                v_state  = "violated"
                v_label  = "✗ PRÉDICTION VIOLÉE (vraie alerte)"
                v_color  = "#EF4444"
                v_explain = (
                    f"L'éclair arrive en <b>{gap_real:.1f} min</b> "
                    f"(prédit ≥ {t_pred:.1f} min) <b>ET à {dist_cur:.1f} km</b> "
                    f"&lt; 3 km : <b>vrai échec de prédiction</b> — "
                    f"zone aéroport compromise."
                )

            st.markdown(
                f"<div style='padding:10px 16px;background:{v_color}15;"
                f"border-left:4px solid {v_color};border-radius:6px;"
                f"margin:8px 0;font-size:0.98rem'>"
                f"<b style='color:{v_color}'>{v_label}</b> &nbsp;·&nbsp; "
                f"{v_explain}</div>",
                unsafe_allow_html=True,
            )
            cmp1, cmp2, cmp3, cmp4 = st.columns(4)
            cmp1.metric("Silence prédit (T*)", f"{t_pred:.1f} min")
            cmp2.metric("Délai réel",          f"{gap_real:.1f} min",
                        delta=f"{ecart:+.1f} min",
                        delta_color="normal" if not in_window else "off")
            cmp3.metric("Distance éclair",     f"{dist_cur:.1f} km",
                        delta="zone danger ⚠️" if danger else "zone safe ✓",
                        delta_color="inverse" if danger else "normal")
            cmp4.metric("Verdict",
                        {"ok": "✓ Validée",
                         "safe": "✓ Safe",
                         "violated": "✗ Violée"}[v_state])

        # ── Graphe live ──────────────────────────────────────────
        fig_d = make_subplots(
            rows=3, cols=1, row_heights=[0.42, 0.30, 0.28],
            shared_xaxes=True, vertical_spacing=0.07,
            subplot_titles=[
                "Distance des éclairs et fin d'alerte prédite",
                "⏳ T*_i prédit (barre) vs délai réel (◆) — violation = éclair rapide ET &lt; 3 km",
                f"Confiance S(30) — seuil θ = {theta:.2f}",
            ],
        )

        # Éclairs à venir — fantôme gris
        future = demo_df.iloc[idx + 1:]
        if len(future):
            fig_d.add_trace(go.Scatter(
                x=future["date"], y=future["dist"],
                mode="markers", name="À venir (non observés)",
                marker=dict(color="#D1D5DB", size=6, opacity=0.35),
                hoverinfo="skip",
            ), row=1, col=1)

        # Éclairs vus, < θ
        seen_low = seen[seen["confiance"] <= theta]
        if len(seen_low):
            fig_d.add_trace(go.Scatter(
                x=seen_low["date"], y=seen_low["dist"],
                mode="markers", name="Éclair vu (conf ≤ θ)",
                marker=dict(color="#9CA3AF", size=10,
                            line=dict(width=1, color="#4B5563")),
                hovertemplate="%{x|%H:%M:%S} — %{y:.1f} km<extra></extra>",
            ), row=1, col=1)

        # Éclairs vus, > θ (confiants)
        seen_hi = seen[seen["confiance"] > theta]
        if len(seen_hi):
            fig_d.add_trace(go.Scatter(
                x=seen_hi["date"], y=seen_hi["dist"],
                mode="markers", name="Éclair confiant (conf > θ) ✓",
                marker=dict(color="#10B981", size=12, symbol="circle",
                            line=dict(width=1.5, color="#065F46")),
                hovertemplate=("%{x|%H:%M:%S} — %{y:.1f} km<br>"
                               "S(30) = %{customdata:.3f}<extra></extra>"),
                customdata=seen_hi["confiance"],
            ), row=1, col=1)

        # Éclair courant — effet halo lumineux (3 cercles concentriques) + étoile
        for halo_size, halo_op in [(60, 0.15), (40, 0.25), (26, 0.4)]:
            fig_d.add_trace(go.Scatter(
                x=[current["date"]], y=[current["dist"]],
                mode="markers", showlegend=False,
                marker=dict(color="#FBBF24", size=halo_size, opacity=halo_op,
                            line=dict(width=0)),
                hoverinfo="skip",
            ), row=1, col=1)
        # Trait d'éclair tombant depuis le haut du graphe
        y_top_panel = max(demo_df["dist"].max() * 1.15, 15)
        fig_d.add_trace(go.Scatter(
            x=[current["date"], current["date"], current["date"]],
            y=[y_top_panel, current["dist"] + 1, current["dist"]],
            mode="lines", showlegend=False,
            line=dict(color="#FBBF24", width=3, dash="solid"),
            opacity=0.7, hoverinfo="skip",
        ), row=1, col=1)
        # Étoile rouge au point d'impact
        fig_d.add_trace(go.Scatter(
            x=[current["date"]], y=[current["dist"]],
            mode="markers+text", name="⚡ Éclair en cours",
            marker=dict(color="#DC2626", size=26, symbol="star",
                        line=dict(width=2.5, color="#7F1D1D")),
            text=["⚡"], textposition="top center",
            textfont=dict(size=22, color="#FBBF24"),
            hovertemplate=("<b>ÉCLAIR ACTUEL</b><br>"
                           "%{x|%H:%M:%S} — %{y:.1f} km<extra></extra>"),
        ), row=1, col=1)

        # Lignes verticales
        if gain_demo > 0:
            fig_d.add_vrect(
                x0=t_modele_demo.timestamp() * 1000,
                x1=t_regle_demo.timestamp()  * 1000,
                fillcolor="#10B981", opacity=0.10, line_width=0, row=1, col=1,
            )
            fig_d.add_vline(
                x=t_modele_demo.timestamp() * 1000,
                line_color="#10B981", line_width=2.5,
                annotation_text=f"<b>Levée {t_modele_demo.strftime('%H:%M')}</b>",
                annotation_font_color="#10B981",
                annotation_position="top left",
                row=1, col=1,
            )
        fig_d.add_vline(
            x=t_regle_demo.timestamp() * 1000,
            line_color="#F59E0B", line_dash="dash", line_width=2,
            annotation_text=f"Règle 30 min ({t_regle_demo.strftime('%H:%M')})",
            annotation_font_color="#F59E0B",
            annotation_position="top right",
            row=1, col=1,
        )
        fig_d.add_hline(y=DIST_3KM, line_dash="dot", line_color="#EF4444",
                        line_width=1.3, row=1, col=1,
                        annotation_text="3 km", annotation_position="right",
                        annotation_font_color="#EF4444")

        # ── Panneau 2 : T*_i prédit (barres) vs délai réel (◆) ─
        valid  = seen[seen["pred_validable"]].copy()
        # Barres : colorées vert si validée, rouge si violée
        if len(valid):
            bar_colors = ["#10B981" if ok else "#EF4444"
                          for ok in valid["pred_ok"]]
            fig_d.add_trace(go.Bar(
                x=valid["date"], y=valid["horizon_min"],
                marker_color=bar_colors, opacity=0.55,
                name="T*_i prédit",
                hovertemplate=("<b>%{x|%H:%M:%S}</b><br>"
                               "T*_i prédit : %{y:.1f} min<extra></extra>"),
            ), row=2, col=1)
            # Diamants : délai réel observé
            fig_d.add_trace(go.Scatter(
                x=valid["date"], y=valid["gap_next_min"],
                mode="markers", name="Délai réel (◆)",
                marker=dict(color="#1E40AF", size=11, symbol="diamond",
                            line=dict(width=1.5, color="white")),
                hovertemplate=("<b>%{x|%H:%M:%S}</b><br>"
                               "Prochain éclair après : %{y:.1f} min"
                               "<extra></extra>"),
            ), row=2, col=1)
            # Petits segments verticaux : écart prédit/réel
            for _, r in valid.iterrows():
                color_seg = "#10B981" if r["pred_ok"] else "#EF4444"
                fig_d.add_trace(go.Scatter(
                    x=[r["date"], r["date"]],
                    y=[r["horizon_min"], r["gap_next_min"]],
                    mode="lines", showlegend=False,
                    line=dict(color=color_seg, width=2, dash="dot"),
                    hoverinfo="skip",
                ), row=2, col=1)

        # Barre courante (en attente) : T*_i de l'éclair actuel, jaune
        fig_d.add_trace(go.Bar(
            x=[current["date"]], y=[current["horizon_min"]],
            marker_color="#FBBF24", opacity=0.7,
            name="T*_i en attente",
            marker_line=dict(width=1.5, color="#92400E"),
            hovertemplate=("<b>EN ATTENTE</b><br>"
                           "T*_i prédit : %{y:.1f} min<br>"
                           "(prochain éclair non encore observé)"
                           "<extra></extra>"),
        ), row=2, col=1)

        # ── Panneau 3 : confiance ──────────────────────────────
        fig_d.add_trace(go.Scatter(
            x=seen["date"], y=seen["confiance"],
            mode="lines+markers", name="S(30) observé",
            line=dict(color="#3B82F6", width=2.5),
            marker=dict(size=7, color="#3B82F6"),
            hovertemplate="%{x|%H:%M:%S} — S(30) = %{y:.3f}<extra></extra>",
        ), row=3, col=1)
        # Marqueur courant sur la confiance — halo + étoile
        for halo_size, halo_op in [(36, 0.18), (24, 0.30)]:
            fig_d.add_trace(go.Scatter(
                x=[current["date"]], y=[current["confiance"]],
                mode="markers", showlegend=False,
                marker=dict(color="#FBBF24", size=halo_size, opacity=halo_op),
                hoverinfo="skip",
            ), row=3, col=1)
        fig_d.add_trace(go.Scatter(
            x=[current["date"]], y=[current["confiance"]],
            mode="markers", showlegend=False,
            marker=dict(color="#DC2626", size=18, symbol="star",
                        line=dict(width=1.8, color="#7F1D1D")),
            hoverinfo="skip",
        ), row=3, col=1)
        fig_d.add_hline(y=theta, line_dash="dot", line_color="#10B981",
                        line_width=1.5, row=3, col=1,
                        annotation_text=f"θ = {theta:.2f}",
                        annotation_font_color="#10B981",
                        annotation_position="right")

        # Plage X commune : du début jusqu'à 3 min après la règle 30
        x_min = demo_df["date"].min() - pd.Timedelta(minutes=2)
        x_max = t_regle_demo + pd.Timedelta(minutes=3)
        for r in (1, 2, 3):
            fig_d.update_xaxes(range=[x_min, x_max], row=r, col=1)
        fig_d.update_xaxes(title_text="Heure (UTC)", row=3, col=1)
        fig_d.update_yaxes(title_text="Distance (km)", row=1, col=1,
                           rangemode="tozero")
        fig_d.update_yaxes(title_text="Minutes", row=2, col=1, rangemode="tozero")
        fig_d.update_yaxes(title_text="S(30)", row=3, col=1, range=[0, 1.05])

        fig_d.update_layout(
            height=760, template="plotly_white",
            margin=dict(t=80, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.06, x=0),
            hovermode="x unified",
            bargap=0.15,
        )
        st.plotly_chart(fig_d, use_container_width=True)

        # ── Historique tabulaire ────────────────────────────────
        with st.expander("📋 Éclairs déjà observés", expanded=False):
            t_seen = seen[["date", "dist", "confiance",
                           "horizon_min", "fin_predite"]].copy()
            t_seen["actif"] = np.where(t_seen["confiance"] > theta, "✓", "—")
            t_seen["date"]        = t_seen["date"].dt.strftime("%H:%M:%S")
            t_seen["fin_predite"] = t_seen["fin_predite"].dt.strftime("%H:%M:%S")
            t_seen["dist"]        = t_seen["dist"].round(2)
            t_seen["confiance"]   = t_seen["confiance"].round(3)
            t_seen["horizon_min"] = t_seen["horizon_min"].round(1)
            t_seen.columns = ["Heure", "Dist. (km)", "S(30)",
                              "T* (min)", "Fin si actif", "> θ ?"]
            st.dataframe(t_seen, hide_index=True, use_container_width=True,
                         height=260)

        # ── Auto-advance pendant la lecture ─────────────────────
        if playing:
            if idx < n_strikes - 1:
                time.sleep(delay_s)
                st.session_state["demo_idx"] = idx + 1
                st.rerun()
            else:
                st.session_state["demo_playing"] = False
                if gain_demo > 0:
                    st.success(
                        f"✅ Simulation terminée — alerte levée à "
                        f"**{t_modele_demo.strftime('%H:%M:%S')}** "
                        f"(gain final : **{gain_demo:.0f} min**)"
                    )
                else:
                    st.info(
                        "ℹ️ Simulation terminée — aucun éclair n'a dépassé "
                        f"le seuil θ = {theta:.2f}. Règle 30 min appliquée."
                    )


# ── PIED DE PAGE ─────────────────────────────────────────────────
st.divider()
st.caption(
    f"{MODEL_DISPLAY.get(version, version)} | θ = {theta:.2f} | "
    f"C-index = {c_idx:.4f} | "
    f"Gain validation = {agg_test.get('gain_h', 0):.1f} h | "
    f"Risque = {agg_test.get('taux_risque', 0):.3f}% | "
    "Data Battle Météorage 2026"
)
