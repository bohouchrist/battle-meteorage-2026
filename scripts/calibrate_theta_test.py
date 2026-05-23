"""Calibration honnête de theta sur les données TEST 2021-2022.

Pour chaque modèle (GBS v6 et v7) :
  1. Charger le modèle figé, le scaler (+ label encoder pour v7)
  2. Charger les données 2021-2022 (test, jamais vues à l'entraînement)
  3. Construire les features
  4. Prédire S(30|xi) et T*_i pour chaque éclair de test
  5. Balayer theta sur [0 ; 0,99] (100 valeurs)
  6. Pour chaque theta : calculer (Gain, Risk) selon le protocole jury
  7. Sélectionner theta_opt = celui qui maximise Gain sous contrainte Risk < 2 %

Sortie :
  results/calibration_theta_test_v6.csv
  results/calibration_theta_test_v7.csv
  results/calibration_theta_test_summary.json
"""
from __future__ import annotations
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT       = Path(__file__).resolve().parent.parent
DATA_PATH  = ROOT / "data" / "data_df_with_alert.parquet"
JURY_PATH  = ROOT / "segment_alerts_all_airports_eval.csv"
MODELS_DIR = ROOT / "models"
RESULTS    = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

MAX_GAP   = 30
DIST_3KM  = 3.0
R_MAX     = 0.02   # contrainte jury : Risk < 2 %

FEATURES_V6 = [
    "h_cos", "h_sin", "doy_cos", "doy_sin", "saison",
    "dist_centre", "dist_avg_5", "dist_min_so_far",
    "silence_min", "freq_5min", "rang", "rang_norm",
]
FEATURES_V7 = FEATURES_V6 + ["airport_enc"]


# ─────────────────────────────────────────────────────────────────
# 1. Données TEST 2021-2022
# ─────────────────────────────────────────────────────────────────
def load_test_data() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df[df["date"].dt.year.isin([2021, 2022])].copy()
    df = df.sort_values(["airport", "airport_alert_id", "date"]).reset_index(drop=True)
    return df


def load_jury_data() -> pd.DataFrame:
    df = pd.read_csv(JURY_PATH)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df[df["alert_id"].notna()].copy()
    df = df.rename(columns={"alert_id": "airport_alert_id"})
    df = df.sort_values(["airport", "airport_alert_id", "date"]).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────
# 2. Feature engineering (identique à app.py / modelisation_v7)
# ─────────────────────────────────────────────────────────────────
def build_features(df: pd.DataFrame, le=None) -> pd.DataFrame:
    df = df.copy()
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

    if le is not None:
        df["airport_enc"] = le.transform(df["airport"])

    # Pas de NaN dans les features
    for col in FEATURES_V7:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    return df


# ─────────────────────────────────────────────────────────────────
# 3. Scoring du modèle
# ─────────────────────────────────────────────────────────────────
def score_model(df: pd.DataFrame, gbs, scaler, features) -> pd.DataFrame:
    X        = scaler.transform(df[features].values.astype(float))
    surv_fns = gbs.predict_survival_function(X)
    s30      = np.array([float(fn(MAX_GAP)) for fn in surv_fns])
    t_stars  = np.full(len(surv_fns), float(MAX_GAP))
    for i, (fn, s) in enumerate(zip(surv_fns, s30)):
        if s <= 0:
            continue
        idx = np.searchsorted(-fn.y, -(s / 0.98), side="left")
        if idx < len(fn.x):
            t_stars[i] = min(float(fn.x[idx]), float(MAX_GAP))

    out = df.copy()
    out["confiance"]    = s30
    out["horizon_min"]  = t_stars
    out["fin_predite"]  = out["date"] + pd.to_timedelta(t_stars, unit="m")
    return out


# ─────────────────────────────────────────────────────────────────
# 4. Évaluation d'un theta sur le protocole jury
# ─────────────────────────────────────────────────────────────────
def evaluer_theta(df_scored: pd.DataFrame, theta: float) -> dict:
    g        = df_scored.groupby(["airport", "airport_alert_id"])
    t_regles = g["date"].max() + pd.Timedelta(minutes=MAX_GAP)

    above  = df_scored[df_scored["confiance"] > theta]
    merged = t_regles.rename("t_regle").to_frame()
    if len(above):
        t_modeles = (above.groupby(["airport", "airport_alert_id"])["fin_predite"]
                          .min().rename("fin_predite"))
        merged    = merged.join(t_modeles, how="left")
    else:
        merged["fin_predite"] = pd.NaT

    merged["t_modele"]    = merged["fin_predite"].combine_first(merged["t_regle"])
    merged["gain_min"]    = ((merged["t_regle"] - merged["t_modele"])
                             .dt.total_seconds().clip(lower=0) / 60)
    merged["modele_actif"] = merged["fin_predite"].notna()

    # Éclairs < 3 km manqués
    t_pred_map = merged[["t_modele"]].reset_index()
    z3 = df_scored[df_scored["dist"] < DIST_3KM][
        ["airport", "airport_alert_id", "date"]
    ].copy()
    z3 = z3.merge(t_pred_map, on=["airport", "airport_alert_id"], how="left")
    z3["manque"] = z3["date"] >= z3["t_modele"]

    tot_3 = len(z3)
    mq_3  = int(z3["manque"].sum())
    gain_h    = float(merged["gain_min"].sum() / 60)
    gain_moy  = float(merged["gain_min"].mean())
    n_actif   = int(merged["modele_actif"].sum())

    return {
        "theta":      float(theta),
        "n_alertes":  len(merged),
        "n_actif":    n_actif,
        "pct_actif":  100 * n_actif / max(len(merged), 1),
        "n_L3":       int(tot_3),
        "n_manques":  mq_3,
        "Risk_pct":   100 * mq_3 / tot_3 if tot_3 > 0 else 0.0,
        "Gain_h":     gain_h,
        "Gain_moy":   gain_moy,
    }


# ─────────────────────────────────────────────────────────────────
# 5. Balayage theta et sélection
# ─────────────────────────────────────────────────────────────────
def calibrate(df_scored: pd.DataFrame, n_thetas: int = 100) -> pd.DataFrame:
    thetas = np.linspace(0.0, 0.99, n_thetas)
    rows   = [evaluer_theta(df_scored, t) for t in thetas]
    return pd.DataFrame(rows)


def selectionner_theta_opt(sweep: pd.DataFrame, r_max_pct: float = 2.0) -> dict:
    valid = sweep[sweep["Risk_pct"] < r_max_pct].copy()
    if len(valid) == 0:
        # Aucun theta ne respecte la contrainte : on retourne le plus prudent
        idx = sweep["Risk_pct"].idxmin()
        return sweep.loc[idx].to_dict() | {"status": "no_valid_theta"}
    idx = valid["Gain_h"].idxmax()
    return valid.loc[idx].to_dict() | {"status": "ok"}


# ─────────────────────────────────────────────────────────────────
# 6. Pipeline pour un modèle
# ─────────────────────────────────────────────────────────────────
def calibrate_model(version: str, df_test: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    print(f"\n=== Calibration {version.upper()} sur TEST 2021-2022 ===")
    gbs    = joblib.load(MODELS_DIR / f"gbs_{version}_model.pkl")
    scaler = joblib.load(MODELS_DIR / f"gbs_{version}_scaler.pkl")

    if version == "v7":
        le       = joblib.load(MODELS_DIR / "gbs_v7_label_encoder.pkl")
        features = FEATURES_V7
    else:
        le       = None
        features = FEATURES_V6

    df_feat   = build_features(df_test, le=le)
    df_scored = score_model(df_feat, gbs, scaler, features)

    print(f"  {len(df_scored):,} éclairs notés")
    print(f"  Confiance S(30) : min={df_scored['confiance'].min():.3f} "
          f"max={df_scored['confiance'].max():.3f} "
          f"med={df_scored['confiance'].median():.3f}")

    sweep   = calibrate(df_scored)
    best    = selectionner_theta_opt(sweep, r_max_pct=R_MAX * 100)

    print(f"  theta optimal : {best['theta']:.3f}")
    print(f"  Gain      : {best['Gain_h']:.2f} h")
    print(f"  Risk      : {best['Risk_pct']:.3f} %")
    print(f"  Manqués   : {best['n_manques']} / {best['n_L3']} (<3 km)")
    print(f"  Actif     : {best['n_actif']} / {best['n_alertes']} alertes "
          f"({best['pct_actif']:.0f} %)")

    return best, sweep


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    df_test = load_test_data()
    print(f"TEST 2021-2022 chargé : {len(df_test):,} éclairs · "
          f"{df_test['airport_alert_id'].nunique()} alertes")

    summary = {}
    for version in ("v6", "v7"):
        best, sweep = calibrate_model(version, df_test)
        sweep.to_csv(RESULTS / f"calibration_theta_test_{version}.csv", index=False)
        summary[version] = {
            "theta_opt": float(best["theta"]),
            "Gain_h":    float(best["Gain_h"]),
            "Risk_pct":  float(best["Risk_pct"]),
            "Gain_moy":  float(best["Gain_moy"]),
            "n_manques": int(best["n_manques"]),
            "n_L3":      int(best["n_L3"]),
            "n_actif":   int(best["n_actif"]),
            "n_alertes": int(best["n_alertes"]),
            "status":    best.get("status", "ok"),
        }

    summary["context"] = {
        "test_period":  "2021-2022",
        "test_strikes": int(len(df_test)),
        "test_alerts":  int(df_test["airport_alert_id"].nunique()),
        "constraint":   "Risk < 2%",
        "method":       "sweep theta [0;0.99] in 100 steps, "
                        "max Gain under Risk < 2%",
    }

    # ── ETAPE 4 : evaluation FIGEE sur JURY 2023-2025 ────────────
    if JURY_PATH.exists():
        print("\n\n=== EVALUATION HORS-ECHANTILLON SUR JURY 2023-2025 ===")
        df_jury = load_jury_data()
        n_alerts_jury = df_jury.groupby(["airport","airport_alert_id"]).ngroups
        print(f"Jury charge : {len(df_jury):,} eclairs - {n_alerts_jury} alertes")

        summary["jury"] = {}
        for version in ("v6", "v7"):
            theta_test = summary[version]["theta_opt"]
            print(f"\n--- {version.upper()} | theta = {theta_test:.2f} (fige depuis test) ---")

            gbs    = joblib.load(MODELS_DIR / f"gbs_{version}_model.pkl")
            scaler = joblib.load(MODELS_DIR / f"gbs_{version}_scaler.pkl")
            if version == "v7":
                le       = joblib.load(MODELS_DIR / "gbs_v7_label_encoder.pkl")
                features = FEATURES_V7
            else:
                le       = None
                features = FEATURES_V6

            df_feat   = build_features(df_jury, le=le)
            df_scored = score_model(df_feat, gbs, scaler, features)
            res       = evaluer_theta(df_scored, theta_test)
            res["status"] = "ok" if res["Risk_pct"] < 2.0 else "FAIL"

            print(f"  Gain     : {res['Gain_h']:.2f} h")
            print(f"  Risk     : {res['Risk_pct']:.4f} %")
            print(f"  Manques  : {res['n_manques']} / {res['n_L3']}")
            print(f"  Actif    : {res['n_actif']} / {res['n_alertes']} "
                  f"alertes ({res['pct_actif']:.0f} %)")
            print(f"  Statut   : {res['status']}")

            summary["jury"][version] = {
                "theta_used": theta_test,
                "Gain_h":     float(res["Gain_h"]),
                "Risk_pct":   float(res["Risk_pct"]),
                "Gain_moy":   float(res["Gain_moy"]),
                "n_manques":  int(res["n_manques"]),
                "n_L3":       int(res["n_L3"]),
                "n_actif":    int(res["n_actif"]),
                "n_alertes":  int(res["n_alertes"]),
                "status":     res["status"],
            }

    with open(RESULTS / "calibration_theta_test_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n\n=== RECAPITULATIF FINAL ===")
    print(json.dumps(summary, indent=2))
    print(f"\nSorties : {RESULTS}")


if __name__ == "__main__":
    main()
