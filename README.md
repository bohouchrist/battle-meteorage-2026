# ⚡ Battle Météorage 2026 — GBS Survival Analysis

Prédiction de la fin des alertes foudre aéroportuaires par **Gradient Boosting Survival**.

> **Performance jury 2023-2025** · 1 352 alertes · **103,5 h économisées** · risque **0,10 %** (20× sous la limite de 2 %).

---

## 🚀 Démarrage rapide — tester l'app

```powershell
pip install -r app/requirements.txt
streamlit run app/app.py
```

L'app **fonctionne immédiatement** : les modèles entraînés sont déjà dans `models/`, et les données jury 2023-2025 sont dans `segment_alerts_all_airports_eval.csv`.

**Tu n'as RIEN à exécuter avant de lancer l'app.**

---

## 📓 Ordre des notebooks (si tu veux tout regénérer)

| # | Notebook | Rôle | Obligatoire ? |
|---|---|---|---|
| 1 | `notebooks/exploration_donnees.ipynb` | EDA initial (distrib, censure, KM) | optionnel |
| 2 | `notebooks/modelisation_v7.ipynb` | **Entraîne** GBS v7 sur 2016-2020 → produit les `.pkl` | ❌ (déjà fait) |
| 3 | `notebooks/calibration_theta_test.ipynb` | **Calibre θ** sur test 2021-2022 pour v6 et v7 (fonction générique `calibrate(version)`) | ✓ pour valider |
| 4 | `notebooks/evaluation_jury_2023_production.ipynb` | **Évalue** le modèle de production sur jury 2023-2025 (gain, risque, calibration, C-index) | ✓ pour les chiffres finaux |
| 5 | `notebooks/model_production_v7.ipynb` | **Empaquette** le modèle prod (modèle + scaler + LE + θ figé) dans un `.pkl` unique | optionnel |

**Pour les chiffres officiels du rapport**, exécute les notebooks **3 → 4**. L'app n'en dépend pas.

---

## 🏗 Structure du projet

```
.
├── app/
│   ├── app.py                  # ⭐ Application Streamlit (démo live)
│   └── requirements.txt
│
├── notebooks/                  # ⭐ 5 notebooks utiles uniquement
│   ├── exploration_donnees.ipynb
│   ├── modelisation_v7.ipynb
│   ├── calibration_theta_test.ipynb
│   ├── evaluation_jury_2023_production.ipynb
│   ├── model_production_v7.ipynb
│   └── prise_en_main_clean.ipynb
│
├── models/                     # Modèles entraînés (v6 + v7)
│   ├── gbs_v7_model.pkl
│   ├── gbs_v7_scaler.pkl
│   ├── gbs_v7_label_encoder.pkl
│   ├── gbs_v7_metadata.json
│   ├── gbs_v6_model.pkl        # pour comparaison dans calibration
│   ├── gbs_v6_scaler.pkl
│   └── gbs_v6_metadata.json
│
├── data/                       # Données train + test (parquet 2016-2022)
│   ├── data_df.parquet
│   ├── data_df_with_alert.parquet
│   └── data_df_alert.parquet
│
├── results/                    # Sorties des notebooks
│   ├── calibration_theta_test_summary.json
│   ├── calibration_theta_test_v6.csv
│   ├── calibration_theta_test_v7.csv
│   ├── jury_summary_eval_2023_v7.json
│   ├── predictions_eval_2023_v7.csv
│   └── strikes_eval_2023_v7.csv
│
├── scripts/
│   └── calibrate_theta_test.py # Version script de la calibration
│
├── _archive/                   # Tout le legacy (notebooks v5/v6/v8, vieilles presentations, modeles RSF...)
│   ├── old_notebooks/
│   ├── old_models/
│   ├── old_docs/
│   ├── old_presentations/
│   └── old_scripts/
│
├── binome_repo/                # Approche Weibull du binôme (repo séparé)
│
├── segment_alerts_all_airports_eval.csv   # Données jury 2023-2025 (80 186 éclairs)
├── presentation_modele_gbs.tex / .pdf     # Slides de soutenance
├── rapport_gbs.tex / .pdf                 # Rapport long
├── script_oral_gbs.docx                   # Script oral
└── README.md                              # Ce fichier
```

---

## 🎯 Le modèle en deux phrases

À chaque éclair `xi`, le GBS prédit `S(t|xi)` = probabilité qu'aucun nouvel éclair ne survienne dans les `t` prochaines minutes. Dès qu'un éclair atteint `S(30|xi) > θ = 0,30`, on lève l'alerte à `t̂_i = min{t : S(t|xi) ≤ S(30|xi)/0,98}`. La décision globale est `min` sur tous les éclairs confiants — règle conservative.

| Hyperparamètre | Valeur | Rôle |
|---|---|---|
| `n_estimators` | 50 | nombre d'arbres |
| `learning_rate` | 0,15 | pas du boosting |
| `max_depth` | 3 | complexité arbre |
| `subsample` | 0,9 | stochastique |
| `loss` | `coxph` | log-vraisemblance partielle Cox |
| **θ** | **0,30** | calibré sur test 2021-2022 (figé) |
| **C-index test** | **0,742** | écart train/test 0,0001 |

---

## 🎬 Ce que fait l'app

Sidebar : choix **aéroport · saison · alerte · seuil θ · cadence**.

Page principale :
- **Hero section** (titre dégradé, aéroport en carte stylisée)
- **3 big numbers** : confiance courante / gain / heure de levée
- **Bandeau statut** clignotant (🟢 actif / 🟡 en attente)
- **Flash éclair** à chaque nouvel impact
- **Vérification rétrospective** (✓ validée / ✓ safe / ✗ violée / ⊘ inactif)
- **Graphe Plotly dark** 3 panneaux (distance, T*_i vs gap réel, confiance S(30))
- **Tableau résumé** par alerte

Le **slider θ est réactif en direct** : monte-le pendant la simulation et vois les barres T*_i changer de couleur, l'heure de levée se recalculer.

---

## 📚 Références

- Cox, D.R. (1972) — *Regression Models and Life-Tables*
- Friedman, J. (2001) — *Greedy Function Approximation*
- Pölsterl, S. (2020) — *scikit-survival*
- Harrell, F.E. (1982) — *C-index*

---

*Battle Météorage 2026 · CHRIST BOHOU · `bohouchrist34@gmail.com`*
