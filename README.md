# ⚡ Battle Météorage 2026 — GBS Survival Analysis

Modèle **Gradient Boosting Survival** pour prédire la fin des alertes foudre dans les aéroports et raccourcir l'immobilisation des pistes.

> **Résultats jury 2023-2025** · 1 352 alertes · **103,5 h économisées** · risque 0,10 % (20× sous la limite de 2 %).

## Démarrage rapide

```powershell
# 1. Installer les dépendances
pip install -r app/requirements.txt

# 2. Lancer la démo Streamlit
streamlit run app/app.py
```

Ouvre `http://localhost:8501`. L'application expose 4 onglets :
1. **Performance globale** — métriques sur validation 2021-2022 + jury 2023-2025
2. **Évaluation jury 2023** — comparaison v6/v7 sur données jury indépendantes
3. **Analyse d'un orage** — décomposition strike-par-strike d'une alerte
4. **🎬 Démo en direct** — simulation chronologique d'une alerte, éclair par éclair

## Le modèle en deux phrases

À chaque éclair `xi`, le GBS prédit `S(t|xi)` = probabilité qu'aucun nouvel éclair ne survienne dans les `t` prochaines minutes. Dès qu'un éclair atteint `S(30|xi) > θ = 0,30`, on lève l'alerte à `t̂_i = min{t : S(t|xi) ≤ S(30|xi)/0,98}`. La décision globale est `min` sur tous les éclairs confiants — règle conservative.

| Paramètre | Valeur | Rôle |
|---|---|---|
| `n_estimators` | 50 | nombre d'arbres |
| `learning_rate` | 0,15 | pas du boosting |
| `max_depth` | 3 | complexité arbre |
| `subsample` | 0,9 | stochastique (Friedman 2002) |
| `loss` | `coxph` | log-vraisemblance partielle Cox |
| **C-index test** | **0,742** | écart train/test 0,0001 |

## Structure

```
.
├── app/                       # Application Streamlit
│   └── app.py                 # 4 onglets dont la démo live
├── notebooks/                 # Notebooks d'entraînement et d'évaluation
│   ├── modelisation_v7.ipynb         # entraînement GBS v7 (modèle final)
│   ├── evaluation_eval_2023_v7.ipynb # évaluation jury 2023-2025
│   ├── exploration_donnees.ipynb     # EDA initial
│   └── comparison_global_vs_stratified.ipynb
├── models/                    # Modèles entraînés (gitignored)
│   ├── gbs_v7_model.pkl
│   ├── gbs_v7_scaler.pkl
│   ├── gbs_v7_label_encoder.pkl
│   └── gbs_v7_metadata.json
├── results/                   # Sorties d'évaluation
│   ├── jury_summary_eval_2023_v7.json
│   ├── predictions_eval_2023_v7.csv
│   └── strikes_eval_2023_v7.csv
├── data/                      # Données (gitignored — > 100 Mo)
├── presentation_modele_gbs.tex / .pdf  # Slides de soutenance
├── rapport_gbs.tex / .pdf              # Rapport long
├── script_oral_gbs.docx                # Script oral
└── binome_repo/               # Approche Weibull du binôme (référence)
```

## Les 13 features

| Famille | Variables | Signal |
|---|---|---|
| Temporelles cycliques (5) | `h_cos`, `h_sin`, `doy_cos`, `doy_sin`, `saison` | heure, saison |
| Cadence (2) | `silence_min`, `freq_5min` | orage qui s'éteint |
| Distance (3) | `dist_centre`, `dist_avg_5`, `dist_min_so_far` | orage qui s'éloigne |
| Contexte (3) | `rang`, `rang_norm`, `airport_enc` | profil aéroport |

`LabelEncoder` pour l'aéroport est fitté **uniquement sur le train** (pas de fuite sur jury).

## Données

| Jeu | Période | Éclairs | Censure |
|---|---|---|---|
| Train | 2016-2020 | 41 509 | 4,6 % |
| Test  | 2021-2022 | 15 090 | 4,4 % |
| Jury  | 2023-2025 | 80 186 | — |

## Démo en direct (onglet 4)

L'onglet **🎬 Démo en direct** simule l'arrivée chronologique des éclairs (2,5 s par défaut) pour visualiser le mécanisme de décision en temps réel :

- **Bandeau de statut** — 🟡 modèle en attente / 🟢 modèle actif avec heure de levée
- **Bandeau flash** — chaque nouvel éclair signalé (heure, distance)
- **Vérification de la prédiction précédente** — pour chaque éclair, on confronte le `T*_i` prédit au délai réel observé, en tenant compte de la **zone de sécurité 3 km** : un éclair rapide mais à 15 km n'est PAS une violation
- **3 panneaux Plotly** : distance, T*_i prédit vs délai réel (barres vs ◆), confiance S(30)

## Références

- Cox, D.R. (1972) — *Regression Models and Life-Tables*
- Friedman, J. (2001) — *Greedy Function Approximation*
- Pölsterl, S. (2020) — *scikit-survival*
- Harrell, F.E. (1982) — *C-index*

---

*Battle Météorage 2026 · Christ Bohou · `bohouchrist34@gmail.com`*
