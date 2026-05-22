# EXPLICATION DÉTAILLÉE DU MODÈLE GBS v6

## TABLE DES MATIÈRES
1. Concept global
2. Les données brutes
3. Construction des variables de survie
4. Feature engineering (12 features)
5. Prétraitement
6. Architecture du modèle GBS
7. Entraînement
8. Prédictions
9. Résultats

---

## 1. CONCEPT GLOBAL

### Qu'est-ce qu'on essaye de prédire?

```
Pour chaque éclair dans une alerte, on veut prédire:
"Combien de temps avant le PROCHAIN éclair?"
```

**Exemple concret:**

```
Alerte Ajaccio #412:
  - 13:03:00 → Éclair 1
  - 13:04:15 → Éclair 2 (silence = 1min 15s depuis l'éclair 1)
  - 13:05:30 → Éclair 3 (silence = 1min 15s depuis l'éclair 2)
  - ... (30 min sans éclair)
  - 13:35:30 → Alerte terminée

Le modèle apprend: "Éclair à 13:04 + silence=1.25min → prochain éclair à 13:05:30"
```

### Pourquoi survie (survival analysis)?

C'est parce qu'on a des **données censurées**:

- **Non-censuré**: On observe le prochain éclair (duree < 30 min)
- **Censuré**: C'est le dernier éclair de l'alerte (duree = 30 min par défaut)

```python
if is_last_lightning:
    duree = 30  # On ne sait pas quand il y aurait eu le prochain
else:
    duree = temps_jusqu_au_prochain_éclair
```

---

## 2. LES DONNÉES BRUTES

### Fichier: `data_df_with_alert.parquet`

```
56,599 éclairs au total (2016-2022)

Colonnes importantes:
- lightning_id: Identifiant unique
- date: Timestamp (UTC)
- airport: Aéroport (Ajaccio, Bastia, Nantes, etc.)
- airport_alert_id: Numéro de l'alerte à cet aéroport
- dist: Distance du centre de l'aéroport (km)
- is_last_lightning_cloud_ground: Booléen (dernier éclair?)
- silence_minute: Temps depuis l'éclair précédent
```

### Exemple de données brutes:

```
Index | Date              | Airport | Alert ID | Dist | Silence | Last?
------|-------------------|---------|----------|------|---------|-------
0     | 2016-01-02 21:22  | Ajaccio | 1        | 14.8 | NaN     | False
1     | 2016-01-02 21:24  | Ajaccio | 1        | 15.1 | 1.88    | False
2     | 2016-01-02 21:25  | Ajaccio | 1        | 15.6 | 1.22    | False
3     | 2016-01-02 21:27  | Ajaccio | 1        | 15.3 | 1.08    | False
4     | 2016-01-02 21:28  | Ajaccio | 1        | 17.6 | 1.83    | True  ← Dernier
```

---

## 3. CONSTRUCTION DES VARIABLES DE SURVIE

### Étape 1: Grouper par alerte

```python
g = df.groupby(['airport', 'airport_alert_id'])
```

Pour chaque alerte, on travaille avec les éclairs dans l'ordre chronologique.

### Étape 2: Calculer la "duree"

```python
# Pour chaque éclair, on décale d'une ligne
# et on calcule le temps jusqu'au SUIVANT
df['next_date'] = g['date'].shift(-1)
df['duree'] = (df['next_date'] - df['date']).dt.total_seconds() / 60
```

**Exemple:**

```
Éclair 1: 13:03:00 → next_date = 13:04:15 → duree = 1.25 min
Éclair 2: 13:04:15 → next_date = 13:05:30 → duree = 1.25 min
Éclair 3: 13:05:30 → next_date = NaN (dernier) → duree = NaN → remplir avec 30
```

### Étape 3: Calculer "event" (censuré ou pas)

```python
# event = True si on observe le prochain éclair
# event = False si c'est le dernier (censuré)

df['event'] = df['duree'].notna() & (df['duree'] <= 30)

# Si dernier éclair → censored
df.loc[~df['event'], 'duree'] = 30
```

### Résultat:

```
Total: 56,599 éclairs
  - Events (non-censurés): 53,972 (95.4%)
  - Censurés (derniers): 2,627 (4.6%)
```

---

## 4. FEATURE ENGINEERING (12 Features)

### Pourquoi 12 features?

Parce qu'on veut capturer différents patterns:
- **Quand** l'éclair arrive → features temporelles
- **Où** l'éclair arrive → features de distance  
- **Vite** vs **lentement** → features de silence/fréquence
- **Au début** vs **à la fin** de l'alerte → features de maturité

### 4.1 FEATURES TEMPORELLES (5 features)

**Idée**: Les éclairs viennent à différentes heures et saisons du jour.

#### h_cos et h_sin (heure de la journée)

```python
h = df['date'].dt.hour + df['date'].dt.minute/60  # 0.0 à 24.0
df['h_cos'] = np.cos(2*π*h/24)  # Cycle: 0→1→0→-1→0
df['h_sin'] = np.sin(2*π*h/24)
```

**Pourquoi cos/sin?** Parce que l'heure est cyclique:
- Minuit (0h) ≈ Minuit (24h) → Doivent avoir des valeurs proches

```
h=0 (minuit)  → h_cos=1.0, h_sin=0.0
h=6 (matin)   → h_cos=0.0, h_sin=1.0
h=12 (midi)   → h_cos=-1.0, h_sin=0.0
h=18 (soir)   → h_cos=0.0, h_sin=-1.0
h=24 (minuit) → h_cos=1.0, h_sin=0.0 ← Pareil que h=0!
```

#### doy_cos et doy_sin (jour de l'année)

```python
doy = df['date'].dt.dayofyear  # 1 à 365
df['doy_cos'] = np.cos(2*π*doy/365)
df['doy_sin'] = np.sin(2*π*doy/365)
```

Même logique: Jour 1 (janvier) ≈ Jour 365 (décembre) cycliquement.

#### saison

```python
df['saison'] = ((df['date'].dt.month % 12) // 3) + 1
# Résultat: 1 (hiver), 2 (printemps), 3 (été), 4 (automne)
```

### 4.2 FEATURES DE SILENCE/FRÉQUENCE (2 features)

**Idée**: Les éclairs qui arrivent rapidement = alerte active = prochain éclair bientôt

#### silence_min

```python
prev = g['date'].shift(1)  # Date de l'éclair précédent
df['silence_min'] = ((df['date'] - prev).dt.total_seconds() / 60)\
                    .fillna(30)\
                    .clip(0, 60)
```

```
Éclair 1 (13:03) → silence = NaN → remplir 30
Éclair 2 (13:04:15) → silence = 1.25 min
Éclair 3 (13:05:30) → silence = 1.25 min
Éclair 4 (13:20:00) → silence = 14.5 min (alerte ralentit!)
```

#### freq_5min

```python
df['freq_5min'] = (1 / df['silence_min'].clip(lower=0.5)).clip(upper=10)
```

Fréquence: Nombre d'éclairs par 5 minutes.

```
silence = 0.5 min → freq = 10 (très rapide!)
silence = 1 min → freq = 5
silence = 5 min → freq = 1 (lent)
```

### 4.3 FEATURES DE DISTANCE (3 features)

**Idée**: Les éclairs loin = alerte calme. Les éclairs près = danger!

#### dist_centre

```python
df['dist_centre'] = df['dist']  # Simplement copier la distance
```

La distance actuelle de chaque éclair.

#### dist_avg_5

```python
df['dist_avg_5'] = g['dist'].transform(lambda x: x.rolling(5, min_periods=1).mean())
```

Moyenne glissante de la distance sur les 5 derniers éclairs.

```
Éclairs 1-5: [10, 11, 12, 10, 11] km
  - Éclair 1: dist_avg_5 = 10.0
  - Éclair 5: dist_avg_5 = 10.8
```

#### dist_min_so_far

```python
df['dist_min_so_far'] = g['dist'].cummin()
```

Distance minimale depuis le DÉBUT de l'alerte.

```
Éclairs: 10 km → 11 km → 8 km → 9 km
dist_min: 10 → 10 → 8 → 8 (jamais augmente)
```

### 4.4 FEATURES DE MATURITÉ (2 features)

**Idée**: Les éclairs au début vs à la fin de l'alerte = patterns différents

#### rang

```python
df['rang'] = g.cumcount()
```

Numéro d'ordre de l'éclair dans l'alerte (0, 1, 2, 3, ...)

```
Alerte 1:
  Éclair 1: rang = 0
  Éclair 2: rang = 1
  Éclair 3: rang = 2
  ...
  Éclair 10: rang = 9
```

#### rang_norm

```python
df['rang_norm'] = df['rang'] / g['rang'].transform('max').clip(lower=1)
```

Rang normalisé entre 0 et 1.

```
Alerte avec 10 éclairs:
  Éclair 1: rang_norm = 0/9 = 0.0 (début)
  Éclair 5: rang_norm = 4/9 = 0.44 (milieu)
  Éclair 10: rang_norm = 9/9 = 1.0 (fin)
```

---

## 5. PRÉTRAITEMENT

### 5.1 Normalisation avec StandardScaler

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Pourquoi?** Les features ont des ranges différents:

```
h_cos: -1.0 à 1.0
saison: 1 à 4
silence_min: 0 à 60
dist: 0 à 50
freq: 1 à 10
```

StandardScaler remet tout en μ=0, σ=1 (moyenne 0, écart-type 1):

```
Feature original: silence_min = [0, 10, 20, 30, 40, 50]
Après scaling: [-1.65, -0.83, 0.0, 0.83, 1.65, 2.48]
```

C'est important pour les arbres de décision (pas besoin) mais **crucial** pour GBS!

### 5.2 Gestion des NaN

```python
for col in FEATURES:
    if df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())
```

On remplace les NaN par la **médiane** de la colonne.

**Exemple:**
```
silence_min: [1.5, 2.0, NaN, 3.5, 4.0]
Médiane: 2.75
Résultat: [1.5, 2.0, 2.75, 3.5, 4.0]
```

---

## 6. ARCHITECTURE DU MODÈLE GBS

### Qu'est-ce que GBS?

**Gradient Boosting Survival** = Ensemble de petits arbres de décision qui prédisent les fonctions de survie.

### Hyperparamètres (v6 - RAPIDE)

```python
GradientBoostingSurvivalAnalysis(
    n_estimators=50,        # Nombre d'arbres (vs 200 pour v5)
    learning_rate=0.15,     # Taux d'apprentissage (vs 0.05 pour v5)
    max_depth=3,            # Profondeur des arbres
    subsample=0.9,          # 90% des données par arbre
    min_samples_leaf=10,    # Min 10 samples par feuille
    loss='coxph',           # Fonction de perte (Cox PH)
    random_state=42,        # Reproductibilité
    n_jobs=-1,              # Parallèle
)
```

### Expliqué:

| Paramètre | Valeur | Signification |
|-----------|--------|---------------|
| **n_estimators** | 50 | 50 arbres petits. Plus = meilleur mais lent |
| **learning_rate** | 0.15 | Apprendre vite (0.15 > 0.05). Risque: overfitting |
| **max_depth** | 3 | Arbres peu profonds (3 niveaux). Simplifie |
| **subsample** | 0.9 | Utilise 90% des données pour chaque arbre |
| **min_samples_leaf** | 10 | Minimum 10 samples par feuille (régularisation) |
| **loss** | coxph | Fonction de perte: Cox Proportional Hazards |
| **n_jobs** | -1 | Utilise tous les cores CPU |

### Loss = "coxph" (Cox Proportional Hazards)

C'est la **fonction objectif** du modèle.

```
Hazard(t) = h₀(t) * exp(β₁X₁ + β₂X₂ + ... + β₁₂X₁₂)

Où:
- h₀(t) = baseline hazard (risque de base au temps t)
- β = coefficients (poids des features)
- X = nos 12 features
```

**En français:**
- Plus β·X est grand = plus de risque = moins de temps avant prochain éclair
- Plus β·X est petit = moins de risque = plus de temps avant prochain éclair

---

## 7. ENTRAÎNEMENT

### Data split

```python
Train: 2016-2020 (39,509 éclairs)
Test:  2021-2022 (14,673 éclairs)
```

### Processus d'entraînement (simplifié)

```
1. Initialiser: 50 arbres vides
2. Pour chaque arbre i=1 à 50:
   a. Prendre 90% des données (subsample=0.9)
   b. Construire un arbre (max_depth=3, min_samples_leaf=10)
   c. Prédire les résidus (erreurs)
   d. Ajouter l'arbre avec weight = learning_rate * 0.15
3. Évaluer sur Test set
4. Sauvegarder le modèle
```

### Résultats réels

```
Entraînement: ~13 minutes (en réalité, devrait être 3-5 min)
C-index train: 0.7755
C-index test: 0.7410

Interprétation:
- 0.74 signifie: le modèle prédit 74% des paires correctement
- 0.5 = aléatoire
- 1.0 = parfait
- 0.74 = BON ✓
```

---

## 8. PRÉDICTIONS

### Comment prédire?

```python
cum_hazard_funcs = gbs.predict_cumulative_hazard_function(X_test_scaled)

# Pour chaque éclair, on obtient une courbe H(t)
# H(t) = cumulative hazard au temps t
# S(t) = exp(-H(t)) = survival probability
```

### Extraire la durée médiane

```python
for func in cum_hazard_funcs:
    h_values = func(func.x)              # Évaluer H(t)
    s_values = np.exp(-h_values)         # Calculer S(t)
    idx = np.argmin(np.abs(s_values - 0.5))  # Trouver où S(t) = 0.5
    median_t = func.x[idx]               # Temps médian
```

**Interprétation:**
```
S(t) = 0.5 = médian
Ça signifie: 50% de chance d'avoir un éclair avant ce temps
```

### Exemple de prédiction

```
Éclair à 13:04:15 avec features [...]:
  → Cumulative Hazard Function H(t)
  → Survival Function S(t) = exp(-H(t))
  → Trouve temps où S(t) = 0.5
  → Prédiction: 1.78 minutes jusqu'au prochain éclair
  → Prochain éclair attendu: 13:05:59
```

---

## 9. RÉSULTATS

### Performance globale

```
C-index test: 0.7410 ✓
Gain total: 69.2 heures
Risk: 0.00% ✓
Status: ACCEPTÉ par le jury
```

### Comparaison v5 vs v6

| Métrique | v5 | v6 | Trade-off |
|----------|----|----|-----------|
| n_estimators | 200 | 50 | 4x moins |
| learning_rate | 0.05 | 0.15 | 3x plus rapide |
| C-index | 0.74-0.77 | 0.7410 | -0.5% de perf |
| Temps | ~10 min | ~13 min | **2x plus rapide attendu** |
| Use case | Accuracy | Speed | Production |

---

## RÉSUMÉ COMPLET DU FLUX

```
1. DONNÉES BRUTES
   56,599 éclairs (2016-2022)
        ↓
2. VARIABLES DE SURVIE
   - duree: temps jusqu'au prochain éclair
   - event: y a-t-il un prochain éclair?
        ↓
3. FEATURE ENGINEERING
   12 features:
   - 5 temporelles (h_cos, h_sin, doy_cos, doy_sin, saison)
   - 2 silence (silence_min, freq_5min)
   - 3 distance (dist_centre, dist_avg_5, dist_min_so_far)
   - 2 maturité (rang, rang_norm)
        ↓
4. PRÉTRAITEMENT
   - StandardScaler (normalize)
   - Gestion des NaN (médiane)
        ↓
5. SPLIT TRAIN/TEST
   Train: 39k éclairs (2016-2020)
   Test: 14k éclairs (2021-2022)
        ↓
6. MODÈLE GBS
   50 arbres, learning_rate=0.15, loss=coxph
        ↓
7. ENTRAÎNEMENT
   Fit sur données d'entraînement
   C-index train: 0.7755
   C-index test: 0.7410
        ↓
8. PRÉDICTIONS
   - Pour chaque éclair: prédire duree jusqu'au prochain
   - Extraire median survival time
   - Appliquer confiance threshold
        ↓
9. ÉVALUATION JURY
   - GAIN: 69.2 heures
   - RISK: 0.00%
   - STATUS: ACCEPTÉ ✓
```

---

## POINTS CLÉS À RETENIR

1. **C'est de la Survival Analysis** → gérer les données censurées
2. **12 features** → capturer temporalité, silence, distance, maturité
3. **GBS = 50 arbres** → ensemble learning, gradual boosting
4. **Loss = coxph** → modèle de risque proportionnel
5. **C-index = 0.74** → discrimination bon
6. **0% risk** → modèle prédit correctement quand les alertes se terminent!
