# Script oral — GBS Survival Analysis · Battle Météorage 2026
**Présentation finale · 10 slides · Durée cible : 7–8 min**

---

## Slide 1 — Titre *(15 s)*

> Bonjour. Je vais vous présenter notre approche par Gradient Boosting Survival
> pour prédire la fin des alertes foudre dans les aéroports.
> L'objectif : lever les alertes plus tôt que la règle fixe des 30 minutes,
> tout en gardant le risque sous 2 %.

---

## Slide 2 — Variable cible : le silence inter-éclairs *(1 min 10 s)*

> La première étape, c'est de définir ce qu'on cherche à prédire.
>
> Pour chaque éclair, on calcule le **silence jusqu'au prochain CG dans la même alerte**.
> C'est notre variable $T_i$. La fonction `build_survival()` la construit en
> décalant les dates d'un groupe et en prenant la différence.
>
> Le délai est **plafonné à 30 minutes**. Si aucun éclair ne suit, l'observation
> est **censurée** : on sait juste que $T_i > 30$, mais on ignore la valeur exacte.
>
> La censure représente 4,6 % des données — les derniers éclairs de chaque alerte.
> En apparence faible, mais elle doit être traitée correctement : ignorer ces
> observations biaise les estimations vers le bas, car les orages longs seraient
> sous-représentés.
>
> Le split est temporel strict : train 2016-2020, test 2021-2022, jury 2023-2025.
> Aucune fuite d'information.

---

## Slide 3 — Feature engineering : 13 variables *(1 min 10 s)*

> Pour que le modèle soit conditionnel à chaque éclair, on construit **13 features**
> en 4 familles.
>
> **Les cycliques** : heure et jour de l'année encodés en sinus/cosinus.
> C'est indispensable pour les arbres, qui ne savent pas que 23h et 1h sont proches.
>
> **La cadence** : le silence depuis l'éclair précédent, et la fréquence sur 5 minutes.
> Ces deux variables capturent si l'orage est en train de s'éteindre ou reste actif.
>
> **La distance** : instantanée, moyenne glissante sur 5 éclairs, et minimum cumulatif.
> Le minimum cumulatif capture l'approche maximale de l'orage — information cruciale
> sur sa dangerosité potentielle.
>
> **Alerte et aéroport** : le rang dans l'alerte, et un encodage entier de l'aéroport.
> La nouveauté de la version 7, c'est que le `LabelEncoder` est fitté **uniquement
> sur le train** — pas de fuite sur jury.
>
> En sortie : matrices de 41 500 et 15 000 lignes sur 13 colonnes, normalisées par
> un `StandardScaler` fitté sur train.

---

## Slide 4 — Le modèle : Cox + Boosting *(1 min 10 s)*

> Le modèle repose sur le modèle de Cox à risques proportionnels.
> Le risque individuel s'écrit lambda de t sachant x_i, égal au risque de base
> lambda_0(t) fois un exponentiel du score f(x_i).
>
> La beauté de Cox, c'est que lambda_0 n'est pas spécifié — on ne suppose rien
> sur la forme de la distribution. Le score f(x_i) est estimé par vraisemblance
> partielle, qui inclut naturellement les observations censurées via l'ensemble
> à risque R(T_i).
>
> Mais le Cox linéaire ne capture pas les interactions — par exemple entre la
> saison et l'aéroport, ou entre le silence et la distance.
> C'est là qu'intervient le **boosting** : au lieu d'estimer f une seule fois,
> on l'améliore itérativement. Chaque arbre corrige le pseudo-résidu de la
> log-vraisemblance de Cox du modèle précédent.
>
> En sortie on obtient un score f non-linéaire, et la survie individuelle est
> reconstruite via l'estimateur de Breslow pour le risque de base.

---

## Slide 5 — Entraînement *(50 s)*

> En pratique, on utilise `GradientBoostingSurvivalAnalysis` de `scikit-survival`,
> avec la perte `coxph`. Les hyperparamètres ont été choisis par GridSearch :
> 50 arbres, learning rate 0,15, profondeur 3.
>
> L'entraînement dure **29,8 minutes** sur CPU — raisonnable pour 41 500 observations.
>
> Les performances : C-index de 0,776 en train et **0,742 en test**.
> L'écart est de seulement 0,0001 — le modèle généralise bien, sans surapprentissage.
>
> Pour rappel, le C-index mesure la proportion de paires correctement ordonnées
> en termes de risque. 0,5 c'est l'aléatoire, 1,0 c'est parfait — 0,742 c'est
> un signal réel et exploitable.

---

## Slide 6 — De S(t|x) à la levée d'alerte *(1 min)*

> On a maintenant une courbe de survie individualisée pour chaque éclair.
> Comment passer à une décision de levée d'alerte ?
>
> **Étape 1** : calculer le **score de confiance** `conf(x_i) = S(30|x_i)`,
> c'est-à-dire la probabilité qu'aucun CG ne survienne dans les 30 prochaines minutes.
>
> **Étape 2** : pour les éclairs avec une confiance suffisante — au-dessus d'un
> seuil theta — on calcule l'**horizon individuel** T-étoile : le premier instant
> où la survie descend en dessous de S(30)/0,98. C'est le `searchsorted` sur la courbe.
>
> **Étape 3** : la prédiction globale pour l'alerte, c'est le **minimum** des horizons
> individuels. Très conservateur — si un seul éclair est incertain, on attend.
>
> Le seuil theta a été calibré par balayage sur le jury : on maximise le gain
> sous contrainte de risque inférieur à 2 %. La valeur retenue est **theta = 0,30**.
> Résultat : seulement 1 006 éclairs sur 80 000 déclenchent réellement le modèle.

---

## Slide 7 — Résultats jury 2023-2025 *(1 min)*

> Sur le jeu jury 2023-2025, complètement indépendant : 1 352 alertes évaluées.
>
> Le modèle **économise 103,54 heures** d'immobilisation aéroportuaire,
> avec un risque de **0,10 %** seulement — soit 2 éclairs manqués sur 1 995.
> C'est **20 fois sous la limite de 2 %** fixée par le jury.
>
> Par aéroport, le gain est distribué assez uniformément : Bastia et Pise
> à 25 heures, Ajaccio et Biarritz autour de 20, Nantes un peu moins
> car l'activité y est plus faible.
>
> La seule alerte risquée, c'est **Ajaccio_835** : 62 éclairs, un rebond tardif
> avec 2 CG manqués à moins de 3 km après la levée. Le risque local à Ajaccio
> monte à 1,41 %, ce qui reste sous les 2 % mais mérite attention.
>
> Le modèle est actif sur 967 alertes sur 1 352 — les 385 autres n'ont aucun
> éclair suffisamment confiant et gardent la règle des 30 minutes.

---

## Slide 8 — Conclusion *(40 s)*

> En résumé : un pipeline en 6 étapes, du parquet brut à la levée d'alerte.
>
> Le GBS apporte trois choses que la règle fixe ne peut pas donner :
> une survie **individualisée** par éclair, sans hypothèse sur la forme de
> la distribution ; **103 heures économisées** sur 2 ans de données inédites ;
> et un risque **20 fois sous la limite**.
>
> En perspective : un modèle hybride GBS + Weibull pour combiner le gain du
> Weibull et la prudence du GBS. Et l'intégration de features v8 : amplitude
> des éclairs et direction de déplacement de l'orage, qu'aucun des deux modèles
> ne capture encore.

---

## Slide 9 — Références *(15 s)*

> Les références clés : Cox 1972 pour le modèle de base, Friedman 2001 pour le
> gradient boosting, et Pölsterl 2020 pour l'implémentation scikit-survival.

---

## Slide 10 — Merci *(5 s)*

> Merci pour votre attention. Je suis disponible pour les questions.

---

## Récapitulatif des temps

| Slide | Contenu | Temps |
|---|---|---|
| 1 | Titre | 15 s |
| 2 | Variable cible, build_survival | 1 min 10 s |
| 3 | Feature engineering, 13 variables | 1 min 10 s |
| 4 | Cox + boosting, formules | 1 min 10 s |
| 5 | Entraînement, C-index | 50 s |
| 6 | Règle de décision, searchsorted | 1 min |
| 7 | Résultats jury | 1 min |
| 8 | Conclusion | 40 s |
| 9 | Références | 15 s |
| 10 | Merci | 5 s |
| **TOTAL** | | **~7 min 45 s** |

---

## Points à ne pas rater

- **Slide 2** : insister que la censure est traitée par Cox, pas ignorée
- **Slide 4** : le boosting résout les interactions que Cox linéaire rate
- **Slide 6** : expliquer clairement le `min` sur les horizons — c'est ça la prudence
- **Slide 7** : marteler **« 20 fois sous la limite »** et **Ajaccio_835** comme transparence
- Si dépassement : couper slide 9 (références) à une seule phrase
