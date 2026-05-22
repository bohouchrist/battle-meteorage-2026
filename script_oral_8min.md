# Script oral — Présentation GBS Battle Météorage 2026
**Durée cible : 8 minutes** (~1 150 mots, rythme calme)

---

## Slide 1 — Titre *(15 s)*

> Bonjour. Je vais vous présenter notre approche pour la Battle Météorage 2026 :
> un modèle de **Gradient Boosting Survival** pour prédire la fin des alertes
> foudre dans les aéroports.

## Slide 2 — Plan *(15 s)*

> Je vais d'abord poser le problème, puis introduire l'analyse de survie,
> vous montrer le cheminement qui nous a menés au modèle GBS, et terminer
> par les résultats sur les données du jury.

---

## Slide 3 — Le problème *(45 s)*

> Aujourd'hui, Météorage applique une règle fixe : toute alerte est levée
> **30 minutes après le dernier éclair nuage-sol** dans un rayon de 20 km.
> Chaque nouvel éclair remet le compteur à zéro.
>
> Cette règle est **sûre mais coûteuse** : elle bloque des aéroports
> entiers pendant des heures, alors que beaucoup d'orages sont déjà finis.
>
> Notre objectif : **prédire la fin de l'orage avant les 30 minutes**,
> tout en garantissant que le risque qu'un éclair dangereux survienne
> après la levée reste **inférieur à 2 %**.
> Sur ce schéma, le gain c'est l'écart entre $t_{\text{pred}}$ et $t_{\text{règle}}$.

---

## Slide 4 — Analyse de survie *(45 s)*

> Pour ça, on modélise le **temps de survie** $T$, c'est-à-dire la durée
> jusqu'au prochain éclair. C'est exactement un problème de survie
> **avec censure à droite** : si l'alerte se ferme avant un nouvel éclair,
> on sait juste que $T$ dépasse 30 minutes.
>
> L'objet central, c'est la fonction de survie $S(t) = P(T>t)$.
> Notre indicateur clé : $S(30)$ — la probabilité qu'aucun éclair ne
> survienne dans les 30 prochaines minutes.
> Et $T^\star$, c'est le premier instant où $S(t)$ croise un seuil $\theta$
> qu'on calibre.

---

## Slide 5 — Les données *(30 s)*

> Nos données : une ligne par éclair CG, sur 5 aéroports — Ajaccio, Bastia,
> Biarritz, Nantes, Pise. **41 500 éclairs en train**, 15 000 en test,
> et un jeu jury de 80 000 éclairs sur 2023-2025 jamais vus.
> Les colonnes clés sont la date, la distance, l'amplitude et l'ID d'alerte.

---

## Slide 6 — Étape 1 : distribution *(40 s)*

> Avant de modéliser, on explore. Trois constats forts.
>
> **Un :** énorme hétérogénéité entre aéroports. Pise concentre 13 500 éclairs,
> Nantes seulement 2 600. Il faut absolument une feature aéroport.
>
> **Deux :** saisonnalité massive — plus de 60 % des éclairs en juillet-août.
>
> **Trois :** pic horaire net entre 12h et 15h UTC, typique de la convection
> de l'après-midi.
>
> Donc on encode l'aéroport, et on utilise des features cycliques sinus/cosinus
> pour l'heure et le jour de l'année.

---

## Slide 7 — Étape 2 : variable cible *(30 s)*

> On construit ensuite la variable cible $T_i$ : pour chaque éclair, le délai
> jusqu'au suivant dans la même alerte. Censuré à 30 minutes sinon.
>
> Résultat marquant : **95 % d'événements observés, seulement 4,6 % censurés.**
> Autrement dit, la grande majorité des éclairs sont suivis très vite par un
> autre. La censure est faible mais bien présente.

---

## Slide 8 — Étape 3 : KM global *(50 s)*

> Premier modèle naturel : Kaplan-Meier global, **une seule courbe pour tout
> le monde**.
>
> Vous voyez ici la courbe estimée. À 2 % de risque, $T^\star$ vaut 28,3 minutes,
> soit un gain de **1,7 minute seulement** sur la règle des 30 minutes.
> Quasi-rien.
>
> Le problème : KM est une **moyenne**. Il ne distingue pas un orage finissant
> d'un orage actif. Pour un même temps écoulé, il donne la même probabilité à
> tous les éclairs.
>
> **C'est ce qui motive le passage à un modèle conditionnel** : il faut
> $S(t \mid x_i)$, propre à chaque éclair.

---

## Slide 9 — Étape 4 : KM par strate *(30 s)*

> Et la variabilité est bien réelle. Si on stratifie KM par aéroport
> ou par saison, les courbes diffèrent fortement. Donc un modèle qui
> exploite ces variables va capter du signal. C'est exactement ce que fait
> le Gradient Boosting Survival.

---

## Slide 10 — Features *(30 s)*

> On a construit **13 features** réparties en 4 familles :
> les temporelles cycliques, la cadence et fréquence des éclairs,
> la distance à l'aéroport — instantanée, moyenne et minimale —
> et le rang dans l'alerte.

---

## Slide 11 — Le modèle GBS *(45 s)*

> Le GBS, c'est un **ensemble d'arbres de décision boostés** où chaque arbre
> corrige le résidu de la log-vraisemblance de Cox du précédent.
> Au final, on obtient une fonction de survie individualisée, sans imposer
> de forme paramétrique.
>
> Les hyperparamètres ont été optimisés par GridSearch : 50 arbres,
> learning rate 0,15, profondeur 3.
>
> Le mécanisme de décision : pour chaque éclair on calcule la confiance,
> $S(30 \mid x_i)$. Si elle dépasse le seuil $\theta$, on calcule l'horizon
> $T^\star$ individuel, et la fin d'alerte globale c'est le **minimum** sur
> tous les éclairs confiants. Très conservateur.

---

## Slide 12 — Validation *(25 s)*

> En validation sur 2021-2022, on obtient un **C-index de 0,742** en test
> contre 0,776 en train. L'écart est très faible : pas de surapprentissage.
> Et 0,742, c'est largement au-dessus de l'aléatoire à 0,5.

---

## Slide 13 — Résultats jury *(45 s)*

> Sur le jeu jury 2023-2025, **complètement indépendant** : 1 352 alertes
> évaluées.
>
> Le modèle économise **103,5 heures** d'immobilisation aéroportuaire,
> pour un risque de **0,10 %** d'éclair dangereux. C'est **20 fois en dessous
> de la limite de 2 %** fixée par le jury.
> Seulement 2 éclairs manqués sur près de 2 000.

---

## Slide 14 — Comparaison *(30 s)*

> Comparaison rapide avec l'approche Weibull AFT de l'autre binôme.
> Le Weibull est paramétrique, maximise le gain brut — environ 330 heures.
> Le GBS est non-paramétrique, plus prudent : moins de gain mais
> **6 fois moins de risque**. Les deux approches sont complémentaires.

---

## Slide 15 — Conclusion *(30 s)*

> En résumé, le GBS apporte trois choses : une fonction de survie
> individuelle, 103 heures économisées, et un risque très en dessous
> de la limite.
>
> En perspective : un modèle hybride GBS + Weibull, et l'intégration de la
> direction de déplacement de l'orage qu'aucun des deux ne capture.

---

## Slide 16 — Formules *(30 s)*

> Pour les curieux, la formalisation : on est sur un modèle de Cox à risques
> proportionnels. Le boosting minimise la log-vraisemblance partielle par
> descente de gradient sur des arbres. La survie est estimée via l'estimateur
> de Breslow pour le risque de base.

---

## Slide 17 — Algorithme *(25 s)*

> Et voici l'algorithme complet en pseudocode : pour chaque nouvel éclair,
> on calcule les features, on prédit $S(t \mid x_i)$, on évalue la confiance,
> et on décide. La décision globale est le minimum des horizons individuels
> — garantie de prudence.

---

## Slide 18 — Références *(15 s)*

> Les références principales : Cox 1972, Kaplan-Meier 1958, Friedman 2001
> pour le boosting, et l'implémentation scikit-survival de Pölsterl 2020.

---

## Slide 19 — Merci *(5 s)*

> Merci pour votre attention, je suis prêt à répondre à vos questions.

---

## Récapitulatif des temps

| Section | Slides | Temps |
|---|---|---|
| Intro | 1-2 | 30 s |
| Problème + survie | 3-4 | 1 min 30 |
| Données + exploration | 5-9 | 3 min |
| Modèle + validation | 10-12 | 1 min 40 |
| Résultats + conclusion | 13-15 | 1 min 45 |
| Annexes (algo, refs) | 16-18 | 1 min 10 |
| Merci | 19 | 5 s |
| **TOTAL** | **19** | **~8 min** |

## Conseils de débit

- Parler **calmement**, ne pas accélérer sur les slides 6-9 (le cœur du raisonnement)
- Pointer la courbe sur slide 8 (le 1,7 min de KM doit choquer)
- Sur slide 13, **insister sur le « 20 fois sous la limite »**
- Si dépassement, raccourcir slide 16 (formules) — visuel suffit
