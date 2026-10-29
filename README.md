# Robust MLP — comparaison des fonctions de coût

Petit projet personnel en PyTorch pour étudier la sensibilité d'un réseau MLP aux valeurs aberrantes dans une tâche de régression.

## Objectif

Je compare cinq entraînements avec **la même architecture et la même initialisation** :

- `mse` : erreur quadratique ;
- `huber` : perte de Huber ;
- `huber_warmup` : MSE pendant 20 époques, puis Huber ;
- `tukey` : fonction biweight de Tukey ;
- `tukey_warmup` : MSE pendant 20 époques, puis Tukey.

L'idée est de vérifier non seulement la résistance à la contamination, mais aussi la stabilité de l'optimisation. Le warmup est une **hypothèse expérimentale**, pas une garantie d'amélioration.

## Installation

Python 3.10+ recommandé.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Sous Windows : `.venv\\Scripts\\activate`.

## Exécution

```bash
python run_experiment.py --quick   # vérification rapide, résultats non interprétables
python run_experiment.py           # expérience complète
python -m pytest -q               # tests
```

Les réglages sont dans `configs/default.yaml`. Le projet fonctionne sur CPU, CUDA ou MPS (Mac Apple Silicon). Le temps d'exécution dépend de la machine.

## Protocole

1. On génère un signal non linéaire : `sin(2πx) + 0.5x²` avec un bruit gaussien.
2. On crée des ensembles train, validation et test distincts. Seules les **sorties du train** sont contaminées (0 à 40 %).
3. Pour une seed donnée, tous les modèles partagent exactement les mêmes données et les mêmes poids initiaux.
4. L'architecture par défaut comporte **4 couches cachées de 64 neurones ReLU**. Elle est volontairement plus simple que l'ancienne version à 10 couches.
5. L'entraînement utilise Adam, jusqu'à 180 époques, avec patience 45 et au moins 70 époques. Le meilleur checkpoint est choisi sur la **RMSE de validation propre**.
6. Les métriques finales (RMSE, MAE, R²) sont calculées sur le **test propre**, jamais utilisé pour entraîner ou sélectionner les modèles.
7. On répète l'expérience avec **3 seeds** (42, 43 et 44) : compromis raisonnable pour un projet personnel. Ce n'est pas suffisant pour une preuve statistique forte.

**Important :** pour isoler l'effet de la contamination dans ce jeu synthétique, la normalisation des sorties utilise les valeurs du train *avant* contamination. C'est une information oracle disponible uniquement dans cette simulation. Pour des données réelles, il faudra utiliser une estimation robuste de l'échelle sans connaître les sorties propres.

## Résultats : seulement trois figures

```text
outputs/
  config_used.yaml        # configuration exacte de l'expérience
  metrics.csv             # une ligne par entraînement
  summary.csv             # moyennes et écarts-types par modèle / contamination
  01_robustesse.png       # RMSE moyenne ± écart-type
  02_convergence.png      # validation, seed 42, contamination maximale
  03_predictions.png      # prédictions, seed 42, contamination maximale
```

Le champ `best_epoch` désigne l'époque du **meilleur checkpoint**, pas forcément l'époque d'arrêt. `collapsed` est un simple signalement : écart-type des prédictions inférieur à 10 % de celui des sorties test. Ce critère est descriptif et ne démontre pas, à lui seul, un échec d'optimisation.

Les fichiers de sortie sont ignorés par Git. Le script remplace uniquement les six fichiers qu'il génère dans `outputs/`, sans supprimer d'autres fichiers.

## Points à garder en tête

- La perte de Tukey est non convexe : elle peut rester bloquée selon l'initialisation.
- La fonction de Huber n'est pas censée être systématiquement meilleure que la MSE sur chaque réalisation.
- Le warmup peut stabiliser l'apprentissage, mais cela doit être vérifié sur plusieurs seeds.
- Les performances sont évaluées sur un test propre ; elles ne mesurent pas la qualité des prédictions sur un test contaminé.
- La sélection sur validation propre est un choix expérimental favorable : en situation réelle, un ensemble de validation parfaitement propre n'est pas toujours disponible.
- L'entraînement est réalisé par rétropropagation avec Adam, **pas par IRLS**.

## Organisation

```text
configs/default.yaml   paramètres
src/data.py            génération et contamination
src/models.py          architecture MLP
src/losses.py          MSE, Huber et Tukey
src/train.py           entraînement et sélection
src/plots.py           trois graphiques
run_experiment.py      lancement des expériences
tests/test_project.py  tests de base
```

## Pistes pour la suite

Comparer plusieurs valeurs de `huber_delta` et `tukey_c`, tester une contamination asymétrique et ajouter des données réelles. Pour une étude plus approfondie, augmenter le nombre de seeds et quantifier les cas de prédictions quasi constantes.
