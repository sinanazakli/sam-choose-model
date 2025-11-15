# 📘 AutoMLClassifierProPlus

## Description

This repository provides an **AutoML class for classification tasks** that automatically trains multiple models, evaluates them, and selects the best one.  
The **AutoMLClassifierProPlus** class supports:

- ✅ Automatic hyperparameter tuning using `RandomizedSearchCV`
- ✅ Scaling for models that require it
- ✅ Model comparison based on **F1 score**, **Accuracy**, and **ROC-AUC**
- ✅ Low-memory option for large datasets
- ✅ Clear results output as a `DataFrame`

## **Use Cases**
- Classification of structured data
- Benchmarking different algorithms
- Quickly find the best model for a dataset

## Features

Supported Models: LogisticRegression, SVC, RandomForest, GradientBoosting, MLPClassifier, and more
Hyperparameter Tuning: Randomized search over defined parameter grids
Cross-Validation: Default StratifiedKFold


# AutoMLClassifierProPlus
## Beschreibung
Dieses Repository bietet eine AutoML-Klasse für Klassifikationsprobleme, die mehrere Modelle automatisch trainiert, bewertet und das beste Modell auswählt.
Die Klasse AutoMLClassifierProPlus unterstützt:

✅ Automatische Hyperparameter-Suche mit RandomizedSearchCV
✅ Skalierung für Modelle, die es benötigen
✅ Vergleich von Modellen anhand von F1-Score, Accuracy und ROC-AUC
✅ Low-Memory-Option für große Datensätze
✅ Übersichtliche Ausgabe der Ergebnisse als DataFrame

Anwendungsfälle

Klassifikation von strukturierten Daten
Benchmarking verschiedener Algorithmen
Schnell das beste Modell für einen Datensatz finden

## Installation

```bash
git clone <your-repo-link>
cd <your-repo>
pip install -r requirements.txt
