import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from sklearn.datasets import load_iris
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, LeaveOneOut, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# Warnungen unterdrücken
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# -----------------------------
# 1. Datensatz laden
# -----------------------------
X, y = load_iris(return_X_y=True)

# -----------------------------
# 2. Train/Test-Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# 3. Modelle + Parameter + passende CV-Strategie
# -----------------------------
models_and_params = {
    "KNN": (
        KNeighborsClassifier(),
        {
            'model__n_neighbors': np.arange(1, 30),
            'model__weights': ['uniform', 'distance'],
            'model__p': [1, 2]
        },
        StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    ),
    "LogisticRegression": (
        LogisticRegression(max_iter=1000),
        {
            'model__C': np.logspace(-3, 2, 20),
            'model__solver': ['lbfgs', 'saga']
        },
        StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    ),
    "SGDClassifier": (
        SGDClassifier(max_iter=1000, random_state=42),
        {
            'model__loss': ['hinge', 'log_loss'],
            'model__alpha': np.logspace(-4, -1, 20)
        },
        StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    ),
    "SVC": (
        SVC(),
        {
            'model__C': np.logspace(-2, 2, 20),
            'model__kernel': ['linear', 'rbf'],
            'model__gamma': ['scale', 'auto']
        },
        StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    ),
    "DecisionTree": (
        DecisionTreeClassifier(),
        {
            'model__max_depth': [None, 3, 5, 10],
            'model__criterion': ['gini', 'entropy']
        },
        KFold(n_splits=5, shuffle=True, random_state=42)
    ),
    "RandomForest": (
        RandomForestClassifier(random_state=42),
        {
            'model__n_estimators': np.arange(50, 300, 50),
            'model__max_depth': [None, 5, 10, 15],
            'model__min_samples_split': [2, 5, 10]
        },
        KFold(n_splits=5, shuffle=True, random_state=42)
    ),
    "GradientBoosting": (
        GradientBoostingClassifier(random_state=42),
        {
            'model__n_estimators': np.arange(50, 300, 50),
            'model__learning_rate': np.linspace(0.01, 0.3, 10),
            'model__max_depth': [3, 5, 7]
        },
        KFold(n_splits=5, shuffle=True, random_state=42)
    ),
    "GaussianNB": (
        GaussianNB(),
        {},
        LeaveOneOut()
    ),
    "MLPClassifier": (
        MLPClassifier(max_iter=2000, early_stopping=True, random_state=42),
        {
            'model__hidden_layer_sizes': [(50,), (100,), (50,50)],
            'model__activation': ['relu', 'tanh'],
            'model__alpha': np.logspace(-4, -1, 10),
            'model__learning_rate_init': [0.001, 0.01]
        },
        StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    )
}

# -----------------------------
# 4. Training, Tuning und Bewertung
# -----------------------------
results = []

for name, (model, param_dist, cv_strategy) in models_and_params.items():
    print(f"\n=== Modell: {name} ===")
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    
    search = RandomizedSearchCV(
        pipeline, param_distributions=param_dist, n_iter=20,
        cv=cv_strategy, n_jobs=-1, random_state=42
    )
    
    start_time = time.time()
    search.fit(X_train, y_train)
    elapsed = time.time() - start_time
    
    print("Beste Parameter:", search.best_params_)
    print(f"Beste CV-Score: {search.best_score_:.4f}")
    print(f"Laufzeit: {elapsed:.2f} Sekunden")
    
    y_pred = search.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')
    
    print(f"Test-Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    
    results.append((name, acc, prec, rec))

# -----------------------------
# 5. Ergebnisse als CSV speichern
# -----------------------------
df = pd.DataFrame(results, columns=["Modell", "Accuracy", "Precision", "Recall"])
df.to_csv("model_results.csv", index=False)
print("\nErgebnisse wurden in 'model_results.csv' gespeichert.")

# -----------------------------
# 6. Grafische Auswertung + PNG speichern
# -----------------------------
results_sorted = sorted(results, key=lambda x: x[1], reverse=True)
model_names = [r[0] for r in results_sorted]
accuracies = [r[1] for r in results_sorted]
precisions = [r[2] for r in results_sorted]
recalls = [r[3] for r in results_sorted]

x = np.arange(len(model_names))
width = 0.25

plt.figure(figsize=(12, 6))
plt.bar(x - width, accuracies, width, label='Accuracy', color='skyblue')
plt.bar(x, precisions, width, label='Precision', color='lightgreen')
plt.bar(x + width, recalls, width, label='Recall', color='salmon')

plt.xticks(x, model_names, rotation=30)
plt.ylabel('Score')
plt.title('Vergleich der Modelle: Accuracy, Precision, Recall')
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.savefig("model_comparison.png")
plt.show()

print("\nGrafik wurde als 'model_comparison.png' gespeichert.")