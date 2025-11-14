import time
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
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
import numpy as np

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
# 3. Cross-Validation Methode
# -----------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# -----------------------------
# 4. Modelle und große Parametergrids
# -----------------------------
models_and_params = {
    "KNN": (KNeighborsClassifier(), {
        'model__n_neighbors': np.arange(1, 30),
        'model__weights': ['uniform', 'distance'],
        'model__p': [1, 2]
    }),
    "LogisticRegression": (LogisticRegression(max_iter=1000), {
        'model__C': np.logspace(-3, 2, 20),
        'model__solver': ['lbfgs', 'saga']
    }),
    "SGDClassifier": (SGDClassifier(max_iter=1000, random_state=42), {
        'model__loss': ['hinge', 'log_loss'],
        'model__alpha': np.logspace(-4, -1, 20)
    }),
    "SVC": (SVC(), {
        'model__C': np.logspace(-2, 2, 20),
        'model__kernel': ['linear', 'rbf'],
        'model__gamma': ['scale', 'auto']
    }),
    "RandomForest": (RandomForestClassifier(random_state=42), {
        'model__n_estimators': np.arange(50, 300, 50),
        'model__max_depth': [None, 5, 10, 15],
        'model__min_samples_split': [2, 5, 10]
    }),
    "GradientBoosting": (GradientBoostingClassifier(random_state=42), {
        'model__n_estimators': np.arange(50, 300, 50),
        'model__learning_rate': np.linspace(0.01, 0.3, 10),
        'model__max_depth': [3, 5, 7]
    }),
    "MLPClassifier": (MLPClassifier(max_iter=500, random_state=42), {
        'model__hidden_layer_sizes': [(50,), (100,), (50,50)],
        'model__activation': ['relu', 'tanh'],
        'model__alpha': np.logspace(-4, -1, 10)
    })
}

# -----------------------------
# 5. Training, Tuning und Bewertung
# -----------------------------
results = []

for name, (model, param_dist) in models_and_params.items():
    print(f"\n=== Modell: {name} ===")
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    
    # RandomizedSearchCV mit 20 zufälligen Kombinationen
    search = RandomizedSearchCV(
        pipeline, param_distributions=param_dist, n_iter=20,
        cv=cv, n_jobs=-1, random_state=42
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
# 6. Zusammenfassung
# -----------------------------
print("\n=== Zusammenfassung ===")
print(f"{'Modell':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10}")
for name, acc, prec, rec in results:
    print(f"{name:<20} {acc:<10.4f} {prec:<10.4f} {rec:<10.4f}")

