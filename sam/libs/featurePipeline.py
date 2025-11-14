from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris

# Beispiel-Datensatz laden
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline: Skalierung + Modell
pipeline = Pipeline([
    ('scaler', StandardScaler()),       # Schritt 1: Standardisierung
    ('knn', KNeighborsClassifier())     # Schritt 2: KNN-Modell
])

# Parameter-Grid für KNN (innerhalb der Pipeline)
param_grid = {
    'knn__n_neighbors': [3, 5, 7],
    'knn__weights': ['uniform', 'distance'],
    'knn__p': [1, 2],  # Minkowski-Metrik: p=1 (Manhattan), p=2 (Euclid)
    'knn__algorithm': ['auto', 'ball_tree', 'kd_tree']
}

# GridSearchCV mit Pipeline
grid = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)
grid.fit(X_train, y_train)

# Ergebnisse
print("Beste Parameter:", grid.best_params_)
print("Beste CV-Score:", grid.best_score_)
print("Test-Accuracy:", grid.score(X_test, y_test))
