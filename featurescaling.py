from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_iris

# Beispiel-Datensatz laden
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 1. StandardScaler ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 2. MinMaxScaler (Alternative) ---
minmax = MinMaxScaler()
X_train_mm = minmax.fit_transform(X_train)
X_test_mm = minmax.transform(X_test)

# --- 3. Cross-Validation mit KNN ---
knn = KNeighborsClassifier()
scores_unscaled = cross_val_score(knn, X_train, y_train, cv=5)
scores_scaled = cross_val_score(knn, X_train_scaled, y_train, cv=5)

print("KNN ohne Skalierung:", scores_unscaled.mean())
print("KNN mit StandardScaler:", scores_scaled.mean())

# --- 4. SGDClassifier (logistische Regression) ---
sgd = SGDClassifier(loss='log_loss', max_iter=1000, random_state=42)
scores_sgd_scaled = cross_val_score(sgd, X_train_scaled, y_train, cv=5)
print("SGD mit Skalierung:", scores_sgd_scaled.mean())

# --- 5. GridSearchCV für KNN ---
param_grid = {
    'n_neighbors': [3, 5, 7],
    'weights': ['uniform', 'distance'],
    'p': [1, 2],  # Minkowski-Metrik: p=1 (Manhattan), p=2 (Euclid)
    'algorithm': ['auto', 'ball_tree', 'kd_tree']
}

grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, n_jobs=-1)
grid.fit(X_train_scaled, y_train)

print("Beste Parameter:", grid.best_params_)
print("Beste CV-Score:", grid.best_score_)
print("Test-Accuracy:", grid.score(X_test_scaled, y_test))
