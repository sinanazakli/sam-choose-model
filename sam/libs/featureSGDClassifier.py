from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Daten laden
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline: Skalierung + Modell
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('sgd', SGDClassifier(max_iter=1000, random_state=42))
])

# Parameter-Grid für SGDClassifier
param_grid = {
    'sgd__loss': ['hinge', 'log_loss'],  # SVM vs. logistische Regression
    'sgd__alpha': [0.0001, 0.001, 0.01], # Regularisierung
    'sgd__penalty': ['l2', 'l1', 'elasticnet']
}

# GridSearchCV
grid = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)
grid.fit(X_train, y_train)

print("Beste Parameter:", grid.best_params_)
print("Beste CV-Score:", grid.best_score_)
print("Test-Accuracy:", grid.score(X_test, y_test))
