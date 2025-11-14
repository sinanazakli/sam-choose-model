import warnings
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, make_scorer
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np

class AutoMLClassifier:
    model_list = [
    {
        "name": "LogisticRegression",
        "model": LogisticRegression(max_iter=1000),
        "category": "Linear",
        "cv_strategy": "StratifiedKFold",
        "best_for": "Lineare Zusammenhänge, große Daten, wenig Features",
        "param_grid": {
            "model__C": np.logspace(-3, 3, 7),
            "model__penalty": ["l2"],
            "model__solver": ["lbfgs", "saga"]
        }
    },
    {
        "name": "RidgeClassifier",
        "model": RidgeClassifier(),
        "category": "Linear",
        "cv_strategy": "StratifiedKFold",
        "best_for": "Hohe Dimensionalität, regularisierte lineare Modelle",
        "param_grid": {
            "model__alpha": np.logspace(-3, 3, 7)
        }
    },
    {
        "name": "SGDClassifier",
        "model": SGDClassifier(max_iter=1000),
        "category": "Linear",
        "cv_strategy": "StratifiedKFold",
        "best_for": "Sehr große Datensätze, Online-Learning",
        "param_grid": {
            "model__loss": ["hinge", "log_loss"],
            "model__alpha": np.logspace(-4, -1, 4),
            "model__penalty": ["l2", "l1"]
        }
    },
    {
        "name": "SVC",
        "model": SVC(),
        "category": "SVM",
        "cv_strategy": "StratifiedKFold",
        "best_for": "Kleine bis mittlere Datensätze, komplexe Grenzen",
        "param_grid": {
            "model__C": np.logspace(-2, 2, 5),
            "model__kernel": ["linear", "rbf"],
            "model__gamma": ["scale", "auto"]
        }
    },
    {
        "name": "LinearSVC",
        "model": LinearSVC(max_iter=2000),
        "category": "SVM",
        "cv_strategy": "StratifiedKFold",
        "best_for": "Große Datensätze, lineare Trennung",
        "param_grid": {
            "model__C": np.logspace(-3, 3, 7),
            "model__penalty": ["l2"]
        }
    },
    {
        "name": "KNeighborsClassifier",
        "model": KNeighborsClassifier(),
        "category": "Instanzbasiert",
        "cv_strategy": "StratifiedKFold",
        "best_for": "Kleine Datensätze, nichtlineare Strukturen",
        "param_grid": {
            "model__n_neighbors": [3, 5, 7, 9],
            "model__weights": ["uniform", "distance"],
            "model__metric": ["minkowski", "euclidean"]
        }
    },
    {
        "name": "GaussianNB",
        "model": GaussianNB(),
        "category": "Naive Bayes",
        "cv_strategy": "StratifiedKFold",
        "best_for": "Textdaten, kontinuierliche Features",
        "param_grid": {}  # kaum Hyperparameter
    },
    {
        "name": "MultinomialNB",
        "model": MultinomialNB(),
        "category": "Naive Bayes",
        "cv_strategy": "StratifiedKFold",
        "best_for": "Textklassifikation (Bag-of-Words, TF-IDF)",
        "param_grid": {
            "model__alpha": np.linspace(0.1, 1.0, 10)
        }
    },
    {
        "name": "BernoulliNB",
        "model": BernoulliNB(),
        "category": "Naive Bayes",
        "cv_strategy": "StratifiedKFold",
        "best_for": "Binäre Features (z. B. Text mit Präsenz/Absenz)",
        "param_grid": {
            "model__alpha": np.linspace(0.1, 1.0, 10)
        }
    },
    {
        "name": "DecisionTreeClassifier",
        "model": DecisionTreeClassifier(),
        "category": "Baumbasiert",
        "cv_strategy": "StratifiedKFold",
        "best_for": "Kleine bis mittlere Datensätze, interpretierbare Modelle",
        "param_grid": {
            "model__max_depth": [None, 5, 10, 20],
            "model__min_samples_split": [2, 5, 10]
        }
    },
    {
        "name": "RandomForestClassifier",
        "model": RandomForestClassifier(),
        "category": "Baumbasiert",
        "cv_strategy": "StratifiedKFold oder KFold",
        "best_for": "Große Datensätze, robuste Performance, viele Features",
        "param_grid": {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 5, 10]
        }
    },
    {
        "name": "ExtraTreesClassifier",
        "model": ExtraTreesClassifier(),
        "category": "Baumbasiert",
        "cv_strategy": "StratifiedKFold oder KFold",
        "best_for": "Große Datensätze, sehr schnelle Berechnung",
        "param_grid": {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [None, 10, 20]
        }
    },
    {
        "name": "GradientBoostingClassifier",
        "model": GradientBoostingClassifier(),
        "category": "Baumbasiert",
        "cv_strategy": "StratifiedKFold oder KFold",
        "best_for": "Mittlere Datensätze, komplexe Muster",
        "param_grid": {
            "model__n_estimators": [50, 100, 200],
            "model__learning_rate": np.linspace(0.01, 0.3, 5),
            "model__max_depth": [3, 5, 7]
        }
    },
    {
        "name": "HistGradientBoostingClassifier",
        "model": HistGradientBoostingClassifier(),
        "category": "Baumbasiert",
        "cv_strategy": "StratifiedKFold oder KFold",
        "best_for": "Sehr große Datensätze, numerische Features",
        "param_grid": {
            "model__max_iter": [100, 200, 300],
            "model__learning_rate": np.linspace(0.01, 0.3, 5),
            "model__max_depth": [None, 10, 20]
        }
    },
    {
        "name": "AdaBoostClassifier",
        "model": AdaBoostClassifier(),
        "category": "Ensemble",
        "cv_strategy": "StratifiedKFold",
        "best_for": "Kleine bis mittlere Datensätze, schwache Basislerner",
        "param_grid": {
            "model__n_estimators": [50, 100, 200],
            "model__learning_rate": np.linspace(0.01, 1.0, 5)
        }
    },
    {
        "name": "BaggingClassifier",
        "model": BaggingClassifier(),
        "category": "Ensemble",
        "cv_strategy": "StratifiedKFold",
        "best_for": "Hohe Varianz-Daten, robuste Modelle",
        "param_grid": {
            "model__n_estimators": [10, 50, 100],
            "model__max_samples": [0.5, 1.0]
        }
    },
    {
        "name": "MLPClassifier",
        "model": MLPClassifier(max_iter=500),
        "category": "Neuronale Netze",
        "cv_strategy": "StratifiedKFold",
        "best_for": "Komplexe nichtlineare Muster, große Datensätze",
        "param_grid": {
            "model__hidden_layer_sizes": [(50,), (100,), (100, 50)],
            "model__activation": ["relu", "tanh"],
            "model__alpha": [0.0001, 0.001, 0.01]
        }
    }
]

    def __init__(self, model_list=model_list, scoring='f1', n_iter=20, random_state=42):
        """
        Initialisiert den AutoML-Workflow.
        """
        self.model_list = model_list
        self.scoring = scoring
        self.n_iter = n_iter
        self.random_state = random_state
        self.results_df = None
        self.best_model = None

    def _automl_classification(self, X_train, y_train, X_test=None, y_test=None):
        """
        Interne Methode: Führt RandomizedSearchCV für alle Modelle aus und berechnet zusätzliche Metriken.
        """
        results = []
        best_score = -np.inf
        scorer = make_scorer(f1_score, average='weighted') if self.scoring == 'f1' else self.scoring

        for m in self.model_list:
            print(f"🔍 Starte Suche für: {m['name']} ...")
            pipeline = Pipeline([('model', m['model'])])

            # CV-Strategie
            if "StratifiedKFold" in m['cv_strategy']:
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            else:
                cv = 5

            # RandomizedSearchCV
            search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=m['param_grid'],
                n_iter=self.n_iter,
                scoring=scorer,
                cv=cv,
                random_state=self.random_state,
                n_jobs=-1
            )

            search.fit(X_train, y_train)
            best_params = search.best_params_
            best_cv_score = search.best_score_

            # Test-Metriken berechnen
            acc, f1, roc_auc = None, None, None
            if X_test is not None and y_test is not None:
                y_pred = search.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')

                # ROC-AUC nur für binäre oder One-vs-Rest
                try:
                    y_proba = search.predict_proba(X_test)
                    if y_proba.shape[1] == 2:  # binär
                        roc_auc = roc_auc_score(y_test, y_proba[:, 1])
                    else:  # multi-class
                        roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
                except:
                    roc_auc = None

            # Ergebnisse speichern
            results.append({
                "Model": m['name'],
                "CV Score": best_cv_score,
                "Accuracy": acc,
                "F1 Score": f1,
                "ROC-AUC": roc_auc,
                "Best Params": best_params
            })

            # Bestes Modell aktualisieren
            if best_cv_score > best_score:
                best_score = best_cv_score
                self.best_model = search.best_estimator_

        # Ergebnisse sortieren
        self.results_df = pd.DataFrame(results).sort_values(by="CV Score", ascending=False)
        print("\n✅ AutoML abgeschlossen!")
        print(self.results_df)

    def print_model_list(self):
        """
        Gibt die Liste der verfügbaren Modelle mit Beschreibungen aus.
        """
        for m in self.model_list:
            print(f"Modell: {m['name']}")
            print(f"  Kategorie: {m['category']}")
            print(f"  Beste Anwendung: {m['best_for']}")
            print(f"  CV-Strategie: {m['cv_strategy']}")
            print("")

    def find(self, data, test_size=0.2, random_state=42):
        """
        Führt den kompletten AutoML-Workflow aus:
        - Daten splitten
        - Modelle optimieren
        - Ergebnisse anzeigen
        """
        # Daten extrahieren
        if isinstance(data, tuple):
            X, y = data
        else:
            X, y = data.data, data.target

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # AutoML starten
        self._automl_classification(X_train, y_train, X_test, y_test)

        print("\n🏆 Bestes Modell:")
        print(self.best_model)

        return self.results_df, self.best_model