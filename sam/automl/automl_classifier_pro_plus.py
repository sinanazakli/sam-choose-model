import time
import warnings
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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

class AutoMLClassifierProPlus:
    model_list = [
        {
            "name": "LogisticRegression",
            "model": LogisticRegression(max_iter=5000),
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
            "model": SGDClassifier(max_iter=5000),
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
        # {
        #     "name": "LinearSVC",
        #     "model": LinearSVC(max_iter=8000),
        #     "category": "SVM",
        #     "cv_strategy": "StratifiedKFold",
        #     "best_for": "Große Datensätze, lineare Trennung",
        #     "param_grid": {
        #         "model__C": np.logspace(-3, 3, 7),
        #         "model__penalty": ["l2"]
        #     }
        # },
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
            "model": MLPClassifier(max_iter=2000, early_stopping=True),
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

    def __init__(self, model_list=model_list, scoring='f1', n_iter=20, random_state=42, suppress_warnings=True):
        """
        AutoML-Klasse mit Low-Memory-Option, Zeitmessung und flexibler Datensatzverarbeitung.
        """
        self.model_list = model_list
        self.scoring = scoring
        self.n_iter = n_iter
        self.random_state = random_state
        self.results_df = None
        self.best_model = None

        if suppress_warnings:
            warnings.filterwarnings("ignore")

    def _needs_scaling(self, model_name):
        scale_models = ['LogisticRegression', 'RidgeClassifier', 'SGDClassifier', 'SVC', 'LinearSVC', 'MLPClassifier']
        return model_name in scale_models

    def _automl_classification(self, X_train, y_train, X_test=None, y_test=None):
        results = []
        best_score = -np.inf
        scorer = make_scorer(f1_score, average='weighted') if self.scoring == 'f1' else self.scoring

        total_start = time.time()
        for m in self.model_list:
            print(f"\n🔍 Starte Suche für: {m['name']} ...")
            start_time = time.time()

            steps = []
            if self._needs_scaling(m['name']):
                steps.append(('scaler', StandardScaler()))
            steps.append(('model', m['model']))
            pipeline = Pipeline(steps)

            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)

            param_space_size = np.prod([len(v) if hasattr(v, '__len__') else 1 for v in m['param_grid'].values()])
            n_iter = min(self.n_iter, param_space_size)

            search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=m['param_grid'],
                n_iter=n_iter,
                scoring=scorer,
                cv=cv,
                random_state=self.random_state,
                n_jobs=2,  # Begrenzung für Stabilität
                pre_dispatch=2
            )

            search.fit(X_train, y_train)
            best_params = search.best_params_
            best_cv_score = search.best_score_

            acc, f1, roc_auc = None, None, None
            if X_test is not None and y_test is not None:
                y_pred = search.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')
                try:
                    y_proba = search.predict_proba(X_test)
                    if y_proba.shape[1] == 2:
                        roc_auc = roc_auc_score(y_test, y_proba[:, 1])
                    else:
                        roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
                except:
                    roc_auc = None

            results.append({
                "Model": m['name'],
                "CV Score": best_cv_score,
                "Accuracy": acc,
                "F1 Score": f1,
                "ROC-AUC": roc_auc,
                "Best Params": best_params
            })

            if best_cv_score > best_score:
                best_score = best_cv_score
                self.best_model = search.best_estimator_

            elapsed = time.time() - start_time
            mins, secs = divmod(elapsed, 60)
            print(f"⏱ Dauer für {m['name']}: {int(mins)}m {secs:.2f}s")

        total_elapsed = time.time() - total_start
        t_mins, t_secs = divmod(total_elapsed, 60)
        print(f"\n✅ AutoML abgeschlossen! Gesamtdauer: {int(t_mins)}m {t_secs:.2f}s")

        self.results_df = pd.DataFrame(results).sort_values(by="CV Score", ascending=False)
        print(self.results_df)

    def find(self, data, test_size=0.2, random_state=42, target_column=None, low_mem=False, reduce_to=None, reduce=None):
        """
        Führt den AutoML-Workflow aus.
        Unterstützt sklearn-Datasets und Pandas DataFrames.
        Low-Memory-Option: reduce_to (Prozent), reduce (Anzahl Zeilen).
        """
        if hasattr(data, 'data') and hasattr(data, 'target'):
            X, y = data.data, data.target
        elif isinstance(data, pd.DataFrame):
            if target_column is None:
                target_column = data.columns[-1]

            if len(data) > 100000:
                print(f"⚠️ Warnung: Datensatz hat {len(data)} Zeilen. Low-Memory empfohlen!")

            if low_mem:
                if reduce_to is not None:
                    frac = min(max(reduce_to, 0.0), 1.0)
                    print(f"⚠️ Low-Memory aktiv: Reduziere Datensatz auf {frac*100:.1f}%")
                    data = data.sample(frac=frac, random_state=random_state)
                elif reduce is not None:
                    reduce = min(reduce, len(data))
                    print(f"⚠️ Low-Memory aktiv: Reduziere Datensatz auf {reduce} Zeilen")
                    data = data.sample(n=reduce, random_state=random_state)

            X = data.drop(columns=[target_column])
            y = data[target_column]
        else:
            raise ValueError("Unterstützte Formate: sklearn-Dataset oder Pandas DataFrame")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # self._automl_classification(X_train.values, y_train.values, X_test.values, y_test.values)

        self._automl_classification(
            np.array(X_train),
            np.array(y_train),
            np.array(X_test),
            np.array(y_test)
        )

        print("\n🏆 Bestes Modell:")
        print(self.best_model)

        return self.results_df, self.best_model


# Utility-Funktion für formatierte Pipeline-Ausgabe
def format_pipeline_info(pipeline):
    scaler = pipeline.named_steps.get('scaler', None)
    model = pipeline.named_steps.get('model', None)

    scaler_info = f"Scaler: {scaler.__class__.__name__}" if scaler else "Scaler: Kein Scaler"
    model_info = f"Modell: {model.__class__.__name__}"

    params = {k: v for k, v in model.get_params().items() if not k.startswith('_')}
    param_str = ", ".join([f"{k}={v}" for k, v in params.items() if k in ['kernel', 'C', 'max_iter', 'learning_rate', 'hidden_layer_sizes']])

    if param_str:
        model_info += f" ({param_str})"

    return f"{scaler_info}\n{model_info}"