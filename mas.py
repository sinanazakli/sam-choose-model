from sklearn.datasets import load_iris
from sam.automl.automl_classifier_pro import AutoMLClassifierPro
from sam.automl.automl_classifier_pro_plus import AutoMLClassifierProPlus
from sam.helper.formatter import format_pipeline_info

# Datensatz laden
iris = load_iris()

# AutoML-Instanzen erstellen
automl = AutoMLClassifierPro()
automl_plus = AutoMLClassifierProPlus()

# Standardwerte
# df_results, best_model = automl.find(iris)
df_results, best_model = automl_plus.find(iris)

# Eigene Parameter
# df_results, best_model = automl.find(iris, test_size=0.3, random_state=123)


print("\nErgebnisse:")
print(df_results)
print("\nBestes Modell:")
print(best_model)

print("\nBestes Modell mit formatierter Ausgabe:")

print(format_pipeline_info(best_model))