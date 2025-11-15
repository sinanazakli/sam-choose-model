from sklearn.datasets import load_iris
# from sam.automl.automl_classifier_pro import AutoMLClassifierPro
from sam.automl.automl_classifier_pro_plus import AutoMLClassifierProPlus
from sam.automl_helper.formatter import format_pipeline_info, format_model_list
import pandas as pd

format_model_list(AutoMLClassifierProPlus.model_list)

# # Datensatz laden
iris = load_iris()

# # AutoML-Instanzen erstellen
automl_plus = AutoMLClassifierProPlus()

# # test_data = pd.read_csv("test.csv")
# # test_data = test_data.sample(n=5000, random_state=42)  # z. B. 5000 Zeilen
# # test_data = test_data.sample(frac=0.01, random_state=42)  # 10 % des Datensatzes

# # automl mit Standardwerten
# # df_results, best_model = automl.find(iris)
# df_results, best_model = automl_plus.find(iris, low_mem=True, reduce_to=0.1)
df_results, best_model = automl_plus.find(iris, low_mem=True, reduce=10)

# # Eigene Parameter
# # df_results, best_model = automl.find(iris, test_size=0.3, random_state=123)

# Ergebnisse anzeigen
print("\nErgebnisse:")
print(df_results)
print("\nBestes Modell:")
print(best_model)

print("\nBestes Modell mit formatierter Ausgabe:")

print(format_pipeline_info(best_model))