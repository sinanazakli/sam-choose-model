from sklearn.datasets import load_iris
# from sam.automl.automl_classifier_pro import AutoMLClassifierPro
from sam.automl.automl_classifier_pro_plus import AutoMLClassifierProPlus
from sam.automl_helper.formatter import format_pipeline_info, format_model_list
import pandas as pd

format_model_list(AutoMLClassifierProPlus.model_list)

# load dataset example 
# load here whatever dataset you want to use
iris = load_iris()

# create automl object to use the AutoML functionality
automl_plus = AutoMLClassifierProPlus()

# if you have big data for example you can sample it down like this
# test_data = pd.read_csv("test.csv")
# test_data = test_data.sample(n=5000, random_state=42)  # e.g. 5000 rows
# test_data = test_data.sample(frac=0.01, random_state=42)  # e.g. 10 % of the data

# automl mit Standardwerten
# df_results, best_model = automl.find(iris)
# df_results, best_model = automl_plus.find(iris, low_mem=True, reduce_to=0.1)
df_results, best_model = automl_plus.find(iris, low_mem=True, reduce=10)

# use test_size and random_state for your own needs
# df_results, best_model = automl.find(iris, test_size=0.3, random_state=123)

# print results
print("\nResults:")
print(df_results)
print("\nBest Modell:")
print(best_model)

# print best model with formatted output
print("\nBest Model with formatet output:")
print(format_pipeline_info(best_model))