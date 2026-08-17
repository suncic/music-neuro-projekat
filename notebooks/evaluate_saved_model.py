import os
import json
import tensorflow as tf

from train_model import load_prepared_data, prepare_inputs_for_model, prepare_outputs_for_model

composer_folder = "data/prepared/bach/seq_16"
model_path = "results/bach/baseline/baseline_seed42/best_epoch_model.keras"
summary_path = "results/bach/baseline/baseline_seed42/experiment_summary.json"

inputs, outputs, vocabs = load_prepared_data(composer_folder)
X_test = prepare_inputs_for_model(inputs["test"])
y_test = prepare_outputs_for_model(outputs["test"])

model = tf.keras.models.load_model(model_path)
test_results = model.evaluate(X_test, y_test, return_dict=True, verbose=1)

with open(summary_path, "r") as f:
    experiment_summary = json.load(f)

experiment_summary["test_results"] = {metric_name: float(metric_value) for metric_name, metric_value in test_results.items()}

with open(summary_path, "w") as f:
    json.dump(experiment_summary, f, indent=4)

print("Test results added to experiment_summary.json")