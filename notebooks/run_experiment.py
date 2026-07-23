import os
import csv
import json
import time
import random
import numpy as np
import tensorflow as tf

from keras.callbacks import EarlyStopping, ModelCheckpoint
from train_model import load_prepared_data, build_model

def save_training_data(trained_model, save_path):
    metric_names = list(trained_model.history.keys())
    number_of_epoch = len(trained_model.epoch)

    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch"] + metric_names)

        for epoch_index in range(number_of_epoch):
            row = [epoch_index + 1]

            for metric in metric_names:
                row.append(float(trained_model.history[metric][epoch_index]))

            writer.writerow(row)
    

def run_experiment(experiment_name, composer_folder, results_folder, 
                   embedding_dim=64, recurrent_type="lstm", lstm_units=128, 
                   second_layer_units=None, learning_rate=0.001, epochs=100, 
                   batch_size=32, patience=10, use_early_stopping=True, evaluate_test=False, seed=42):

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    run_folder = os.path.join(results_folder, experiment_name)
    os.makedirs(run_folder, exist_ok=True)

    X_train, y_train, X_validation, y_validation, X_test, y_test, vocab = load_prepared_data(composer_folder)

    experiment_configuration = {
        "experiment_name": experiment_name,
        "composer_folder": composer_folder,
        "sequence_length": int(X_train.shape[1]),
        "train_sequences": int(len(X_train)),
        "validation_sequences": int(len(X_validation)),
        "test_sequences": int(len(X_test)),
        "vocab_size": len(vocab),
        "embedding_dim": embedding_dim,
        "recurrent_type": recurrent_type,
        "lstm_units": lstm_units,
        "second_layer_units": second_layer_units,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "maximum_epochs": epochs,
        "patience": patience,
        "use_early_stopping": use_early_stopping,
        "evaluate_test": evaluate_test,
        "seed": seed
    }

    with open(os.path.join(run_folder, "config.json"), "w") as f:
        json.dump(experiment_configuration, f, indent=4)
 
    model = build_model(vocab_size=len(vocab), embedding_dim=embedding_dim, recurrent_type=recurrent_type,
                        lstm_units=lstm_units, second_layer_units=second_layer_units, learning_rate=learning_rate)
    
    best_model_path = os.path.join(run_folder, "best_epoch_model.keras")
    callbacks = [ModelCheckpoint(filepath=best_model_path, monitor="val_loss", mode="min", save_best_only=True, verbose=1)]

    if use_early_stopping:
        callbacks.append(EarlyStopping(monitor="val_loss", mode="min", patience=patience, restore_best_weights=True, verbose=1))

    start_time = time.perf_counter()
    training_model = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), epochs=epochs, 
                            batch_size=batch_size, callbacks=callbacks, verbose=1, shuffle=True) 
    
    training_time = time.perf_counter() - start_time
    save_training_data(training_model, os.path.join(run_folder, "train_model.csv"))

    validation_losses = training_model.history["val_loss"]
    best_epoch_index = int(np.argmin(validation_losses))
    best_epoch = best_epoch_index + 1

    experiment_summary = {
        "completed_epochs": len(training_model.epoch),
        "best_epoch" : best_epoch,
        "best_val_loss": float(validation_losses[best_epoch_index]),
        "val_accuracy_at_best_epoch": float(training_model.history["val_accuracy"][best_epoch_index]),
        "val_top3_accuracy_at_best_epoch": float(training_model.history["val_top3_accuracy"][best_epoch_index]),
        "train_loss_at_best_epoch": float(training_model.history["loss"][best_epoch_index]),
        "train_accuracy_at_best_epoch": float(training_model.history["accuracy"][best_epoch_index]),
        "training_time_seconds": training_time,
        "average_epoch_time_seconds": (training_time / len(training_model.epoch)),
        "model_parameters": int(model.count_params())
    }

    if evaluate_test:
        best_model = tf.keras.models.load_model(best_model_path)
        test_results = best_model.evaluate(X_test, y_test, return_dict=True, verbose=1)

        experiment_summary["test_loss"] = float(test_results["loss"])
        experiment_summary["test_accuracy"] = float(test_results["accuracy"])
        experiment_summary["test_top3_accuracy"] = float(test_results["top3_accuracy"])


    with open(os.path.join(run_folder, "experiment_summary.json"), "w") as f:
        json.dump(experiment_summary, f, indent=4)

    print("Experiment completed.")
    return experiment_summary