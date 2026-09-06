import os
import csv
import json
import time
import random
import numpy as np
import tensorflow as tf

from keras import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import LSTM, GRU, Dense, Embedding, Input, Concatenate
from keras.optimizers import Adam

from prepare_data import MUSIC_EVENT_PROPERTIES

def load_prepared_data(folder):
    dataset_name = ["train", "validation", "test"]
    inputs = {}
    outputs = {}

    for dataset in dataset_name:
        inputs[dataset] = {}
        outputs[dataset] = {}

        for property_name in MUSIC_EVENT_PROPERTIES:
            input_file_name = f"X_{property_name}_{dataset}.npy"
            output_file_name = f"y_{property_name}_{dataset}.npy"

            inputs[dataset][property_name] = np.load(os.path.join(folder, input_file_name))
            outputs[dataset][property_name] = np.load(os.path.join(folder, output_file_name))

    with open(os.path.join(folder, "vocabularies.json"), "r") as file:
        vocabs = json.load(file)

    return inputs, outputs, vocabs
    

def build_model(vocabularies, embedding_dim=64, recurrent_type="lstm", 
                recurrent_units=128, second_layer_units=None, learning_rate=0.001):
    
    network_inputs = {}
    network_outputs = {}
    embedded_inputs = []
    losses = {}
    metrics = {}

    for property_name in MUSIC_EVENT_PROPERTIES:
        vocabulary_size = len(vocabularies[property_name])
        property_input = Input(shape=(None,), name=f"{property_name}_input", dtype="int32")
        property_embedding = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, name=f"{property_name}_embedding")(property_input)
        network_inputs[f"{property_name}_input"] = property_input
        embedded_inputs.append(property_embedding)

    combined_embeddings = Concatenate(axis=-1, name="combined_embedding")(embedded_inputs)

    if recurrent_type == "lstm":
        if second_layer_units is None:
            recurrent_output = LSTM(recurrent_units, name="lstm")(combined_embeddings)
        else:
            first_lstm_output = LSTM(recurrent_units, return_sequences=True, name="first_lstm")(combined_embeddings)
            recurrent_output = LSTM(second_layer_units, name="second_lstm")(first_lstm_output)
    elif recurrent_type == "gru":
        if second_layer_units is None:
            recurrent_output = GRU(recurrent_units, name="gru")(combined_embeddings)
        else:
            first_gru_output = GRU(recurrent_units, return_sequences=True, name="first_gru")(combined_embeddings)
            recurrent_output = GRU(second_layer_units, name="second_gru")(first_gru_output)
    else:
        raise ValueError(f"Nepoznata neuronska rekurentna mreza: {recurrent_type}")

    for property_name in MUSIC_EVENT_PROPERTIES:
        vocabulary_size = len(vocabularies[property_name])
        property_output = Dense(vocabulary_size, activation="softmax", name=f"{property_name}_output")(recurrent_output)
        network_outputs[f"{property_name}_output"] = property_output

        losses[f"{property_name}_output"] = "sparse_categorical_crossentropy"
        metrics[f"{property_name}_output"] = ["accuracy", tf.keras.metrics.SparseTopKCategoricalAccuracy(k=min(3, vocabulary_size), name="top3_accuracy")]

    model = Model(inputs=network_inputs, outputs=network_outputs)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=losses, metrics=metrics)
    return model

def prepare_inputs_for_model(inputs):
    return {f"{property_name}_input": inputs[property_name] for property_name in MUSIC_EVENT_PROPERTIES}

def prepare_outputs_for_model(outputs):
    return {f"{property_name}_output": outputs[property_name] for property_name in MUSIC_EVENT_PROPERTIES}

def save_training_data(training_results, save_path):
    metric_names = list(training_results.history.keys())
    number_of_epoch = len(training_results.epoch)

    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch"] + metric_names)

        for epoch_index in range(number_of_epoch):
            row = [epoch_index + 1]
            for metric in metric_names:
                row.append(float(training_results.history[metric][epoch_index]))

            writer.writerow(row)

def train_and_evaluate(experiment_name, composer_folder, results_folder, lstm_units=128, use_early_stopping=True,
                        learning_rate=0.001, second_layer_units=None, patience=10,
                        epochs=100, batch_size=32, recurrent_type="lstm", embedding_dim=64, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    run_folder = os.path.join(results_folder, experiment_name)
    os.makedirs(run_folder, exist_ok=True)

    print(f"Ucitavanje podataka iz {composer_folder}")
    inputs, outputs, vocabs = load_prepared_data(composer_folder)

    X_train = prepare_inputs_for_model(inputs["train"])
    y_train = prepare_outputs_for_model(outputs["train"])
    X_validation = prepare_inputs_for_model(inputs["validation"])
    y_validation = prepare_outputs_for_model(outputs["validation"])
    X_test = prepare_inputs_for_model(inputs["test"])
    y_test = prepare_outputs_for_model(outputs["test"])

    first_property = MUSIC_EVENT_PROPERTIES[0]
    sequence_length = inputs["train"][first_property].shape[1]
    vocabulary_sizes = {property_name: len(vocabs[property_name]) for property_name in MUSIC_EVENT_PROPERTIES}
    experiment_configuration = {
        "experiment_name": experiment_name,
        "composer_folder": composer_folder,
        "sequence_length": int(sequence_length),
        "train_sequences": int(len(inputs["train"][first_property])),
        "validation_sequences": int(len(inputs["validation"][first_property])),
        "test_sequences": int(len(inputs["test"][first_property])),
        "vocabulary_sizes": vocabulary_sizes,
        "embedding_dim": embedding_dim,
        "recurrent_type": recurrent_type,
        "lstm_units": lstm_units,
        "second_layer_units": second_layer_units,
        "optimizer": "Adam",
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "maximum_epochs": epochs,
        "patience": patience,
        "use_early_stopping": use_early_stopping,
        "seed": seed
    }

    with open(os.path.join(run_folder, "config.json"), "w") as f:
        json.dump(experiment_configuration, f, indent=4)


    print(f"\nGradjenje modela: lstm_units={lstm_units}, optimizer=Adam, lr={learning_rate}")
    model = build_model(
        vocabularies=vocabs, 
        recurrent_type=recurrent_type, 
        second_layer_units=second_layer_units, 
        recurrent_units=lstm_units,
        learning_rate=learning_rate,
        embedding_dim=embedding_dim
    )
    model.summary()

    best_model_path = os.path.join(run_folder, "best_epoch_model.keras")
    callbacks = [ModelCheckpoint(filepath=best_model_path, monitor="val_loss", mode="min", save_best_only=True, verbose=1)]
    if use_early_stopping:
        callbacks.append(EarlyStopping(monitor="val_loss", mode="min", patience=patience, restore_best_weights=True, verbose=1))

    start_time = time.perf_counter()
    print(f"\nPocinje treniranje modela: {epochs} epoha, velicina batch-a {batch_size}")
    training_results = model.fit(
        X_train, 
        y_train, 
        epochs=epochs, 
        batch_size=batch_size, 
        validation_data=(X_validation, y_validation),
        callbacks=callbacks,
        verbose=1,
        shuffle=True
    )
    training_time = time.perf_counter() - start_time
    save_training_data(training_results, os.path.join(run_folder, "train_model.csv"))

    validation_losses = training_results.history["val_loss"]
    best_epoch_index = int(np.argmin(validation_losses))
    experiment_summary = {
        "completed_epochs": len(training_results.epoch),
        "best_epoch": best_epoch_index + 1,
        "best_val_loss": float(validation_losses[best_epoch_index]),
        "train_loss_at_best_epoch": float(training_results.history["loss"][best_epoch_index]),
        "training_time_seconds": float(training_time),
        "average_epoch_time_seconds": float(training_time / len(training_results.epoch)),
        "model_parameters": int(model.count_params())
    }

    for property_name in MUSIC_EVENT_PROPERTIES:
        output_name = f"{property_name}_output"
        metric_names = {
            f"train_{property_name}_accuracy_at_best_epoch": f"{output_name}_accuracy",
            f"validation_{property_name}_accuracy_at_best_epoch": f"val_{output_name}_accuracy",
            f"train_{property_name}_top3_at_best_epoch": f"{output_name}_top3_accuracy",
            f"validation_{property_name}_top3_at_best_epoch": f"val_{output_name}_top3_accuracy"
        }

        for summary_name, history_name in metric_names.items():
            experiment_summary[summary_name] = float(training_results.history[history_name][best_epoch_index])

    print("\nEvaluiranje najboljeg modela na test skupu")
    best_model = tf.keras.models.load_model(best_model_path)
    test_results = best_model.evaluate(X_test, y_test, return_dict=True, verbose=1)
    experiment_summary["test_results"] = {metric_name: float(metric_value) for metric_name, metric_value in test_results.items()}

    with open(os.path.join(run_folder, "experiment_summary.json"), "w") as f:
        json.dump(experiment_summary, f, indent=4)

    print(f"Eksperiment je zavrsen. Rezultati su u {run_folder}")
    return experiment_summary