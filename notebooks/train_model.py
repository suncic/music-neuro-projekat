import os
import json
import numpy as np
import tensorflow as tf

from keras import Model
from keras.layers import LSTM, GRU, Dense, Embedding, Input, Concatenate
from keras.optimizers import Adam, SGD, RMSprop

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
    

def build_model(vocabularies, embedding_dim=64, recurrent_type="lstm", lstm_units=128, 
                optimizer="adam", second_layer_units=None, learning_rate=0.001):
    
    network_inputs = {}
    network_outputs = {}
    embedded_inputs = []
    losses = {}
    metrics = {}

    for property_name in MUSIC_EVENT_PROPERTIES:
        vocabulary_size = len(vocabularies[property_name])
        property_input = Input(shape=(None,), name=f"{property_name}_input", dtype="int32")
        property_embedding_layer = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, name=f"{property_name}_embedding")
        property_embedding = property_embedding_layer(property_input)
        network_inputs[f"{property_name}_input"] = property_input
        embedded_inputs.append(property_embedding)

    concatenate_layer = Concatenate(axis=-1, name="combined_embedding")
    combined_embeddings = concatenate_layer(embedded_inputs)

    if recurrent_type == "lstm":
        if second_layer_units is None:
            recurrent_output = LSTM(lstm_units, name="lstm")(combined_embeddings)
        else:
            first_lstm_output = LSTM(lstm_units, return_sequences=True, name="first_lstm")(combined_embeddings)
            recurrent_output = LSTM(second_layer_units, name="second_lstm")(first_lstm_output)
    elif recurrent_type == "gru":
        if second_layer_units is None:
            recurrent_output = GRU(lstm_units, name="gru")(combined_embeddings)
        else:
            first_gru_output = GRU(lstm_units, return_sequences=True, name="first_gru")(combined_embeddings)
            recurrent_output = GRU(second_layer_units, name="second_gru")(first_gru_output)
    else:
        raise ValueError(f"Unknown recurrent type: {recurrent_type}")

    for property_name in MUSIC_EVENT_PROPERTIES:
        vocabulary_size = len(vocabularies[property_name])
        property_output = Dense(vocabulary_size, activation="softmax", name=f"{property_name}_output")(recurrent_output)
        network_outputs[f"{property_name}_output"] = property_output

    if optimizer == 'adam':
        opt = Adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        opt = SGD(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        opt = RMSprop(learning_rate=learning_rate)
    else:
        opt = optimizer

    for property_name in MUSIC_EVENT_PROPERTIES:
        output_name = f"{property_name}_output"
        vocabulary_size = len(vocabularies[property_name])
        losses[output_name] = "sparse_categorical_crossentropy"
        metrics[output_name] = ["accuracy", tf.keras.metrics.SparseTopKCategoricalAccuracy(k=min(3, vocabulary_size), name="top3_accuracy")]

    model = Model(inputs=network_inputs, outputs=network_outputs)
    model.compile(optimizer=opt, loss=losses, metrics=metrics)
    return model

def prepare_inputs_for_model(inputs):
    return {f"{property_name}_input": inputs[property_name] for property_name in MUSIC_EVENT_PROPERTIES}

def prepare_outputs_for_model(outputs):
    return {f"{property_name}_output": outputs[property_name] for property_name in MUSIC_EVENT_PROPERTIES}

def train_and_evaluate(composer_folder, model_save_path, lstm_units=128,
                        optimizer='adam', learning_rate=0.001, second_layer_units=None,
                        epochs=5, batch_size=32, recurrent_type="lstm"):
    
    print(f"Loading data from {composer_folder}")
    inputs, outputs, vocabs = load_prepared_data(composer_folder)

    X_train = prepare_inputs_for_model(inputs["train"])
    y_train = prepare_outputs_for_model(outputs["train"])

    X_validation = prepare_inputs_for_model(inputs["validation"])
    y_validation = prepare_outputs_for_model(outputs["validation"])

    X_test = prepare_inputs_for_model(inputs["test"])
    y_test = prepare_outputs_for_model(outputs["test"])

    print(f"\nBuilding model (lstm_units={lstm_units}, optimizer={optimizer}, lr={learning_rate})")
    model = build_model(
        vocabularies=vocabs, 
        recurrent_type=recurrent_type, 
        second_layer_units=second_layer_units, 
        lstm_units=lstm_units, 
        optimizer=optimizer, 
        learning_rate=learning_rate
    )
    model.summary()

    print(f"\nTraining model ({epochs} epochs, batch_size={batch_size})")
    training_results = model.fit(
        X_train, 
        y_train, 
        epochs=epochs, 
        batch_size=batch_size, 
        validation_data=(X_validation, y_validation)
    )

    print("\nEvaluating model on test set")
    test_results = model.evaluate(X_test, y_test)
    for metric_name, metric_value in test_results.items():
        print(f"{metric_name}: {metric_value:.4f}")

    if model_save_path:
        model_folder = os.path.dirname(model_save_path)

        if model_folder:
            os.makedirs(model_folder, exist_ok=True)

        model.save(model_save_path)
        print(f"Saved to {model_save_path}")
    
    return training_results, test_results

if __name__ == "main":

    sequences_length = [4, 8, 16, 32]
    composers = ["bach", "mozart", "beethoven"]

    for composer in composers:
        for sequence_length in sequences_length:

            train_and_evaluate(
                composer_folder=f"data/prepared/{composer}/seq_{sequence_length}",
                model_save_path=f"models/{composer}_seq_{sequence_length}.keras"
            )