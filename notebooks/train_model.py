import os
import json
import numpy as np
import tensorflow as tf

from keras.models import Sequential
from keras.layers import LSTM, GRU, Dense, Embedding
from keras.optimizers import Adam, SGD, RMSprop

def load_prepared_data(folder):
    X_train = np.load(os.path.join(folder, "X_train.npy"))
    y_train = np.load(os.path.join(folder, "y_train.npy"))

    X_validation = np.load(os.path.join(folder, "X_validation.npy"))
    y_validation = np.load(os.path.join(folder, "y_validation.npy"))

    X_test = np.load(os.path.join(folder, "X_test.npy"))
    y_test = np.load(os.path.join(folder, "y_test.npy"))

    with open(os.path.join(folder, "vocab.json"), "r") as file:
        vocab = json.load(file)

    return X_train, y_train, X_validation, y_validation, X_test, y_test, vocab
    

def build_model(vocab_size, embedding_dim=64, recurrent_type="lstm", lstm_units=128, optimizer='adam', second_layer_units=None, learning_rate=0.001):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim))

    if recurrent_type == "lstm":
        if second_layer_units is None:
            model.add(LSTM(lstm_units))
        else:
            model.add(LSTM(lstm_units, return_sequences=True))
            model.add(LSTM(second_layer_units))
    elif recurrent_type == "gru":
        model.add(GRU(lstm_units))
    else:
        raise ValueError(f"Unknown recurrent type: {recurrent_type}")

    
    model.add(Dense(vocab_size, activation='softmax'))

    if optimizer == 'adam':
        opt = Adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        opt = SGD(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        opt = RMSprop(learning_rate=learning_rate)
    else:
        opt = optimizer

    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, 
                  metrics=['accuracy', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3_accuracy")])
    return model


def train_and_evaluate(composer_folder, model_save_path,lstm_units=128,
                        optimizer='adam', learning_rate=0.001,
                        epochs=5, batch_size=32):
    

    print(f"Loading data from {composer_folder}")
    X_train, y_train, X_validation, y_validation, X_test, y_test, vocab = load_prepared_data(composer_folder)

    vocab_size = len(vocab)
    sequence_length = X_train.shape[1]

    print(f"Vocabulary size: {vocab_size}")
    print(f"Sequence length: {sequence_length}")
    print(f"Total train sequences: {X_train.shape[0]}")
    print(f"Total validation sequences: {X_validation.shape[0]}")
    print(f"Total test sequences: {X_test.shape[0]}")

    print(f"\nBuilding model (lstm_units={lstm_units}, optimizer={optimizer}, lr={learning_rate})")
    model = build_model(vocab_size=vocab_size, lstm_units=lstm_units, optimizer=optimizer, learning_rate=learning_rate)

    print(f"\nTraining model ({epochs} epochs, batch_size={batch_size})")
    training_results = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_validation, y_validation))

    print("\nEvaluating model on test set")
    test_loss, test_accuracy, test_top3_accuracy = model.evaluate(X_test, y_test)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test top3 accuracy: {test_top3_accuracy:.4f}")

    if model_save_path:
        model_folder = os.path.dirname(model_save_path)

        if model_folder:
            os.makedirs(model_folder, exist_ok=True)

        model.save(model_save_path)
        print(f"Saved to {model_save_path}")
    
    return training_results, test_loss, test_accuracy

if __name__ == "__main__":

    sequences_length = [4, 8, 16, 32]
    composers = ["bach", "mozart", "beethoven"]

    for composer in composers:
        for sequence_length in sequences_length:

            train_and_evaluate(
                composer_folder=f"../data/prepared/{composer}/seq_{sequence_length}",
                model_save_path=f"../models/{composer}_seq_{sequence_length}.keras"
            )