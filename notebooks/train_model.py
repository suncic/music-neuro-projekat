import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding
from keras.optimizers import Adam, SGD, RMSprop

def load_prepared_data(folder):
    X = np.load(os.path.join(folder, 'X.npy'))
    y = np.load(os.path.join(folder, 'y.npy'))

    with open(os.path.join(folder, 'vocab.json'), 'r') as f:
        vocab = json.load(f)

    return X, y, vocab

def build_model(vocab_size, sequence_length, lstm_units=128, optimizer='adam', learning_rate=0.001):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=64))
    model.add(LSTM(128))
    model.add(Dense(vocab_size, activation='softmax'))

    if optimizer == 'adam':
        opt = Adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        opt = SGD(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        opt = RMSprop(learning_rate=learning_rate)
    else:
        opt = optimizer

    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

def train_and_evaluate(composer_folder, model_save_path,
                        sequence_length=16, lstm_units=128,
                        optimizer='adam', learning_rate=0.001,
                        epochs=5, batch_size=32):
    

    print(f"Loading data from {composer_folder}...")
    X, y, vocab = load_prepared_data(composer_folder)

    vocab_size = len(vocab)
    sequence_length = X.shape[1]

    print(f"Vocabulary size: {vocab_size}")
    print(f"Sequence length: {sequence_length}")
    print(f"Total sequences: {X.shape[0]}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    print(f"Train sequences: {X_train.shape[0]}")
    print(f"Test sequences: {X_test.shape[0]}")

    print(f"\nBuilding model (lstm_units={lstm_units}, optimizer={optimizer}, lr={learning_rate})...")
    model = build_model(vocab_size, X.shape[1], lstm_units, optimizer, learning_rate)

    print(f"\nTraining model ({epochs} epochs, batch_size={batch_size})...")
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1)

    print("\nEvaluating on test set...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    if model_save_path:
        model.save(model_save_path)
        print(f"Saved to {model_save_path}")
    
    return history, test_loss, test_accuracy

if __name__ == "__main__":
    # Default run is Bach
    train_and_evaluate(
        composer_folder='../data/prepared/bach',
        model_save_path='../models/bach_model.keras'
    )