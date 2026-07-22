import os
import json
import numpy as np
from sklearn.model_selection import train_test_split

def load_notes(parsed_folder):
    pieces = []
    files = sorted(f for f in os.listdir(parsed_folder) if f.endswith('.json'))

    for file_name in files:
        path = os.path.join(parsed_folder, file_name)
        with open(path, 'r') as f:
            notes = json.load(f)
            pieces.append(notes)

    return pieces

def split_pieces(pieces, test_size=0.1, validation_size=0.1, random_state=42):
    train_val_pieces, test_pieces = train_test_split(pieces, test_size=test_size, random_state=random_state)
    validation_ratio = validation_size / (1 - test_size)
    train_pieces, validation_pieces = train_test_split(train_val_pieces, test_size=validation_ratio, random_state=random_state)

    return train_pieces, validation_pieces, test_pieces

def create_vocabulary(pieces):
    all_notes = []

    for piece in pieces:
        all_notes.extend(piece)

    unique_notes = sorted(set(all_notes))

    return{
        note: number
        for number, note in enumerate(unique_notes)
    }

def prepare_sequences(pieces, note_to_int, sequence_length=16):
    network_input = []
    network_output = []

    for piece in pieces:
        for i in range(len(piece) - sequence_length):
            sequence_in = piece[i: i + sequence_length]
            sequence_out = piece[i + sequence_length]

            if any(note not in note_to_int for note in sequence_in):
                continue

            if sequence_out not in note_to_int:
                continue

            network_input.append([note_to_int[n] for n in sequence_in])
            network_output.append(note_to_int[sequence_out])

    
    print(f"Number of sequences: {len(network_input)}")

    return np.array(network_input), np.array(network_output)

def prepare_composer(parsed_folder, output_folder, sequence_length=16):
    pieces = load_notes(parsed_folder)
    vocab = create_vocabulary(pieces)
    train_pieces, validation_pieces, test_pieces = split_pieces(pieces)

    X_train, y_train = prepare_sequences(train_pieces, vocab, sequence_length)
    X_validation, y_validation = prepare_sequences(validation_pieces, vocab, sequence_length)
    X_test, y_test = prepare_sequences(test_pieces, vocab, sequence_length)

    os.makedirs(output_folder, exist_ok=True)

    np.save(os.path.join(output_folder, 'X_train.npy'), X_train)
    np.save(os.path.join(output_folder, 'y_train.npy'), y_train)

    np.save(os.path.join(output_folder, 'X_validation.npy'), X_validation)
    np.save(os.path.join(output_folder, 'y_validation.npy'), y_validation)

    np.save(os.path.join(output_folder, 'X_test.npy'), X_test)
    np.save(os.path.join(output_folder, 'y_test.npy'), y_test)

    with open(os.path.join(output_folder, 'vocab.json'), 'w') as f:
        json.dump(vocab, f)

    print(f"Train compositions: {len(train_pieces)}")
    print(f"Validation compositions: {len(validation_pieces)}")
    print(f"Test compositions: {len(test_pieces)}")

    print(f"Train sequences: {len(X_train)}")
    print(f"Validation sequences: {len(X_validation)}")
    print(f"Test sequences: {len(X_test)}")


print("===== Bach =====")
prepare_composer('../data/parsed/bach', '../data/prepared/bach')

print("===== Mozart =====")
prepare_composer('../data/parsed/mozart', '../data/prepared/mozart')

print("===== Beethoven =====")
prepare_composer('../data/parsed/beethoven', '../data/prepared/beethoven')