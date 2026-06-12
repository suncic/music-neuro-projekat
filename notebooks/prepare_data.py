import os
import json
import numpy as np

def load_notes(parsed_folder):
    pieces = []
    files = [f for f in os.listdir(parsed_folder) if f.endswith('.json')]

    for file_name in files:
        path = os.path.join(parsed_folder, file_name)
        with open(path, 'r') as f:
            notes = json.load(f)
            pieces.append(notes)

    return pieces

def prepare_sequences(pieces, sequence_length=16):
    all_notes = []
    for piece in pieces:
        all_notes.extend(piece)

    unique_notes = sorted(set(all_notes))
    note_to_int = {note: number for number, note in enumerate(unique_notes)}

    print(f"Total notes (all pieces): {len(all_notes)}")
    print(f"Unique notes: {len(unique_notes)}")

    network_input = []
    network_output = []

    for piece in pieces:
        for i in range(len(piece) - sequence_length):
            sequence_in = piece[i: i + sequence_length]
            sequence_out = piece[i + sequence_length]

            network_input.append([note_to_int[n] for n in sequence_in])
            network_output.append([note_to_int[sequence_out]])

            print(f"Number of sequences: {len(network_input)}")


    return np.array(network_input), np.array(network_output), note_to_int

def save_prepared_data(X, y, vocab, output_folder):
    np.save(os.path.join(output_folder, 'X.npy'), X)
    np.save(os.path.join(output_folder, 'y.npy'), y)

    with open(os.path.join(output_folder, 'vocab.json'), 'w') as f:
        json.dump(vocab, f)

    print(f"Saved to {output_folder}")

print("===== Bach =====")
bach_pieces = load_notes('../data/parsed/bach')
X_bach, y_bach, vocab_bach = prepare_sequences(bach_pieces)
save_prepared_data(X_bach, y_bach, vocab_bach, '../data/prepared/bach')

print("===== Mozart =====")
mozart_pieces = load_notes('../data/parsed/mozart')
X_mozart, y_mozart, vocab_mozart = prepare_sequences(mozart_pieces)
save_prepared_data(X_mozart, y_mozart, vocab_mozart, '../data/prepared/mozart')

print("===== Beethoven =====")
beethoven_pieces = load_notes('../data/parsed/beethoven')
X_beethoven, y_beethoven, vocab_beethoven = prepare_sequences(beethoven_pieces)
save_prepared_data(X_beethoven, y_beethoven, vocab_beethoven, '../data/prepared/beethoven')