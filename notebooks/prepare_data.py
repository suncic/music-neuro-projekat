import os
import json
import numpy as np
from sklearn.model_selection import train_test_split

MUSIC_EVENT_PROPERTIES = [
    "type",
    "pitch",
    "duration",
    "time_shift",
    "tempo"
]

def load_compositions(parsed_folder):
    pieces = []
    files = sorted(f for f in os.listdir(parsed_folder) if f.endswith('.json'))

    for file_name in files:
        path = os.path.join(parsed_folder, file_name)
        with open(path, 'r') as f:
            music_events = json.load(f)
            pieces.append(music_events)

    return pieces

def split_compositions(pieces, test_size=0.1, validation_size=0.1, random_state=42):
    train_val_pieces, test_pieces = train_test_split(pieces, test_size=test_size, random_state=random_state)
    validation_ratio = validation_size / (1 - test_size)
    train_pieces, validation_pieces = train_test_split(train_val_pieces, test_size=validation_ratio, random_state=random_state)
    return train_pieces, validation_pieces, test_pieces

def create_vocabularies(pieces):
    unique_events_values = {property_name: set() for property_name in MUSIC_EVENT_PROPERTIES}

    for composition in pieces:
        for music_event in composition:
            for property_name in MUSIC_EVENT_PROPERTIES:
                unique_events_values[property_name].add(str(music_event[property_name]))

    vocabularies = {}

    for property_name in MUSIC_EVENT_PROPERTIES:
        sorted_values = sorted(unique_events_values[property_name])
        vocabularies[property_name] = {value: number for number, value in enumerate(sorted_values)}

    return vocabularies

def prepare_sequences(pieces, vocabularies, sequence_length=16):
    network_input = {property_name: [] for property_name in MUSIC_EVENT_PROPERTIES}
    network_output = {property_name: [] for property_name in MUSIC_EVENT_PROPERTIES}

    for composition in pieces:
        if len(composition) <= sequence_length:
            continue

        for position in range(len(composition) - sequence_length):
            input_events = composition[position: position + sequence_length]
            next_event = composition[position + sequence_length]
            sequence_is_valid = True
            encoded_input_sequence = {}

            for property_name in MUSIC_EVENT_PROPERTIES:
                property_vocabulary = vocabularies[property_name]
                input_values = [str(music_event[property_name]) for music_event in input_events]
                output_value = str(next_event[property_name])

                if any(value not in property_vocabulary for value in input_values):
                    sequence_is_valid = False
                    break

                if output_value not in property_vocabulary:
                    sequence_is_valid = False
                    break

                encoded_input_sequence[property_name] = [property_vocabulary[value] for value in input_values]

            if not sequence_is_valid:
                continue

            for property_name in MUSIC_EVENT_PROPERTIES:
                network_input[property_name].append(encoded_input_sequence[property_name])
                output_value = str(next_event[property_name])
                network_output[property_name].append(vocabularies[property_name][output_value])


    for property_name in MUSIC_EVENT_PROPERTIES:
        network_input[property_name] = np.array(network_input[property_name], dtype=np.int32)
        network_output[property_name] = np.array(network_output[property_name], dtype=np.int32)

    for property_name in MUSIC_EVENT_PROPERTIES:
        print(property_name, network_input[property_name].shape, network_output[property_name].shape)

    return network_input, network_output

def save_prepared_data(output_folder, dataset_name, network_inputs, network_outputs):
    for property_name in MUSIC_EVENT_PROPERTIES:
        input_file_name = f"X_{property_name}_{dataset_name}.npy"
        output_file_name = f"y_{property_name}_{dataset_name}.npy"

        np.save(os.path.join(output_folder, input_file_name), network_inputs[property_name])
        np.save(os.path.join(output_folder, output_file_name), network_outputs[property_name])

def prepare_composer(parsed_folder, output_folder, sequence_length=16):
    compositions = load_compositions(parsed_folder)
    vocabs = create_vocabularies(compositions)
    train_compositions, validation_compositions, test_compositions = split_compositions(compositions)

    X_train, y_train = prepare_sequences(train_compositions, vocabs, sequence_length)
    X_validation, y_validation = prepare_sequences(validation_compositions, vocabs, sequence_length)
    X_test, y_test = prepare_sequences(test_compositions, vocabs, sequence_length)

    os.makedirs(output_folder, exist_ok=True)
    save_prepared_data(output_folder, "train", X_train, y_train)
    save_prepared_data(output_folder, "validation", X_validation, y_validation)
    save_prepared_data(output_folder, "test", X_test, y_test)

    with open(os.path.join(output_folder, 'vocabularies.json'), 'w') as f:
        json.dump(vocabs, f)

    print(f"Kompozicije za treniranje: {len(train_compositions)}")
    print(f"Kompozicije za validaciju: {len(validation_compositions)}")
    print(f"Kompozicije za testiranje: {len(test_compositions)}")
    print(f"Sekvence za treniranje: {len(X_train)}")
    print(f"Sekvence za validaciju: {len(X_validation)}")
    print(f"Sekvence za testiranje: {len(X_test)}")

if __name__ == "__main__":
    sequences_length = [4, 8, 16, 32]
    composers = ["bach", "mozart"]

    for composer in composers:
        print(f"===== {composer.capitalize()} =====")

        for sequence_lenght in sequences_length:
            print(f"Duzina sekvence: {sequence_lenght}")
            prepare_composer(
                parsed_folder=f"data/parsed/{composer}", 
                output_folder=f"data/prepared/{composer}/seq_{sequence_lenght}", 
                sequence_length=sequence_lenght
            )