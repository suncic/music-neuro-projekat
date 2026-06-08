import os
import json
from music21 import converter, instrument, note, chord

def parse_midi_folder(file_path):
    notes = []

    try:
        midi = converter.parse(file_path)

        for element in midi.flatten().notes:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.pitches))

    except Exception as e:
        print(f"Error in {file_path}: {e}")

    return notes

def parse_composer(midi_folder, parsed_folder, composer_name):
    print(f"Parsing {composer_name} MIDI files...")
    notes = []

    files = [f for f in os.listdir(midi_folder) if f.endswith('.mid')]
    print(f"Found {len(files)} files")

    for file_name in files:
        midi_path = os.path.join(midi_folder, file_name)
        json_name = file_name.replace('.mid', '.json')
        json_path = os.path.join(parsed_folder, json_name)

        if os.path.exists(json_path):
            print(f"Skipped (already exists): {json_name}")
            continue

        notes = parse_midi_folder(midi_path)

        if len(notes) == 0:
            print(f"Skipped (no notes): {file_name}")
            continue

        with open(json_path, 'w') as f:
            json.dump(notes, f)

        print(f"Saved: {json_name} {len(notes)} notes")

    print(f"Done: {composer_name}")

parse_composer('../data/bach', '../data/parsed/bach', 'Bach')
parse_composer('../data/mozart', '../data/parsed/mozart', 'Mozart')
parse_composer('../data/beethoven', '../data/parsed/beethoven', 'Beethoven')

print("Parsing complete")