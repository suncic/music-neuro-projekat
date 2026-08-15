import os
import json
from music21 import converter, note, chord, tempo

DURATIONS = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
TIME_SHIFTS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
TEMPO_STEP = 5
DEFAULT_TEMPO = 120

def find_nearest_allowed_value(value, allowed_values):
    return min(allowed_values, key=lambda allowed_value: abs(allowed_value - value))

def round_tempo(bpm):
    return max(TEMPO_STEP, int(round(bpm / TEMPO_STEP) * TEMPO_STEP))

def get_tempo_changes(flat_midi):
    beat_tempo_pairs = []
    tempo_marks = flat_midi.getElementsByClass(tempo.MetronomeMark)

    for tempo_mark in tempo_marks:
        bpm = tempo_mark.getQuarterBPM()
        if bpm is None:
            continue
        
        beat_tempo_pairs.append((float(tempo_mark.offset), round_tempo(float(bpm))))

    if not beat_tempo_pairs or beat_tempo_pairs[0][0] > 0:
        beat_tempo_pairs.insert(0, (0.0, DEFAULT_TEMPO))

    return sorted(beat_tempo_pairs, key=lambda pair: pair[0])

def find_tempo_at_music_event_start(music_event_start, tempo_changes):
    tempo_at_event_start = DEFAULT_TEMPO

    for tempo_changes_start, new_tempo in tempo_changes:
        if tempo_changes_start > music_event_start:
            break

        tempo_at_event_start = new_tempo

    return tempo_at_event_start

def parse_midi_file(file_path):
    music_events = []

    try:
        midi = converter.parse(file_path)
        flat_midi = midi.flatten()
        tempo_changes = get_tempo_changes(flat_midi)
        previous_start_event_time = None

        for element in flat_midi.notesAndRests:
            current_start_event_time = float(element.offset)
            duration = find_nearest_allowed_value(float(element.duration.quarterLength), DURATIONS)

            if previous_start_event_time is None:
                time_difference = 0.0
            else:
                time_difference = find_nearest_allowed_value(max(0.0, current_start_event_time - previous_start_event_time), TIME_SHIFTS)

            if isinstance(element, note.Note):
                music_event_type = "NOTE"
                pitch = str(element.pitch)
            elif isinstance(element, chord.Chord):
                music_event_type = "CHORD"
                pitch = ".".join(str(value) for value in element.pitches)
            elif isinstance(element, note.Rest):
                music_event_type = "REST"
                pitch = "NONE"
            else:
                continue

            music_events.append(
                {
                    "type": music_event_type,
                    "pitch": pitch,
                    "duration": duration,
                    "time_shift": time_difference,
                    "tempo": find_tempo_at_music_event_start(current_start_event_time, tempo_changes),
                }
            )

            previous_start_event_time = current_start_event_time

    except Exception as e:
        print(f"Error in {file_path}: {e}")

    return music_events

def parse_composer(midi_folder, parsed_folder, composer_name):
    print(f"Parsing {composer_name} MIDI files")
    files = sorted(f for f in os.listdir(midi_folder) if f.endswith('.mid'))
    print(f"Found {len(files)} files")

    os.makedirs(parsed_folder, exist_ok=True)

    for file_name in files:
        midi_path = os.path.join(midi_folder, file_name)
        json_name = file_name.replace('.mid', '.json')
        json_path = os.path.join(parsed_folder, json_name)

        if os.path.exists(json_path):
            print(f"Skipped (already exists): {json_name}")
            continue

        music_events = parse_midi_file(midi_path)

        if len(music_events) == 0:
            print(f"Skipped (no music events): {file_name}")
            continue

        with open(json_path, 'w') as f:
            json.dump(music_events, f)

        print(f"Saved: {json_name} {len(music_events)} music events")

    print(f"Done: {composer_name}")

if __name__ == "__main__":
    parse_composer('data/bach', 'data/parsed/bach', 'Bach')
    parse_composer('data/mozart', 'data/parsed/mozart', 'Mozart')
    parse_composer('data/beethoven', 'data/parsed/beethoven', 'Beethoven')

    print("Parsing complete")