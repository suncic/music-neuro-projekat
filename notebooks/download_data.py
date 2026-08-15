import music21
from music21 import corpus
import os

def download_composer(composer_name, folder, limit=20):
    os.makedirs(folder, exist_ok=True)
    print(f"Downloading {composer_name} compositions...")

    pieces = corpus.getComposer(composer_name)
    print(f"Found {len(pieces)} compositions")

    for i, path in enumerate(pieces[:limit]):
        try:
            file_name = f"{composer_name}_{i+1}.mid"
            full_path = os.path.join(folder, file_name)

            if os.path.exists(full_path):
                print(f"Skipped (already exist): {file_name}")
                continue

            composition = corpus.parse(path)
            composition.write('midi', fp=full_path)
            print(f"Saved: {file_name}")
        except Exception as e:
            print(f"Error: {e}")

    print(f"Done: {composer_name}")
    print()

download_composer('bach', 'data/bach')
download_composer('mozart', 'data/mozart')
download_composer('beethoven', 'data/beethoven')