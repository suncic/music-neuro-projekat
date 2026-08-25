import music21
from music21 import corpus
import os

def download_composer(composer_name, folder, limit=20):
    os.makedirs(folder, exist_ok=True)
    print(f"Preuzimanje {composer_name} kompozicija")

    pieces = corpus.getComposer(composer_name)
    print(f"Pronadjeno {len(pieces)} kompozicija")

    for i, composition_pointer in enumerate(pieces[:limit]):
        try:
            file_name = f"{composer_name}_{i+1}.mid"
            full_path = os.path.join(folder, file_name)

            if os.path.exists(full_path):
                print(f"Preskocena kompozicija posto vec postoji preuzeta: {file_name}")
                continue

            composition = corpus.parse(composition_pointer)
            composition.write('midi', fp=full_path)
            print(f"Sacuvano {file_name}")
        except Exception as e:
            print(f"Greska prilikom preuzimanja: {e}")

    print(f"Zavrseno preuzimanje kompozicija od {composer_name}")
    print()

download_composer('bach', 'data/bach')
download_composer('mozart', 'data/mozart')