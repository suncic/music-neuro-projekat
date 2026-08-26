from train_model import train_and_evaluate

COMPOSER_SETTINGS = {
    "bach": {
        "learning_rate": 0.001,
        "batch_size": 128
    },
    "mozart": {
        "learning_rate": 0.001,
        "batch_size": 128
    }
}

def run_selected_experiment(number, composer):
    settings = COMPOSER_SETTINGS[composer]

    learning_rate = settings["learning_rate"]
    batch_size = settings["batch_size"]
    composer_folder = f"data/prepared/{composer}/seq_16"

    print(f"\n{'=' * 50}")
    print(f"Kompozitor: {composer.capitalize()}")
    print(f"{'=' * 50}")

    if number == 0:
        train_and_evaluate(
            experiment_name="baseline",
            composer_folder=composer_folder,
            results_folder=f"results/{composer}/baseline",
            embedding_dim=64,
            recurrent_type="lstm",
            lstm_units=128,
            learning_rate=0.001,
            batch_size=32,
            epochs=100,
            patience=10,
            use_early_stopping=True,
            seed=42
        )
    
    elif number == 1:
        learning_rates = [0.01, 0.001, 0.0001]

        for current_learning_rate in learning_rates:
            train_and_evaluate(
                experiment_name=f"lr_{current_learning_rate}",
                composer_folder=composer_folder,
                results_folder=f"results/{composer}/learning_rate",
                learning_rate=current_learning_rate,
                batch_size=32,
                lstm_units=128,
                epochs=100,
                patience=10,
                seed=42
            )

    elif number == 2:
        batch_sizes = [16, 32, 64, 128]

        for current_batch_size in batch_sizes:
            train_and_evaluate(
                experiment_name=f"batch_{current_batch_size}",
                composer_folder=composer_folder,
                results_folder=f"results/{composer}/batch_size",
                learning_rate=0.001,
                batch_size=current_batch_size,
                lstm_units=128,
                epochs=100,
                patience=10,
                seed=42
            )

    elif number == 3:
        units_values = [64, 128, 256]

        for units in units_values:
            train_and_evaluate(
                experiment_name=f"lstm_units_{units}",
                composer_folder=composer_folder,
                results_folder=f"results/{composer}/lstm_units",
                learning_rate=learning_rate,
                batch_size=batch_size,
                lstm_units=units,
                epochs=100,
                patience=10,
                seed=42
            )

    elif number == 4:
        sequence_lengths = [4, 8, 16, 32]

        for sequence_length in sequence_lengths:
            train_and_evaluate(
                experiment_name=f"sequence_{sequence_length}",
                composer_folder=f"data/prepared/{composer}/seq_{sequence_length}",
                results_folder=f"results/{composer}/sequence_length",
                learning_rate=learning_rate,
                batch_size=batch_size,
                lstm_units=128, # posle eksperimenta 3 promeniti ovo
                epochs=100,
                patience=10,
                seed=42
            )

    elif number == 5:
        architectures = [
            {
                "name": "lstm_128",
                "recurrent_type": "lstm",
                "lstm_units": 128,
                "second_layer_units": None
            },
            {
                "name": "lstm_128_64",
                "recurrent_type": "lstm",
                "lstm_units": 128,
                "second_layer_units": 64
            },
            {
                "name": "gru_128",
                "recurrent_type": "gru",
                "lstm_units": 128,
                "second_layer_units": None
            },
            {
                "name": "gru_128_64",
                "recurrent_type": "gru",
                "lstm_units": 128,
                "second_layer_units": 64
            }
        ]

        for architecture in architectures:
            train_and_evaluate(
                experiment_name=f"{architecture['name']}",
                composer_folder=composer_folder, # u zavisnosti od velicine sekvence ovo se treba promeniti
                results_folder=f"results/{composer}/architecture",
                recurrent_type=architecture["recurrent_type"],
                lstm_units=architecture["lstm_units"],
                second_layer_units=architecture["second_layer_units"],
                learning_rate=learning_rate,
                batch_size=batch_size,
                epochs=100,
                patience=10,
                seed=42
            )

    elif number == 6:
        # bez ranog zaustavljanja
        train_and_evaluate(
            experiment_name="without_early_stopping",
            composer_folder=composer_folder, # u zavisnosti od velicine sekvence ovo se treba promeniti
            results_folder=f"results/{composer}/early_stopping",
            learning_rate=learning_rate,
            batch_size=batch_size,
            lstm_units=128, # treba promeniti
            epochs=100,
            use_early_stopping=False,
            seed=42
        )

        # sa ranim zaustavljanjem
        train_and_evaluate(
            experiment_name="with_early_stopping",
            composer_folder=composer_folder, # u zavisnosti od velicine sekvence ovo se treba promeniti
            results_folder=f"results/{composer}/early_stopping",
            learning_rate=learning_rate,
            batch_size=batch_size,
            lstm_units=128, # treba promeniti
            epochs=100,
            patience=10,
            use_early_stopping=True,
            seed=42
        )

if __name__ == "__main__":
    print("Eksperimenti:")
    print("0 - Baseline model")
    print("1 - Brzina ucenja")
    print("2 - Batch size")
    print("3 - Broj LSTM jedinica")
    print("4 - Dužina sekvence")
    print("5 - Arhitektura")
    print("6 - Early stopping i overfitting")

    number = int(input("Izaberite eksperiment (0 - 6): "))
    if number not in range(0, 7):
        print("Nepoznat broj. Izaberite broj od 0 do 6.")
        raise SystemExit

    composers = ["bach", "mozart"]
    for composer in composers:
        run_selected_experiment(number, composer)
