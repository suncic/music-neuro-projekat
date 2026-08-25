from train_model import train_and_evaluate

if __name__ == "__main__":
    number = int(input("Izaberite eksperiment (0 - 6): "))
    composers = ["bach", "mozart"]

    if number < 0 or number > 6:
        print("Nepoznat broj. Izbrati broj od 0 do 6.")

    for composer in composers:
        print(f"\n{'=' * 50}")
        print(f"Kompozitor: {composer.capitalize()}")
        print(f"{'=' * 50}")

        if number == 0:
            train_and_evaluate(
                experiment_name="baseline",
                composer_folder=f"data/prepared/{composer}/seq_16",
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

            for learning_rate in learning_rates:
                train_and_evaluate(
                    experiment_name=f"lr_{learning_rate}",
                    composer_folder=f"data/prepared/{composer}/seq_16",
                    results_folder=f"results/{composer}/learning_rate",
                    learning_rate=learning_rate,
                    batch_size=32,
                    lstm_units=128,
                    epochs=100,
                    patience=10,
                    seed=42
                )

        elif number == 2:
            batch_sizes = [16, 32, 64, 128]

            for batch_size in batch_sizes:
                train_and_evaluate(
                    experiment_name=f"batch_{batch_size}",
                    composer_folder=f"data/prepared/{composer}/seq_16",
                    results_folder=f"results/{composer}/batch_size",
                    learning_rate=0.001,
                    batch_size=batch_size,
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
                    composer_folder=f"data/prepared/{composer}/seq_16",
                    results_folder=f"results/{composer}/lstm_units",
                    learning_rate=0.001,
                    batch_size=64,
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
                    learning_rate=0.001,
                    batch_size=64,
                    lstm_units=128,
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
                    composer_folder=f"data/prepared/{composer}/seq_16",
                    results_folder=f"results/{composer}/architecture",
                    recurrent_type=architecture["recurrent_type"],
                    lstm_units=architecture["lstm_units"],
                    second_layer_units=architecture["second_layer_units"],
                    learning_rate=0.001,
                    batch_size=64,
                    epochs=100,
                    patience=10,
                    seed=42
                )

        elif number == 6:
            # bez ranog zaustavljanja
            train_and_evaluate(
                experiment_name="without_early_stopping",
                composer_folder=f"data/prepared/{composer}/seq_16",
                results_folder=f"results/{composer}/early_stopping",
                learning_rate=0.001,
                batch_size=64,
                lstm_units=128,
                epochs=100,
                use_early_stopping=False,
                seed=42
            )

            # sa ranim zaustavljanjem
            train_and_evaluate(
                experiment_name="with_early_stopping",
                composer_folder=f"data/prepared/{composer}/seq_16",
                results_folder=f"results/{composer}/early_stopping",
                learning_rate=0.001,
                batch_size=64,
                lstm_units=128,
                epochs=100,
                patience=10,
                use_early_stopping=True,
                seed=42
            )