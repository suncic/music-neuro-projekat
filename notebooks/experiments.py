from run_experiment import run_experiment

if __name__ == "__main__":
    number = int(input("Choose experiment to run (0 - 6): "))

    # bazni eksperiment
    if number == 0:
        run_experiment(
            experiment_name="baseline_seed42",
            composer_folder="../data/prepared/bach/seq_16",
            results_folder="../results/bach/baseline",
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
        learning_rates = [0.01, 0.001, 0.0001] # 0.1

        for learning_rate in learning_rates:
            run_experiment(
                experiment_name=f"lr_{learning_rate}_seed42",
                composer_folder="../data/prepared/bach/seq_16",
                results_folder="../results/bach/learning_rate",
                learning_rate=learning_rate,
                batch_size=32,
                lstm_units=128,
                epochs=100,
                patience=10,
                evaluate_test=False,
                seed=42
            )

    elif number == 2:
        batch_sizes = [16, 32, 64, 128]

        for batch_size in batch_sizes:
            run_experiment(
                experiment_name=f"batch_{batch_size}_seed42",
                composer_folder="../data/prepared/bach/seq_16",
                results_folder="../results/bach/batch_size",
                learning_rate=0.001,
                batch_size=batch_size,
                lstm_units=128,
                epochs=100,
                patience=10,
                evaluate_test=False,
                seed=42
            )

    elif number == 3:
        units_values = [64, 128, 256]

        for units in units_values:
            run_experiment(
                experiment_name=f"lstm_units_{units}_seed42",
                composer_folder="../data/prepared/bach/seq_16",
                results_folder="../results/bach/lstm_units",
                learning_rate=0.001,
                batch_size=32,
                lstm_units=units,
                epochs=100,
                patience=10,
                evaluate_test=False,
                seed=42
            )

    elif number == 4:
        sequence_lengths = [4, 8, 16, 32]

        for sequence_length in sequence_lengths:
            run_experiment(
                experiment_name=f"sequence_{sequence_length}_seed42",
                composer_folder=f"../data/prepared/bach/seq_{sequence_length}",
                results_folder="../results/bach/sequence_length",
                learning_rate=0.001,
                batch_size=32,
                lstm_units=128,
                epochs=100,
                patience=10,
                evaluate_test=False,
                seed=42
            )

    elif number == 5:
        architectures = [
            {
                "name": "lstm_64",
                "recurrent_type": "lstm",
                "lstm_units": 64,
                "second_layer_units": None
            },
            {
                "name": "lstm_128",
                "recurrent_type": "lstm",
                "lstm_units": 128,
                "second_layer_units": None
            },
            {
                "name": "deep_lstm_128_64",
                "recurrent_type": "lstm",
                "lstm_units": 128,
                "second_layer_units": 64
            },
            {
                "name": "gru_128",
                "recurrent_type": "gru",
                "lstm_units": 128,
                "second_layer_units": None
            }
        ]

        for architecture in architectures:
            run_experiment(
                experiment_name=f"{architecture['name']}_seed42",
                composer_folder="../data/prepared/bach/seq_16",
                results_folder="../results/bach/architecture",
                recurrent_type=architecture["recurrent_type"],
                lstm_units=architecture["lstm_units"],
                second_layer_units=architecture["second_layer_units"],
                learning_rate=0.001,
                batch_size=32,
                epochs=100,
                patience=10,
                evaluate_test=False,
                seed=42
            )

    elif number == 6:
        # bez ranog zaustavljanja
        run_experiment(
            experiment_name="without_early_stopping_seed42",
            composer_folder="../data/prepared/bach/seq_16",
            results_folder="../results/bach/early_stopping",
            learning_rate=0.001,
            batch_size=32,
            lstm_units=128,
            epochs=100,
            use_early_stopping=False,
            evaluate_test=False,
            seed=42
        )

        # sa ranim zaustavljanjem
        run_experiment(
            experiment_name="with_early_stopping_seed42",
            composer_folder="../data/prepared/bach/seq_16",
            results_folder="../results/bach/early_stopping",
            learning_rate=0.001,
            batch_size=32,
            lstm_units=128,
            epochs=100,
            patience=10,
            use_early_stopping=True,
            evaluate_test=False,
            seed=42
        )

    else:
        print("Invalid experiment number. Cjoose a number from 0 to 6.")