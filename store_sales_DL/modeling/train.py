from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# Importing the configuration for paths
from store_sales_DL.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


#@app.command()
def main(
    #Input parameters for the training function
    train_processed_input_path = PROCESSED_DATA_DIR / "train_data.npz",
    validation_processed_input_path = PROCESSED_DATA_DIR / "validation_data.npz",
    test_npz_input_path = PROCESSED_DATA_DIR / "test_data.npz",
    #Output parameters for the training function
    best_model_path = MODELS_DIR / "best_model.keras"
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Training some model...")

    # Load the datasets
    train_data = np.load(train_processed_input_path)
    validation_data = np.load(validation_processed_input_path)
    test_data = np.load(test_npz_input_path)

    train_inputs, train_targets = train_data["inputs"], train_data["targets"]
    validation_inputs, validation_targets = validation_data["inputs"], validation_data["targets"]
    test_inputs = test_data["inputs"]

    # Define hyperparameters
    HYPERPARAMETERS = {
        "input_dim": train_inputs.shape[1],  # Number of input features (matches the number of columns in the input data)
        "hidden_units": [96, 64],  # Number of neurons in each hidden layer (controls model complexity)
        "dropout_rate": 0.1,  # Fraction of neurons to drop during training (prevents overfitting)
        "learning_rate": 0.001,  # Step size for the optimizer (controls how quickly the model learns)
        "batch_size": 64,  # Number of samples processed before updating model weights (affects training speed and memory usage)
        "epochs": 20,  # Number of complete passes through the training dataset (controls training duration)
        "early_stopping_patience": 4,  # Number of epochs to wait for improvement in validation loss before stopping (prevents overfitting)
    }


    def build_model(hyperparams):
        model = Sequential()
        model.add(Dense(hyperparams["hidden_units"][0], activation="relu", input_dim=hyperparams["input_dim"]))
        model.add(Dense(hyperparams["hidden_units"][1], activation="relu"))
        model.add(Dense(1, activation="linear"))
        return model

    logger.info("Starting compiling the model...")
    # Compile the model
    model = build_model(HYPERPARAMETERS)
    model.compile(
        optimizer=Adam(learning_rate=HYPERPARAMETERS["learning_rate"]),
        loss="mse",
        metrics=["mae"]
    )
    logger.info("Completed compiling the model")
    # Define callbacks
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=HYPERPARAMETERS["early_stopping_patience"],
            restore_best_weights=True
        )
    ]
    logger.info("Starting training the model...")
    # Train the model
    history = model.fit(
        train_inputs,
        train_targets,
        validation_data=(validation_inputs, validation_targets),
        batch_size=HYPERPARAMETERS["batch_size"],
        epochs=HYPERPARAMETERS["epochs"],
        callbacks=callbacks,
        verbose=1
    )
    logger.info("Completed training the model")
    # Evaluate the model on the validation set
    val_loss, val_mae = model.evaluate(validation_inputs, validation_targets, verbose=0)
    print(f"Validation Loss: {val_loss:.4f}, Validation MAE: {val_mae:.4f}")

    model.save(best_model_path)
    logger.success(f"Best model saved to {best_model_path}")

    logger.success("Modeling training complete.")
    # -----------------------------------------


if __name__ == "__main__":
    #app()
    main()
