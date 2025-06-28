from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from store_sales_DL.config import MODELS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


#@app.command()
def main(
    #Input parameters for the training function
    test_npz_input_path = PROCESSED_DATA_DIR / "test_data.npz",
    model_path = MODELS_DIR / "best_model.keras",
    test_raw_input_path = RAW_DATA_DIR / "test.csv",
    train_raw_input_path = RAW_DATA_DIR / "train.csv",
    stores_raw_input_path = RAW_DATA_DIR / "stores.csv",
    #Output parameters for the training function
    test_predictions_output_path = PROCESSED_DATA_DIR / "test_predictions.npy",
    kaggle_submit_output_path = PROCESSED_DATA_DIR / "test_predictions_DL.csv",
    dashboard_dataset_output_path = PROCESSED_DATA_DIR / "dashboard_data.csv"
):

    logger.info("Performing inference for model...")

    # Load the test dataset
    test_data = np.load(test_npz_input_path)
    # Load the inputs from the test dataset
    test_inputs = test_data["inputs"]

    # Load the trained model
    model = load_model(model_path)
    # Make predictions on the test set
    test_predictions = model.predict(test_inputs)

    # Save predictions to a file
    np.save(test_predictions_output_path, test_predictions)

    # Step 1: Read the raw test dataset to get the 'id' column
    test_raw_df = pd.read_csv(test_raw_input_path)
    train_raw_df = pd.read_csv(train_raw_input_path)
    stores_raw_df = pd.read_csv(stores_raw_input_path)
    id_column = test_raw_df['id']

    # Step 2: Load the test predictions and rename the column to "sales"
    predictions = np.load(test_predictions_output_path)
    predictions_df = pd.DataFrame(predictions, columns=["sales"])

    # Step 3: Combine the 'id' column and predictions
    combined_df = pd.concat([id_column, predictions_df], axis=1)

    # Step 4: Save the combined DataFrame to the specified path
    combined_df.to_csv(kaggle_submit_output_path, index=False)
    logger.info("Starting to build dataset for visualizations...")

    test_with_predictions_df = pd.concat([test_raw_df, predictions_df], axis=1)
    # Add 'istrain' column
    test_with_predictions_df['History or predicted'] = 'Predicted'
    train_raw_df['History or predicted'] = 'History'

    # Union (concatenate) the two DataFrames
    full_df = pd.concat([train_raw_df, test_with_predictions_df], ignore_index=True)
    # Merge only 'city' and 'state' columns from stores_raw_df into full_df
    full_with_city_state = full_df.merge(
    stores_raw_df[['store_nbr', 'city', 'state']],
    on='store_nbr',
    how='left')
    full_with_city_state.to_csv(dashboard_dataset_output_path, index=False)

    logger.success("Inference and visualization dataset creation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    #app()
    main()

