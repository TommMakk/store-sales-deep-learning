from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm
import pandas as pd
import numpy as np
from store_sales_DL.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, PROJ_ROOT, INTERIM_DATA_DIR
import os

app = typer.Typer()

#################################################################################
# PREREQUISITE                                                                  #
#################################################################################

#Download kaggle competition data files from:
#https://www.kaggle.com/competitions/Store-Sales-Forecasting/data
#put files to the following directory:
# ~/data/raw/xxx.csv

#################################################################################
def interim_train_and_test_with_lags(
    train_raw_input_path: Path,
    train_output_path: Path,
    test_raw_input_path: Path,
    test_output_path: Path
) -> None:
    """
    Generates interim train and test datasets with lag-based sales features.

    This function reads the raw train and test CSV files, concatenates them, removes
    problematic dates, and creates lagged sales features using exponential weighted means.
    The resulting datasets are split back into train and test sets and saved to disk.

    Args:
        train_raw_input_path (Path): Path to the raw training data CSV.
        train_output_path (Path): Path to save the processed training data CSV.
        test_raw_input_path (Path): Path to the raw test data CSV.
        test_output_path (Path): Path to save the processed test data CSV.

    Returns:
        None
    """
    logger.info("Starting to generate interim train and test datasets with lags...")

    # Load raw train and test datasets
    train_interim_df = pd.read_csv(train_raw_input_path).copy()
    test_df = pd.read_csv(test_raw_input_path).copy()

    # Add a flag to distinguish between train and test data
    train_interim_df['test'] = 0
    test_df['test'] = 1

    # Concatenate train and test data for consistent feature engineering
    data = pd.concat([train_interim_df, test_df], axis=0)

    # Remove rows with zero sales on '2013-01-01'
    data = data[data['date'] != '2013-01-01']

    # Reset index for groupby operations
    data_ = data.copy().reset_index()

    # Group by store and family for lag feature calculation
    grouped_data = data_.groupby(['store_nbr', 'family'])

    # Define alphas and lags for exponential weighted mean features
    alphas = [0.95, 0.8, 0.65, 0.5]
    lags = [1, 7, 30]

    # Create lagged sales features with different alphas and lags
    for a in alphas:
        for i in lags:
            data_[f'sales_lag_{i}_alpha_{a}'] = np.log1p(
                grouped_data['sales'].transform(
                    lambda x: x.shift(i).ewm(alpha=a, min_periods=1).mean()
                )
            )

    # Fill NaN values with 0 for all lagged sales columns
    sales_cols = [col for col in data_.columns if col.startswith("sales_")]
    data_[sales_cols] = data_[sales_cols].fillna(0)

    # Split back into test and train sets, dropping unnecessary columns
    test_only_df = data_[data_['test'] == 1].drop(columns=['sales', 'test', 'index'])
    train_only_df = data_[data_['test'] == 0].drop(columns=['test', 'index'])

    # Save processed datasets to the specified output paths
    test_only_df.to_csv(test_output_path, index=False)
    train_only_df.to_csv(train_output_path, index=False)

    logger.info("Completed to generate interim train and test datasets with lags")

def interim_family_encoded_df(
    train_raw_input_path: Path,
    family_output_path: Path
) -> None:
    """
    Encodes the 'family' column from the training dataset using one-hot encoding.

    This function reads the raw training data CSV, extracts unique values from the 'family'
    column, performs one-hot encoding (dropping the first column to avoid multicollinearity),
    and saves the resulting DataFrame to the specified output path.

    Args:
        train_raw_input_path (Path): Path to the raw training data CSV.
        family_output_path (Path): Path to save the one-hot encoded family DataFrame.

    Returns:
        None
    """
    logger.info("Starting to encode family column...")

    # Load the raw training dataset
    train_df = pd.read_csv(train_raw_input_path).copy()

    # Extract unique values from the 'family' column
    unique_family_df = train_df[['family']].drop_duplicates()

    # Perform one-hot encoding for the 'family' column, drop first to avoid multicollinearity
    family_one_hot_encoded = pd.get_dummies(unique_family_df, columns=['family'], drop_first=True)

    # Convert one-hot encoded values to integers (0 and 1)
    family_one_hot_encoded = family_one_hot_encoded.astype(int)

    # Combine the original 'family' column with the one-hot encoded columns
    family_encoded_df = pd.concat([unique_family_df.reset_index(drop=True), family_one_hot_encoded], axis=1)

    # Save the resulting DataFrame to the specified output path
    family_encoded_df.to_csv(family_output_path, index=False)

    logger.info("Completed to encode family column")

def interim_stores_encoded_df(
    stores_raw_input_path: Path,
    stores_encoded_output_path: Path
) -> None:
    """
    Encodes the store-related columns from the stores dataset using one-hot encoding.

    This function reads the raw stores data CSV, performs one-hot encoding on the
    'city', 'state', and 'type' columns (dropping the first category to avoid multicollinearity),
    converts boolean columns to integers, drops the 'store_nbr' column to avoid duplication,
    concatenates the original and encoded columns, and saves the resulting DataFrame
    to the specified output path.

    Args:
        stores_raw_input_path (Path): Path to the raw stores data CSV.
        stores_encoded_output_path (Path): Path to save the one-hot encoded stores DataFrame.

    Returns:
        None
    """
    logger.info("Starting to encode stores column...")

    # Load the raw stores dataset
    stores_df = pd.read_csv(stores_raw_input_path).copy()

    # One-hot encode the 'city', 'state', and 'type' columns
    stores_encoded_df = pd.get_dummies(stores_df, columns=['city', 'state', 'type'], drop_first=True)

    # Convert any boolean columns to integers (0 and 1)
    stores_encoded_df = stores_encoded_df.astype(int)

    # Drop the 'store_nbr' column from the one-hot encoded DataFrame to avoid duplication
    stores_encoded_df = stores_encoded_df.drop(columns=['store_nbr'])

    # Concatenate the original columns with the one-hot encoded columns
    stores_final_df = pd.concat([stores_df, stores_encoded_df], axis=1)

    # Save the final DataFrame to the specified path
    stores_final_df.to_csv(stores_encoded_output_path, index=False)

    logger.info("Completed to encode stores column")

def interim_oil_filled_dates_and_nulls(
    oil_raw_input_path: Path,
    oil_filled_output_path: Path
) -> None:
    """
    Fills missing oil price values and missing dates in the oil dataset.

    This function reads the raw oil data CSV, ensures all dates in the range
    2013-01-01 to 2017-08-31 are present, and fills missing 'dcoilwtico' values
    using the previous available value or the next available value if necessary.
    The resulting DataFrame is saved to the specified output path.

    Args:
        oil_raw_input_path (Path): Path to the raw oil data CSV.
        oil_filled_output_path (Path): Path to save the processed oil DataFrame.

    Returns:
        None
    """
    logger.info("Starting to process oil dataset...")

    # Load the raw oil dataset
    oil_filled_df = pd.read_csv(oil_raw_input_path).copy()

    # Ensure the 'date' column is in datetime format
    oil_filled_df['date'] = pd.to_datetime(oil_filled_df['date'])

    # Create a date range from 2013-01-01 to 2017-08-31
    date_range = pd.date_range(start='2013-01-01', end='2017-08-31')

    # Reindex the DataFrame to include all dates in the date range
    oil_filled_df = oil_filled_df.set_index('date').reindex(date_range).rename_axis('date').reset_index()

    # Fill missing 'dcoilwtico' values using previous or next available value
    for i in range(len(oil_filled_df)):
        if pd.isnull(oil_filled_df.loc[i, 'dcoilwtico']):
            # Use previous value if available
            if i > 0 and not pd.isnull(oil_filled_df.loc[i - 1, 'dcoilwtico']):
                oil_filled_df.loc[i, 'dcoilwtico'] = oil_filled_df.loc[i - 1, 'dcoilwtico']
            # Otherwise, use next value if available
            elif i < len(oil_filled_df) - 1 and not pd.isnull(oil_filled_df.loc[i + 1, 'dcoilwtico']):
                oil_filled_df.loc[i, 'dcoilwtico'] = oil_filled_df.loc[i + 1, 'dcoilwtico']

    # Save the final DataFrame to the specified path
    oil_filled_df.to_csv(oil_filled_output_path, index=False)

    logger.info("Completed to process oil dataset")

def interim_holidays(
    stores_raw_input_path: Path,
    holidays_events_raw_input_path: Path,
    holidays_events_intermin_output_path: Path,
    holiday_output_path: Path,
    event_output_path: Path,
    additional_output_path: Path,
    transfer_output_path: Path,
    bridge_output_path: Path
) -> None:
    """
    Processes and encodes holiday events for each city and saves multiple event-type DataFrames.

    This function merges holiday events with store locations to create city-level holiday features,
    filters and splits the data by event type, and saves the results to specified output paths.

    Args:
        stores_raw_input_path (Path): Path to the raw stores data CSV.
        holidays_events_raw_input_path (Path): Path to the raw holidays_events data CSV.
        holidays_events_intermin_output_path (Path): Path to save the merged holidays per city DataFrame.
        holiday_output_path (Path): Path to save the holiday events DataFrame.
        event_output_path (Path): Path to save the event events DataFrame.
        additional_output_path (Path): Path to save the additional events DataFrame.
        transfer_output_path (Path): Path to save the transfer events DataFrame.
        bridge_output_path (Path): Path to save the bridge events DataFrame.

    Returns:
        None
    """
    logger.info("Starting to process holiday datasets...")

    # Load the raw stores dataset
    stores_df = pd.read_csv(stores_raw_input_path).copy()
    # Create a new DataFrame with unique combinations of 'city' and 'state'
    unique_city_state_df = stores_df.drop_duplicates(subset=['city', 'state'])

    # Load the raw holidays_events dataset
    holidays_events_df = pd.read_csv(holidays_events_raw_input_path).copy()
    # Filter holidays_events_df for 'National', 'Regional', and 'Local' locales
    national_holidays_df = holidays_events_df[holidays_events_df['locale'] == "National"]
    regional_holidays_df = holidays_events_df[holidays_events_df['locale'] == "Regional"]
    local_holidays_df = holidays_events_df[holidays_events_df['locale'] == "Local"]

    # Merge 'National' holidays with all rows in unique_city_state_df (cross join)
    national_merged_df = national_holidays_df.assign(key=1).merge(unique_city_state_df.assign(key=1), on='key').drop('key', axis=1)

    # Merge 'Regional' holidays with unique_city_state_df on 'state' and 'locale_name'
    regional_merged_df = regional_holidays_df.merge(unique_city_state_df, left_on='locale_name', right_on='state', how='left')

    # Merge 'Local' holidays with unique_city_state_df on 'city' and 'locale_name'
    local_merged_df = local_holidays_df.merge(unique_city_state_df, left_on='locale_name', right_on='city', how='left')

    # Concatenate the three merged DataFrames
    holidays_per_city_df = pd.concat([national_merged_df, regional_merged_df, local_merged_df], ignore_index=True)

    # Drop unnecessary columns and rename for consistency
    holidays_per_city_df.drop(columns=['store_nbr', 'type_y', 'cluster'], inplace=True)
    holidays_per_city_df.rename(columns={'type_x': 'type'}, inplace=True)

    # Save the merged holidays per city DataFrame
    holidays_per_city_df.to_csv(holidays_events_intermin_output_path, index=False)

    # Filter out rows where 'transferred' is True
    holidays_per_city_without_transferred_df = holidays_per_city_df[holidays_per_city_df['transferred'] != True]

    # Remove duplicates based on 'date', 'type', and 'city'
    unique_holidays_per_city_df = holidays_per_city_without_transferred_df.drop_duplicates(subset=['date', 'type', 'city'])

    # Create DataFrames for each event type
    holiday_df = unique_holidays_per_city_df[unique_holidays_per_city_df['type'] == "Holiday"].copy()
    event_df = unique_holidays_per_city_df[unique_holidays_per_city_df['type'] == "Event"].copy()
    additional_df = unique_holidays_per_city_df[unique_holidays_per_city_df['type'] == "Additional"].copy()
    transfer_df = unique_holidays_per_city_df[unique_holidays_per_city_df['type'] == "Transfer"].copy()
    bridge_df = unique_holidays_per_city_df[unique_holidays_per_city_df['type'] == "Bridge"].copy()

    # Add binary indicator columns for each event type using .loc
    holiday_df.loc[:, 'is_holiday'] = holiday_df['description'].notnull().astype(int)
    event_df.loc[:, 'is_event'] = event_df['description'].notnull().astype(int)
    additional_df.loc[:, 'is_additional'] = additional_df['description'].notnull().astype(int)
    transfer_df.loc[:, 'is_transfer'] = transfer_df['description'].notnull().astype(int)
    bridge_df.loc[:, 'is_bridge'] = bridge_df['description'].notnull().astype(int)

    # Save each event type DataFrame to the specified path
    holiday_df.to_csv(holiday_output_path, index=False)
    event_df.to_csv(event_output_path, index=False)
    additional_df.to_csv(additional_output_path, index=False)
    transfer_df.to_csv(transfer_output_path, index=False)
    bridge_df.to_csv(bridge_output_path, index=False)

    logger.info("Completed to process holiday datasets...")

#@app.command()
def main(
    # Defining input paths
    oil_raw_input_path: Path = RAW_DATA_DIR / "oil.csv",
    holidays_events_raw_input_path: Path = RAW_DATA_DIR / "holidays_events.csv",
    stores_raw_input_path: Path = RAW_DATA_DIR / "stores.csv",
    train_raw_input_path: Path = RAW_DATA_DIR / "train.csv",
    test_raw_input_path: Path = RAW_DATA_DIR / "test.csv",
    # Define the output paths
    test_output_path = INTERIM_DATA_DIR / "test.csv",
    train_output_path = INTERIM_DATA_DIR / "train.csv",
    family_output_path = INTERIM_DATA_DIR / "family_encoded_df.csv",
    stores_encoded_output_path = INTERIM_DATA_DIR / "stores_encoded_df.csv",
    oil_filled_output_path = INTERIM_DATA_DIR / "oil_filled_dates_and_nulls.csv",
    holidays_events_intermin_output_path: Path = INTERIM_DATA_DIR / "holidays_events_per_city.csv",
    holiday_output_path = INTERIM_DATA_DIR / "holidays_holiday_df.csv",
    event_output_path = INTERIM_DATA_DIR / "holidays_event_df.csv",
    additional_output_path = INTERIM_DATA_DIR / "holidays_additional_df.csv",
    transfer_output_path = INTERIM_DATA_DIR / "holidays_transfer_df.csv",
    bridge_output_path = INTERIM_DATA_DIR / "holidays_bridge_df.csv"
):
    logger.info("Starting dataset processing pipeline...")

    interim_train_and_test_with_lags(
        train_raw_input_path,
        train_output_path,
        test_raw_input_path,
        test_output_path
    )
    interim_family_encoded_df(
        train_raw_input_path,
        family_output_path
    )
    interim_stores_encoded_df(
        stores_raw_input_path,
        stores_encoded_output_path
    )
    interim_oil_filled_dates_and_nulls(
        oil_raw_input_path,
        oil_filled_output_path
    )
    interim_holidays(
        stores_raw_input_path,
        holidays_events_raw_input_path,
        holidays_events_intermin_output_path,
        holiday_output_path,
        event_output_path,
        additional_output_path,
        transfer_output_path,
        bridge_output_path
    )

    logger.success("Dataset processing pipeline complete.")



if __name__ == "__main__":
    #app()
    main()
