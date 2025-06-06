from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
from store_sales_DL.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, PROJ_ROOT, INTERIM_DATA_DIR, drop_columns

app = typer.Typer()

def extract_date_features(input_DF: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts basic date features from the 'date' column in the input DataFrame.

    Parameters:
    - input_DF (pd.DataFrame): The input DataFrame containing a 'date' column.

    Returns:
    - pd.DataFrame: The updated DataFrame with extracted date features.
    """

    logger.info("Started extract_date_features...")

    # Ensure the 'date' column is in datetime format
    input_DF['date'] = pd.to_datetime(input_DF['date'])

    # Extract Basic Date Features
    input_DF['year'] = input_DF['date'].dt.year  # Year (e.g., 2021, 2022)
    input_DF['month'] = input_DF['date'].dt.month  # Month (1 = January, 12 = December)
    input_DF['day_of_month'] = input_DF['date'].dt.day  # Day of the Month (1, 15, 31)
    input_DF['day_of_week'] = input_DF['date'].dt.dayofweek  # Day of the Week (0 = Monday, 6 = Sunday)

    logger.info("Completed extract_date_features")

    # Return the updated DataFrame
    return input_DF


def get_paydays(date: datetime) -> tuple[datetime, datetime]:
    """
    Calculates the 15th and the last day of the given month.

    Parameters:
    - date (datetime): The input date.

    Returns:
    - tuple[datetime, datetime]: A tuple containing the 15th and the last day of the month.
    """
    fifteenth = date.replace(day=15)  # Set the day to the 15th of the month
    # Calculate the last day of the month by moving to the next month and subtracting one day
    last_day = (date.replace(day=1) + timedelta(days=32)).replace(day=1) - timedelta(days=1)
    return fifteenth, last_day


def days_to_next_payday(date: datetime) -> int:
    """
    Calculates the number of days to the next payday (15th or last day of the month).

    Parameters:
    - date (datetime): The input date.

    Returns:
    - int: The number of days to the next payday.
    """

    fifteenth, last_day = get_paydays(date)  # Get the 15th and last day of the month
    if date <= fifteenth:  # If the date is before or on the 15th
        next_payday = fifteenth
    else:  # If the date is after the 15th
        next_payday = last_day

    return (next_payday - date).days  # Calculate the difference in days


def days_from_previous_payday(date: datetime) -> int:
    """
    Calculates the number of days from the previous payday (15th or last day of the month).

    Parameters:
    - date (datetime): The input date.

    Returns:
    - int: The number of days from the previous payday.
    """

    fifteenth, last_day = get_paydays(date)  # Get the 15th and last day of the month
    if date == fifteenth or date == last_day:  # If the date is a payday
        return 0  # Return 0 since it's the payday
    elif date > fifteenth:  # If the date is after the 15th
        previous_payday = fifteenth
    else:  # If the date is before or on the 15th
        # The previous payday is the last day of the previous month
        previous_payday = (date.replace(day=1) - timedelta(days=1))

    return (date - previous_payday).days  # Calculate the difference in days


def is_payday(date: datetime) -> int:
    """
    Checks if the given date is a payday (15th or last day of the month).

    Parameters:
    - date (datetime): The input date.

    Returns:
    - int: 1 if the date is a payday, 0 otherwise.
    """

    fifteenth, last_day = get_paydays(date)  # Get the 15th and last day of the month

    return int(date == fifteenth or date == last_day)  # Return 1 if it's a payday, otherwise 0

def add_days_after_earthquake_feature(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a numerical feature for days after the earthquake to the input DataFrame.

    Parameters:
    - input_df (pd.DataFrame): The input DataFrame containing a 'date' column.

    Returns:
    - pd.DataFrame: The updated DataFrame with the 'days_after_earthquake' feature.
    """

    logger.info("Starting add_days_after_earthquake_feature...")

    # Ensure the 'date' column is in datetime format
    input_df['date'] = pd.to_datetime(input_df['date'])

    # Define the earthquake date (hardcoded)
    earthquake_date = datetime(2016, 4, 16)

    # Create a numerical feature for days after the earthquake
    input_df['days_after_earthquake'] = input_df['date'].apply(
        lambda x: (x - earthquake_date).days + 1 if x >= earthquake_date else 0
    )

    logger.info("Completed add_days_after_earthquake_feature")

    return input_df

def merge_oil_data(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges oil price data with the input DataFrame based on the 'date' column.

    Parameters:
    - input_df (pd.DataFrame): The input DataFrame containing a 'date' column.

    Returns:
    - pd.DataFrame: The updated DataFrame with oil price data merged.
    """

    logger.info("Starting merge_oil_data...")

    # Define the path for the oil data
    oil_data_path = INTERIM_DATA_DIR / "oil_filled_dates_and_nulls.csv"

    # Load the oil dataset
    oil_df = pd.read_csv(oil_data_path)

    # Ensure the 'date' column is in datetime format for both DataFrames
    input_df['date'] = pd.to_datetime(input_df['date'])
    oil_df['date'] = pd.to_datetime(oil_df['date'])

    # Merge the DataFrames on the 'date' column
    merged_df = pd.merge(input_df, oil_df, on='date', how='left')

    logger.info("Completed merge_oil_data")

    return merged_df


def merge_stores_data(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges store data with the input DataFrame based on the 'store_nbr' column.

    Parameters:
    - input_df (pd.DataFrame): The input DataFrame containing a 'store_nbr' column.

    Returns:
    - pd.DataFrame: The updated DataFrame with store data merged.
    """

    logger.info("Starting merge_stores_data...")

    # Define the input path for stores_encoded_df.csv
    stores_encoded_path = INTERIM_DATA_DIR / "stores_encoded_df.csv"

    # Load the stores_encoded_df dataset
    stores_encoded_df = pd.read_csv(stores_encoded_path)

    # Ensure the 'store_nbr' column is present in both DataFrames
    input_df['store_nbr'] = input_df['store_nbr'].astype(int)
    stores_encoded_df['store_nbr'] = stores_encoded_df['store_nbr'].astype(int)

    # Merge the DataFrames on the 'store_nbr' column
    merged_df = pd.merge(input_df, stores_encoded_df, on='store_nbr', how='left')

    logger.info("Completed merge_stores_data")

    return merged_df

def merge_family_data(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges family data with the input DataFrame based on the 'family' column.

    Parameters:
    - input_df (pd.DataFrame): The input DataFrame containing a 'family' column.

    Returns:
    - pd.DataFrame: The updated DataFrame with family data merged.
    """

    logger.info("Starting merge_family_data...")

    # Define the input path for family_encoded_df.csv
    family_encoded_path = INTERIM_DATA_DIR / "family_encoded_df.csv"

    # Load the family_encoded_df dataset
    family_encoded_df = pd.read_csv(family_encoded_path)

    # Ensure the 'family' column is present in both DataFrames and is of the same type
    input_df['family'] = input_df['family'].astype(str)
    family_encoded_df['family'] = family_encoded_df['family'].astype(str)

    # Merge the DataFrames on the 'family' column
    merged_df = pd.merge(input_df, family_encoded_df, on='family', how='left')

    logger.info("Completed merge_family_data")

    return merged_df


def merge_holiday_event_data(input_DF: pd.DataFrame) -> pd.DataFrame:
    """
    Merges holiday and event data with the input DataFrame.

    Parameters:
    - input_DF (pd.DataFrame): The input DataFrame to merge with holiday and event data.

    Returns:
    - pd.DataFrame: The merged DataFrame with holiday and event data.
    """

    logger.info("Starting merge_holiday_event_data...")

    # Define the input paths
    holiday_input_path = INTERIM_DATA_DIR / "holidays_holiday_df.csv"
    event_input_path = INTERIM_DATA_DIR / "holidays_event_df.csv"
    additional_input_path = INTERIM_DATA_DIR / "holidays_additional_df.csv"
    transfer_input_path = INTERIM_DATA_DIR / "holidays_transfer_df.csv"
    bridge_input_path = INTERIM_DATA_DIR / "holidays_bridge_df.csv"

    # Read each DataFrame from the specified path
    holiday_df = pd.read_csv(holiday_input_path)
    event_df = pd.read_csv(event_input_path)
    additional_df = pd.read_csv(additional_input_path)
    transfer_df = pd.read_csv(transfer_input_path)
    bridge_df = pd.read_csv(bridge_input_path)

    # Select only the necessary columns for merging
    holiday_df = holiday_df[['date', 'city', 'is_holiday']]
    event_df = event_df[['date', 'city', 'is_event']]
    additional_df = additional_df[['date', 'city', 'is_additional']]
    transfer_df = transfer_df[['date', 'city', 'is_transfer']]
    bridge_df = bridge_df[['date', 'city', 'is_bridge']]

    # Ensure the 'date' column is in datetime format for all DataFrames
    input_DF['date'] = pd.to_datetime(input_DF['date'])
    holiday_df['date'] = pd.to_datetime(holiday_df['date'])
    event_df['date'] = pd.to_datetime(event_df['date'])
    additional_df['date'] = pd.to_datetime(additional_df['date'])
    transfer_df['date'] = pd.to_datetime(transfer_df['date'])
    bridge_df['date'] = pd.to_datetime(bridge_df['date'])

    # Make a copy of the input DataFrame
    merged_DF = input_DF.copy()

    # Merge each DataFrame with the input DataFrame
    merged_DF = merged_DF.merge(holiday_df, on=['date', 'city'], how='left')
    merged_DF = merged_DF.merge(event_df, on=['date', 'city'], how='left')
    merged_DF = merged_DF.merge(additional_df, on=['date', 'city'], how='left')
    merged_DF = merged_DF.merge(transfer_df, on=['date', 'city'], how='left')
    merged_DF = merged_DF.merge(bridge_df, on=['date', 'city'], how='left')

    # Fill NaN values with 0 (since missing values mean the event did not occur)
    merged_DF[['is_holiday', 'is_event', 'is_additional', 'is_transfer', 'is_bridge']] = merged_DF[
        ['is_holiday', 'is_event', 'is_additional', 'is_transfer', 'is_bridge']
    ].fillna(0)

    logger.info("Completed merge_holiday_event_data")

    # Return the final merged DataFrame
    return merged_DF

def optimize_model(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimizes the input DataFrame by dropping unnecessary columns based on feature engineering,
    correlation matrix, and VIF analysis.

    Parameters:
    - input_df (pd.DataFrame): The input DataFrame to optimize.

    Returns:
    - pd.DataFrame: The optimized DataFrame.
    """

    logger.info("Starting to optimize model...")

    # Ensure the 'date' column is in datetime format
    input_df['date'] = pd.to_datetime(input_df['date'])

    # Drop origin of feature engineered columns
    columns_to_remove = ['date', 'store_nbr', 'family', 'is_payday', 'days_from_previous_payday', 'cluster.1', 'city', 'state', 'type']
    input_df = drop_columns(input_df, columns_to_remove)

    # Extract columns starting with "state_"
    state_columns_to_remove = [col for col in input_df.columns if col.startswith("state_")]

    # Dropping columns based on correlation matrix, VIF, etc.
    columns_to_remove = ['is_payday', 'days_from_previous_payday']
    input_df = drop_columns(input_df, state_columns_to_remove)
    input_df = drop_columns(input_df, columns_to_remove)

    # Dropping id column
    columns_to_remove = ['id']
    input_df = drop_columns(input_df, columns_to_remove)

    logger.info("Feature engineering function completed successfully.")

    return input_df


def feature_engineering(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Executes the full preprocessing pipeline by calling all sub-functions in sequence.

    Parameters:
    - input_df (pd.DataFrame): The input DataFrame to preprocess.

    Returns:
    - pd.DataFrame: The fully preprocessed DataFrame.
    """

    logger.info("Completed optimizing model...")

    # Extract date features
    input_df = extract_date_features(input_df)

    # Add payday-related features
    #input_df['days_to_next_payday'] = input_df['date'].apply(days_to_next_payday)
    #input_df['days_from_previous_payday'] = input_df['date'].apply(days_from_previous_payday)
    #input_df['is_payday'] = input_df['date'].apply(is_payday)
    #days_to_next_payday('2021-01-01')  # Example call to ensure the function is define
    logger.info("Starting payday features generation...")
    input_df.loc[:, 'days_to_next_payday'] = input_df['date'].apply(days_to_next_payday)
    input_df.loc[:, 'days_from_previous_payday'] = input_df['date'].apply(days_from_previous_payday)
    input_df.loc[:, 'is_payday'] = input_df['date'].apply(is_payday)
    logger.info("Completed generating payday features.")

    # Add earthquake-related feature
    input_df = add_days_after_earthquake_feature(input_df)

    # Merge oil data
    input_df = merge_oil_data(input_df)

    # Merge stores data
    input_df = merge_stores_data(input_df)

    # Merge family data
    input_df = merge_family_data(input_df)

    # Merge holiday and event data
    input_df = merge_holiday_event_data(input_df)

    
    logger.info("Feature engineering function completed successfully.")

    return input_df

#@app.command()
def main(
    # Define the input paths
    test_interim_input_path = INTERIM_DATA_DIR / "test.csv",
    train_interim_input_path = INTERIM_DATA_DIR / "train.csv",
    family_input_path = INTERIM_DATA_DIR / "family_encoded_df.csv",
    stores_encoded_input_path = INTERIM_DATA_DIR / "stores_encoded_df.csv",
    oil_filled_input_path = INTERIM_DATA_DIR / "oil_filled_dates_and_nulls.csv",
    holidays_events_intermin_input_path: Path = INTERIM_DATA_DIR / "holidays_events_per_city.csv",
    holiday_input_path = INTERIM_DATA_DIR / "holidays_holiday_df.csv",
    event_input_path = INTERIM_DATA_DIR / "holidays_event_df.csv",
    additional_input_path = INTERIM_DATA_DIR / "holidays_additional_df.csv",
    transfer_input_path = INTERIM_DATA_DIR / "holidays_transfer_df.csv",
    bridge_input_path = INTERIM_DATA_DIR / "holidays_bridge_df.csv",
    # Define the output paths
    train_processed_path: Path = PROCESSED_DATA_DIR / "train.csv",
    test_processed_input_path: Path = PROCESSED_DATA_DIR / "test.csv",
    train_processed_output_path = PROCESSED_DATA_DIR / "train_data.npz",
    validation_processed_output_path = PROCESSED_DATA_DIR / "validation_data.npz",
    test_npz_output_path = PROCESSED_DATA_DIR / "test_data.npz"
):

    logger.info("Generating features from dataset...")
    # Load the data from a .csv and make a copy so that the initial file wont be affected
    train_interim_df = pd.read_csv(train_interim_input_path).copy()
    logger.info("Started feature engineering for train set...")
    #Genetrate features for train set
    train_preprocessed_df = feature_engineering(train_interim_df)
    #Optimizing model by dropping unnecessary columns
    train_preprocessed_df = optimize_model(train_preprocessed_df)
    logger.info("Feature engineering for train set completed.")
    # Load the data from a .csv and make a copy so that the initial file wont be affected
    test_interim_df = pd.read_csv(test_interim_input_path).copy()
    logger.info("Started feature engineering for test set...")
    #Genetrate features for test set
    test_preprocessed_df = feature_engineering(test_interim_df)
    #Optimizing model by dropping unnecessary columns
    test_preprocessed_df = optimize_model(test_preprocessed_df)
    logger.info("Feature engineering for test set completed.")
    # Save the final DataFrame to the specified path
    train_preprocessed_df.to_csv(train_processed_path, index=False)
    # Save the final DataFrame to the specified path
    test_preprocessed_df.to_csv(test_processed_input_path, index=False)


    # Define the target and inputs
    inputs = train_preprocessed_df.drop(columns=['sales']).values  # Drop the target column and convert to numpy array
    targets = train_preprocessed_df['sales'].values  # Extract the target column as a numpy array

    # Extract the inputs (all columns except 'sales', if it exists)
    test_inputs = test_preprocessed_df.drop(columns=['sales'], errors='ignore').values  # Drop 'sales' if it exists

    # Split the data into training and validation sets (80-20 split)
    train_inputs, validation_inputs, train_targets, validation_targets = train_test_split(
        inputs, targets, test_size=0.2, random_state=42
    )

    # Save the training and validation datasets as .npz files
    np.savez(train_processed_output_path, inputs=train_inputs, targets=train_targets)
    np.savez(validation_processed_output_path, inputs=validation_inputs, targets=validation_targets)
    np.savez(test_npz_output_path, inputs=test_inputs)


    logger.success("Features generation complete.")



if __name__ == "__main__":
    #app()
    main()
