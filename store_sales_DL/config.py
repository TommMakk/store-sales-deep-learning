from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
import os

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
#PROJ_ROOT = "/home/user/Repos/store_sales/store_sales"
#PROJ_ROOT = Path(os.getenv("project_dir"))
PROJ_ROOT = Path(__file__).resolve().parents[1]
#print(PROJ_ROOT)
DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
MODELS_DIR = PROJ_ROOT / "models"
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"



def drop_columns(df: pd.DataFrame, columns_to_drop: list) -> pd.DataFrame:
    """
    Drops specified columns from a DataFrame in a structured way.

    Parameters:
    - df (pd.DataFrame): The DataFrame from which columns will be dropped.
    - columns_to_drop (list): List of column names to drop.

    Returns:
    - pd.DataFrame: A new DataFrame with the specified columns removed.
    """

    logger.info("Started dropping columns...")

    # Check if all columns to drop exist in the DataFrame
    missing_columns = [col for col in columns_to_drop if col not in df.columns]
    if missing_columns:
        print(f"Warning: The following columns are not in the DataFrame and will be ignored: {missing_columns}")
    
    # Drop the columns
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    
    logger.info("Completed dropping columns...")
    # Return the updated DataFrame
    return df

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
