from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

from store_sales_DL.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, PROJ_ROOT
import os

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR / "oil.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    # ----------------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Processing dataset...")
    A = Path(__file__).resolve().parents[1]
    logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info(RAW_DATA_DIR)
            logger.info("Something happened for iteration 5.")
    logger.success("Processing dataset complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
