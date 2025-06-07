# Data Preparation

This section describes the steps taken to prepare the data for the Store Sales - Deep Learning Solution project.

## 1. Raw Data Collection

- Download the competition data from [Kaggle Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data).
- Place all raw CSV files in the `data/raw/` directory.

## 2. Data Processing Pipeline

The data processing pipeline is responsible for:
- Loading raw data files (sales, stores, oil, holidays, etc.).
- Merging datasets to create a unified view.
- Handling missing values and correcting data types.
- Saving interim datasets to `data/interim/` for further processing.

You can run the data processing pipeline with:

python store-sales-DL/dataset.py

or, using the Makefile:

make dataset

## 3. Output

- The processed interim datasets are saved in the `data/interim/` directory.
- These datasets are used as input for the feature engineering step.

## 4. Notes

- Ensure all raw data files are present and named correctly before running the pipeline.
- Review the logs/output for any warnings or errors during processing.

---

For more details on feature engineering, see the [Feature Engineering](feature_engineering.md) section.