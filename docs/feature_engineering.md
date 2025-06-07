# Feature Engineering

This section describes the feature engineering process used to enhance the predictive power of the deep learning model for store sales forecasting.

## Overview

Feature engineering transforms raw data into meaningful inputs for the model. In this project, several types of features were created to capture temporal patterns, external influences, and categorical relationships.

## Types of Features

- **Lag Features:**  
  Previous sales values (e.g., sales from 1, 7, 14, or 28 days ago) are included to help the model recognize trends and seasonality.

- **Rolling Statistics:**  
  Rolling means and standard deviations over various windows (e.g., 7-day, 14-day) provide information about recent sales trends and volatility.

- **Promotions:**  
  Binary indicators for whether a product was on promotion, as well as rolling counts of recent promotions, are included to capture the impact of marketing activities.

- **Calendar Features:**  
  Day of week, month, year, holidays, and paydays are encoded to help the model learn periodic patterns and the effects of special dates.

- **Store and Product Metadata:**  
  Store type, location, and product family are encoded using one-hot or label encoding to provide context about each sale.

- **External Data:**  
  Oil prices and holiday events are merged with the main dataset to account for macroeconomic and external factors.

## Feature Engineering Pipeline

The feature engineering process is automated in the `features.py` script. This script:

- Loads interim datasets from the data processing step.
- Creates and merges all engineered features.
- Handles missing values and encodes categorical variables.
- Outputs the final processed datasets to `data/processed/` for model training and evaluation.

You can run the feature engineering pipeline with:

python store-sales-DL/features.py

or, using the Makefile:

make features

---

Well-designed features are crucial for improving model accuracy and capturing the complex dynamics of retail sales.