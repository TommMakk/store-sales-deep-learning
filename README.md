# Store Sales - Deep Learning Solution

A deep learning approach for the [Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/overview) Kaggle competition.

---

## Overview

This repository contains a reproducible pipeline for forecasting store sales using deep learning. The solution leverages modern data science best practices and a modular codebase, following the [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/) project structure.

The goal is to predict daily sales for a large Ecuadorian grocery retailer, using historical sales data, promotions, oil prices, holidays, and store information.

---

## Project Structure

```
├── LICENSE
├── Makefile
├── README.md
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- Documentation and project reports.
├── models             <- Trained and serialized models, model predictions, or model summaries.
├── notebooks          <- Jupyter notebooks for exploration and analysis.
├── pyproject.toml     <- Project configuration and metadata.
├── references         <- Data dictionaries, manuals, and explanatory materials.
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures for reporting.
├── requirements.txt   <- Python dependencies for the project.
├── setup.cfg          <- Configuration for code style tools.
└── store-sales-DL     <- Source code for this project.
    ├── __init__.py
    ├── config.py               <- Project configuration variables.
    ├── dataset.py              <- Data loading and preprocessing scripts.
    ├── features.py             <- Feature engineering code.
    ├── modeling
    │   ├── __init__.py
    │   ├── predict.py          <- Model inference code.
    │   └── train.py            <- Model training code.
    └── plots.py                <- Visualization code.
```

---

## Getting Started

### 1. Download Data

- Register and download the competition data from [Kaggle Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data).
- Place the raw CSV files in the `data/raw/` directory.

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Data Preparation

Run the data processing pipeline to generate interim datasets:

```bash
python store-sales-DL/dataset.py
```

<details> <summary><strong>Alternatively, use Make:</strong></summary>

```bash
make dataset
```
</details>
Next, generate the full set of modeling features and processed datasets:

```bash
python store-sales-DL/features.py
```
The `features.py` script is responsible for advanced feature engineering and assembling the final processed datasets used for model training and evaluation.
<details> <summary><strong>Alternatively, use Make:</strong></summary>

```bash
make features
```
</details>

### 4. Model Training

Train the deep learning model:

```bash
python store-sales-DL/modeling/train.py
```
<details> <summary><strong>Alternatively, use Make:</strong></summary>

```bash
make train
```
</details>

### 5. Model Inference

Generate predictions using the trained model:

```bash
python store-sales-DL/modeling/predict.py
```
<details> <summary><strong>Alternatively, use Make:</strong></summary>

```bash
make predict
```
</details>

---
## Deep Learning Approach

### Methodology

Our solution leverages a robust deep learning pipeline tailored for time series forecasting in the retail domain. The key steps include:

- **Data Preprocessing & Cleaning:**  
  Raw sales, oil, holiday, and store data are merged, missing values are handled, and categorical variables are encoded using one-hot encoding.

- **Feature Engineering:**  
  We generate lag-based features, rolling statistics, and calendar-based features (e.g., day of week, holidays, paydays) to capture temporal patterns and external influences on sales.

- **Exploratory Data Analysis (EDA):**  
  Data distributions, trends, and correlations are visualized to inform feature selection and model design.

- **Model Development:**  
  Our primary model is a feedforward neural network implemented in Keras/TensorFlow. The architecture consists of multiple dense layers with ReLU activations and dropout regularization to prevent overfitting.

- **Training Strategy:**  
  We use the Adam optimizer with a learning rate of 0.001. Early stopping is employed to halt training when validation loss stops improving, ensuring optimal generalization.

- **Evaluation:**  
  Model performance is monitored using Mean Squared Error (MSE) and Mean Absolute Error (MAE) on a validation set. The final model is saved and used for inference on the test set.

### Model Architecture

- **Input Layer:** Accepts engineered features for each store-date-family combination.
- **Hidden Layers:** Two dense layers (96 and 64 units) with ReLU activation and dropout.
- **Output Layer:** Single neuron with linear activation for sales prediction.

### Training and Optimization

- **Optimizer:** Adam (`learning_rate=0.001`)
- **Loss Function:** Mean Squared Error (MSE)
- **Batch Size:** 64
- **Epochs:** Up to 20, with early stopping (patience=4)
- **Regularization:** Dropout (rate=0.1) to reduce overfitting

### Reproducibility & Automation

- All steps, from data preparation to model training and inference, are automated via modular scripts.
- Model checkpoints and logs are saved for transparency and reproducibility.

---

This deep learning approach is designed to capture both short-term and long-term sales patterns, leveraging rich feature engineering and modern neural network techniques to deliver accurate forecasts for the Kaggle Store Sales competition.

---

## Conclusion

This deep learning solution achieved a public leaderboard score of **0.58356** on the [Kaggle Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/overview) competition (lower is better).

**Key factors for strong predictive performance:**
- Inclusion of **lag features** (previous sales), which capture temporal dependencies and seasonality.
- Use of **promotion indicators**, reflecting the significant impact of promotions on sales.

**Real-world business applications of accurate sales forecasting:**
- **Inventory management:** Reduce overstock and stockouts.
- **Food waste reduction:** Better match supply with demand, especially for perishable goods.
- **Supply chain optimization:** Improve ordering and logistics planning.
- **Staffing and resource planning:** Align workforce and resources with expected demand.
- **Promotional planning:** Evaluate and optimize the impact of marketing campaigns.

---

## Future Improvements

While the current deep learning approach provides strong predictive performance, there are several avenues for further improvement:

- **Advanced Architectures:**  
  - Explore recurrent neural networks (RNNs), Long Short-Term Memory (LSTM) networks, or Temporal Convolutional Networks (TCNs) to better capture sequential dependencies in time-series data.
- **Hybrid Models:**  
  - Combine deep learning with tree-based models (e.g., XGBoost, LightGBM) or statistical models (e.g., ARIMA, Prophet) to leverage the strengths of each approach.
- **Feature Enrichment:**  
  - Incorporate additional external data sources (e.g., weather, macroeconomic indicators).
  - Experiment with automated feature selection or embedding layers for categorical variables.
- **Hyperparameter Optimization:**  
  - Use automated tools (e.g., Optuna, Keras Tuner) for more thorough hyperparameter search.
- **Cross-Validation:**  
  - Implement time-series cross-validation (e.g., rolling or expanding window) to better estimate model generalization.
- **Ensembling:**  
  - Blend predictions from multiple models to reduce variance and improve robustness.

### Model Selection for Time-Series

- Deep learning models are powerful and flexible, but can be overkill for many time-series forecasting problems, especially when data is limited or patterns are well-captured by simpler models.
- **Classical approaches** such as ARIMA, SARIMA, Exponential Smoothing, or tree-based models like XGBoost often provide competitive or superior results with less computational overhead and easier interpretability.
- For many business applications, these models may be preferable unless the dataset is large, highly complex, or contains significant nonlinearities that deep learning can uniquely exploit.

---

## References

- [Kaggle Competition Overview](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/overview)
- [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/)

---

## License

This project is open source and available under the [MIT License](LICENSE).

---

[![CCDS Project template](https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter)](https://cookiecutter-data-science.drivendata.org/)