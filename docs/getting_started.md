# Getting Started

This section provides step-by-step instructions to set up the Store Sales - Deep Learning Solution project on your local machine.

## 1. Clone the Repository

Clone the project from GitHub:

git clone https://github.com/yourusername/store-sales-DL.git
cd store-sales-DL

## 2. Set Up a Python Environment

It is recommended to use a virtual environment:

python3 -m venv venv
source venv/bin/activate

## 3. Install Dependencies

Install the required Python packages:

pip install -r requirements.txt

## 4. Download the Data

- Register and download the competition data from [Kaggle Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data).
- Place all raw CSV files in the `data/raw/` directory.

## 5. Run the Pipeline

You can run each step of the pipeline using Python scripts or the Makefile:

- Data Preparation:  
  python store-sales-DL/dataset.py  
  or  
  make dataset

- Feature Engineering:  
  python store-sales-DL/features.py  
  or  
  make features

- Model Training:  
  python store-sales-DL/modeling/train.py  
  or  
  make train

- Model Inference:  
  python store-sales-DL/modeling/predict.py  
  or  
  make predict

---

For more details on each step, refer to the corresponding sections in this documentation.