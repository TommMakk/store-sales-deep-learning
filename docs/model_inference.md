# Model Inference

This section explains how to use the trained deep learning model to generate sales predictions for the Store Sales - Deep Learning Solution project.

## Overview

Model inference involves loading the best trained model and applying it to new or unseen data (such as the test set) to generate sales forecasts. This step is essential for evaluating model performance and submitting predictions to the competition.

## Steps for Model Inference

1. **Prepare the Test Data**

   - Ensure that the test data has been processed and features have been engineered in the same way as the training data.
   - The processed test dataset should be located in the appropriate directory (e.g., `data/processed/`).

2. **Run the Inference Script**

   You can generate predictions using the provided script:

   python store-sales-DL/modeling/predict.py

   or, using the Makefile:

   make predict

3. **Output**

   - The script will load the best saved model (e.g., `models/best_model.keras`).
   - Predictions for the test set will be generated and saved, typically as a CSV file in the `models/` or `data/processed/` directory.
   - The output file can be used for submission to Kaggle or further analysis.

## Notes

- Ensure that the model and test data paths in the script match your project structure.
- Review the logs/output for any errors or warnings during inference.
- You can modify the inference script to adjust output format or post-processing as needed.

---

For more details on model training, see the [Model Training](model_training.md) section.