# Advanced Predictive Modeling using CNN-LSTM with Attention Layers

Welcome to the Polar Sea Ice Forecasting project! This repository contains a deep learning-based solution for Polar Sea Ice forecasting, leveraging advanced neural network architectures to make accurate predictions. The following guide will help you understand how to get started, how the model is structured, and how to use the provided code effectively.

This project builds upon the primary sea ice prediction model developed by the Big Data Lab at UMBC, known as "[Sea Ice Forecasting using Attention-based Ensemble LSTM](https://github.com/big-data-lab-umbc/sea-ice-prediction/tree/main/climate-change-ai-workshop)" My work aims to enhance this foundational model by incorporating advanced techniques, seeking to improve its predictive accuracy and overall performance.

## Key Features

- **Data Handling:** Efficient loading and preprocessing of time series data.
- **Model Architecture:** A hybrid model combining CNN, LSTM, and Transformer layers for improved forecasting performance.
- **Training and Evaluation:** Includes callbacks for early stopping and saving the best model to prevent overfitting.
- **Visualization:** Plots for training history and comparison of predictions with actual values.

## Getting Started

To use this project, follow these steps:

### 1. Prerequisites

Ensure you have the following Python libraries installed:
- `numpy`
- `pandas`
- `matplotlib`
- `tensorflow`
- `scikit-learn`

You can install these dependencies using pip:

```bash
pip install numpy pandas matplotlib tensorflow scikit-learn
```

### 2. Data Preparation

The data used in this project is also available at the [Big Data Lab repository](https://github.com/big-data-lab-umbc/sea-ice-prediction/tree/main/climate-change-ai-workshop/data).
Ensure your data files are in the `data` directory:
- `dailyt30_features.npy`
- `monthly_features.npy`
- `dailyt30_target.npy`
- `monthly_target.npy`

The data files should be preprocessed as described in the code. The script will automatically handle loading and preprocessing.

### 3. Model Training

The model is defined in the `build_model` function, which creates a hybrid architecture of CNN, LSTM, and Transformer blocks. The training process involves:

- **Compiling** the model with the Adam optimizer and mean squared error loss.
- **Fitting** the model with early stopping and model checkpoint callbacks to save the best-performing model.


### 4. Evaluation and Visualization

After training, the script evaluates the model's performance and generates visualizations:

- **Training History:** Plots showing the loss curves for both training and validation sets.
- **Predictions vs Actuals:** Scatter plots comparing predicted values with actual observations.

## Code Structure

- **Data Loading and Preprocessing:** Handles data loading, scaling, and reshaping.
- **Model Definition:** Builds a hybrid model with CNN, LSTM, and Transformer layers.
- **Training and Callbacks:** Includes early stopping and model checkpoint mechanisms.
- **Visualization:** Plots training history and prediction accuracy.


## Results

## Model Performance Comparison

| Models                                                                 | Test RMSE         | Test NRMSE       | Test R² Score |
|------------------------------------------------------------------------|--------------------|------------------|---------------|
| Advanced Predictive Modeling using CNN-LSTM with Attention Layers    | 787,587.23        | 0.0651           | 0.9417        |
| Sea Ice Forecasting using Attention-based Ensemble LSTM                | 818,216.64        | 0.0744           | 0.9400        |





## Contact

For any questions or support, please reach out to [zahrasafdari8181@gmail.com](mailto:zahrasafdari8181@gmail.com).

---

Thank you for exploring this project!
