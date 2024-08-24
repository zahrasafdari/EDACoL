# EDACoL: An Enhanced Dual-Attention ConvLSTM Model 

Welcome to the Polar Sea Ice Forecasting project! This repository contains a deep learning-based solution for Polar Sea Ice forecasting, leveraging advanced neural network architectures to make accurate predictions. The following guide will help you understand how to get started, how the model is structured, and how to use the provided code effectively.

This project builds upon the primary sea ice prediction model developed by the Big Data Lab at UMBC, known as "[Sea Ice Forecasting using Attention-based Ensemble LSTM](https://github.com/big-data-lab-umbc/sea-ice-prediction/tree/main/climate-change-ai-workshop)" My work aims to enhance this foundational model by incorporating advanced techniques, seeking to improve its predictive accuracy and overall performance.

## Model Architecture

The sea ice prediction model is designed using a hybrid architecture that combines convolutional layers, attention mechanisms, and LSTM networks. Below is a concise description of the model components:

### 1. Inputs
- **Model 1 Input**: Processes 30 days of daily climate data.
- **Model 2 Input**: Processes 1 day of monthly climate data.

### 2. Model 1: Daily Data Processing
- **Conv1D Layer**: Extracts spatial features from the daily data.
- **Multi-Head Attention Layer**: Captures long-range dependencies within the daily data sequence.
- **LSTM Layers**: Learn and retain temporal patterns across the daily data.
- **Global Average Pooling**: Reduces the LSTM output to a fixed-size vector.
- **Dense Layer**: Refines the extracted features before output.

### 3. Model 2: Monthly Data Processing
- **Conv1D, Multi-Head Attention, LSTM, Global Average Pooling, and Dense Layers**: Structured similarly to Model 1 but adapted for the monthly data input.

### 4. Ensemble
- **Concatenation**: Combines the outputs from Model 1 and Model 2.
- **Dense Layers**: Further process the combined data to generate the final prediction.

### 5. Model Compilation
- **Optimizer**: Adam optimizer is used for training.
- **Loss Function**: Mean Squared Error (MSE) is used to evaluate prediction accuracy.

This architecture is designed to integrate both daily and monthly climate data, effectively capturing spatial and temporal patterns for accurate sea ice concentration prediction.

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




## Contact

For any questions or support, please reach out to [zahrasafdari8181@gmail.com](mailto:zahrasafdari8181@gmail.com).

---

Thank you for exploring this project!
