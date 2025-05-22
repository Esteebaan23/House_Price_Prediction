# House Price Prediction using Feature Engineering and Machine Learning

This repository contains an end-to-end pipeline for predicting house prices by combining traditional machine learning models with advanced feature engineering techniques.

## Overview

The project starts with a raw housing dataset and performs thorough data cleaning including:
- **Outlier Removal:** Eliminating extreme values to reduce noise.
- **Missing Value Imputation:** Filling gaps based on statistical measures (mean or mode).
- **Categorical Encoding:** Transforming categorical features using techniques like Ordinal and One-Hot Encoding.

Subsequently, feature creation and selection were applied to construct a well-structured dataset, optimized for training predictive models.

## Models and Results

Two models were developed and evaluated:

- **XGBoost:**  
  - **RMSE:** Reduced from \$30,369.21 to \$20,777.14  
  - **R²:** Increased from 87.98% to 92.18%

- **Random Forest:**  
  - **RMSE:** Reduced from \$29,009.37 to \$22,006.05  
  - **R²:** Increased from 89.03% to 91.23%

These results demonstrate the significant impact of feature engineering on enhancing model performance by reducing prediction errors and increasing accuracy.



