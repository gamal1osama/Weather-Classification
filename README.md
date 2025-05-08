# Weather Data Classification Project

## 🔗 Live Notebook

You can explore the full code and outputs in the [Kaggle Notebook](https://www.kaggle.com/code/gamalosama/weather-classification).

---

## Overview

This project focuses on classifying weather types (Sunny, Cloudy, Rainy, Snowy) using a dataset with various weather-related features. It involves data exploration, cleaning, preprocessing, and model building to predict the weather type accurately.

## Data

The dataset (`weather data classification.csv`) contains the following features:

- **Numerical Features:**
  - Temperature (C)
  - Humidity (%)
  - Wind Speed (km/h)
  - Precipitation (%)
  - Atmospheric Pressure (hPa)
  - UV Index
  - Visibility (km)
- **Categorical Features:**
  - Cloud Cover
  - Season
  - Location

- **Target Variable:**
  - Weather Type (Sunny, Cloudy, Rainy, Snowy)

## Steps

1. **Data Exploration:**
   - Understand the dataset's structure, dimensions, and data types.
   - Identify missing values and potential outliers.
   - Explore the distributions of numerical and categorical features.
2. **Data Cleaning:**
   - Handle missing values (e.g., using imputation).
   - Clean and standardize data formats (e.g., wind speed units).
   - Address outliers (e.g., using IQR-based removal).
3. **Exploratory Data Analysis (EDA):**
   - Univariate analysis to visualize individual feature distributions.
   - Bivariate analysis to explore relationships between features and the target variable.
   - Multivariate analysis (correlation matrix, pair plots) to understand feature interactions.
   - Seasonal and location-based analysis to identify patterns.
4. **Preprocessing:**
   - Split the data into training and testing sets.
   - Encode categorical features (e.g., using one-hot encoding).
   - Scale numerical features (e.g., using standardization).
5. **Modeling:**
   - Train various classification models (e.g., Random Forest, XGBoost, etc.).
   - Evaluate model performance using cross-validation and metrics like accuracy.
   - Fine-tune hyperparameters using techniques like grid search or randomized search.
6. **Evaluation:**
   - Evaluate the best model on the test set.
   - Analyze the confusion matrix and classification report.
   - Visualize model performance using precision-recall curves and ROC curves.

## Results

- **Best Model:** RandomForestClassifier  
- **Accuracy:** [Insert the final accuracy score here]  
- **Key Insights:** [Summarize any key findings from the EDA and model results]  

## Libraries Used

- pandas  
- numpy  
- matplotlib  
- seaborn  
- scikit-learn  
- xgboost  
- lightgbm  
- catboost  

## Future Work

- Explore more advanced feature engineering techniques.  
- Experiment with other classification algorithms.  
- Deploy the model for real-time weather prediction.  

## Contributing

Feel free to contribute to this project by:

- Improving the data cleaning or preprocessing steps.
- Trying different models or hyperparameter tuning strategies.
- Adding new features or datasets.


