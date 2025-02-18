# Multinomial Logistic Regression with scikit-learn

## Project Overview

This project explores Multinomial Logistic Regression for multiclass classification while trying to address severe class imbalance. The objective is to evaluate different techniques, like class weighting, oversampling with SMOTE, One-vs-Rest (OvR), and Bayesian Optimization, on a weather dataset with three target classes:   
- No Precipitation
- Rain
- Snowfall

## Data
The dataset consists of historical weather parameters obtained via the Open-Meteo API and will serve as the foundation for evaluating different ML algorithms in a multi-class classification setting.


📂 Project Structure

Each notebook focuses on a specific aspect of the study:

✅ 01 - Introduction   
    Overview of the problem, objectives, and dataset understanding.   
✅ 02 - Basic EDA   
    Exploratory Data Analysis (EDA), class distribution, and feature analysis.   
✅ 03 - Logistic Regression Essentials   
    Theoretical background on multinomial logistic regression.   
✅ 04 - Standard Logistic Regression   
    Baseline model without imbalance handling.   
✅ 05 - Weighted Logistic Regression   
    Using class_weight to address class imbalance.   
✅ 06 - SMOTE Logistic Regression   
    Applying Synthetic Minority Oversampling Technique (SMOTE) for data balancing.   
✅ 07 - OvR Logistic Regression   
    Handling multiclass classification via One-vs-Rest (OvR) strategy.   
✅ 08 - Bayesian Optimization   
    Fine-tuning hyperparameters with Bayesian Optimization (BayesSearchCV).   
✅ 09 - Model Comparison & Final Thoughts   
    Evaluation, confusion matrices, conclusions, and future work.

## Key Findings

A consistent trade-off between precision and recall across all logistic regression iterations was observed and finaly
all failed to handle class imbalance in the dataset efficiently.   
While all class balancing techniques increased the model’s sensitivity to minority classes, they did so at the cost 
of introducing more misclassifications in the majority class and therefore reducing the precision.

This could be due to several factors:

- Complex, potentially non-linear relationships among the predictors that Logistic Regression fails to capture.
- Severe class imbalance, which continues to impact performance despite various balancing strategies.
- The algorithm may have reached its full potential given the constraints of this dataset.

## Model Performance (Confusion Matrices)

Below the confusion matrices summarizing the performance of each model:

![Confusion Matrix](images/confusion_matrices_plot.png)

## Dependencies

- `scikit-learn`
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `imblearn`
- `bayesian-optimization` (for BayesSearchCV)

## Future Work

Future exploration will include more advanced classification algorithms, such as Random Forests, Support Vector Machines (SVM), and Neural Networks, to further address class imbalance and improve predictive accuracy if possible.

## Author

This project was developed by Nikos Papakostas. Feedback and contributions are welcome!

---
