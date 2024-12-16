# HR Analysis: Predicting Employee Promotions

This repository contains a project that analyzes HR data to predict employee promotions using machine learning models: **Random Forest Classifier** and **LightGBM Classifier**.

## Files

- **train.csv**: The row data (Consists of 14 columns).
- **HR_analysis_RandomForestClassifier.ipynb**: The model built using the Random Forest classifier, including training and evaluation. Additionally, there is a visualization of each column.
- **HR_Analysis_LGBMClassifier.ipynb**: The model built using the LightGBM (LGBM) classifier, including training and evaluation.

## Objective

To build predictive models that can help HR departments identify potential candidates for promotion based on various employee attributes.

## Steps

1. **Data Preprocessing**: 
   - Handled missing data using imputation.
   - Categorical features were encoded.
   - Prepared the dataset for model training.
   
2. **Model Building**:
   - **Random Forest Classifier**: Trained a random forest model for promotion prediction.
   - **LightGBM Classifier**: Trained a light gradient boosting model.
   
3. **Evaluation**:
   - Evaluated both models using confusion matrix, classification report, and accuracy score.
   - Both models achieved similar performance with slight differences in class prediction.

## Results

- **Random Forest Classifier**: 
   - Accuracy: 93%
   - Confusion Matrix: [[9843  211], [ 574  334]]
   - Struggled with class imbalance, leading to lower recall for promoted employees.
   
- **LightGBM Classifier**: 
   - Accuracy: 94%
   - Confusion Matrix: [[10034    20], [  593   315]]
   - Slightly better than Random Forest but still struggled with class imbalance.

## Future Work

- **Class Imbalance Handling**: Explore techniques like SMOTE to improve model performance for predicting promoted employees.
- **Hyperparameter Optimization**: Fine-tune model parameters for better accuracy.

## Setup and Usage:
This project is designed to be used in Google Colab. To run the project, follow these steps:

1. Open a new notebook in Google Colab.
2. Upload the project files to the Colab environment.
3. Execute the files **train.csv**, **Hr analysis RandomForestClassifier**, and **Hr analysis LGBMClassifier** in the given order.
4. Use the **Confusion Matrix** and **Classification Report** to evaluate the performance of each model.

## Additional Information:
Both the Random Forest and LGBM classifiers were used in this project. Various hyperparameters were tested for each model to achieve the best possible results. Each model's performance was analyzed using metrics like precision, recall, and F1-score, with an emphasis on understanding how each model handles the classification problem.
