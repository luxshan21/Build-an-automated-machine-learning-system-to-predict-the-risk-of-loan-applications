# Automated Machine Learning System for Loan Risk Prediction

## 1. Introduction
In designing an automated machine learning (ML) system for production, the goal is to create a scalable, maintainable, and efficient ML pipeline. This system enables automated data preprocessing, model training, evaluation, and deployment, ensuring the model remains effective and adapts to new data over time.

## 2. System Overview
This project is structured as an end-to-end ML pipeline with the following stages:

![image](https://github.com/user-attachments/assets/2ffcb8f9-9c98-4399-8edc-f468d5d36fc3)

### A. Data Preprocessing
1. **Combine Datasets** - Use the join method to merge datasets into a single comprehensive dataset.
2. **Select Target and Input Features** - Identify the target column and relevant input features from the combined dataset.
3. **Encode Categorical Columns** - Apply label encoding to convert categorical variables into numerical form.
4. **Split Data** - Divide the dataset into training (80%) and testing (20%) sets.
5. **Impute Missing Values** - Fill missing values using mean values.

### B. Feature Engineering
1. **Calculate Mutual Information** - Measure the mutual information between the target variable and each feature.
2. **Set Feature Selection Threshold** - Use a threshold value (e.g., 0.05) to select the top features based on mutual information scores.
3. **Save Preprocessing Artifacts** - Store encoder mappings, imputed values, and selected top features for deployment.

### C. Model Selection and Training
1. **Choose LightGBM Model** - Use `LGBMClassifier` and configure model parameters.
2. **Hyperparameter Tuning** - Perform hyperparameter tuning using `RandomizedSearchCV`.
3. **Cross-Validation** - Validate model performance with cross-validation.
4. **Get Best Model** - Retrieve the model with optimal parameters.

### D. Model Evaluation
1. **Evaluate with AUC** - Assess model performance on the test set using the Area Under the Curve (AUC) metric.
2. **Plot Training and Validation Performance** - Visualize the model’s performance during training and validation.
3. **Plot Confusion Matrix** - Generate a confusion matrix to analyze classification outcomes.
4. **Generate Classification Report** - Produce a detailed classification report.

### E. Save the Model for Deployment
1. **Save the Trained Model** - Store the trained model for future deployment and use.

### F. Model Deployment
1. **Load Saved Model** - Load the trained LightGBM model.
2. **Load Deployment Dataset** - Load the dataset prepared for deployment testing.
3. **Load Encoders and Imputer Values** - Retrieve saved encoder mappings and imputed mean values.
4. **Load Selected Features** - Load the top features identified during feature selection.
5. **Apply Preprocessing** - Use saved preprocessing values to prepare the new dataset.
6. **Predict Model Accuracy** - Evaluate the model's accuracy on the new dataset.




