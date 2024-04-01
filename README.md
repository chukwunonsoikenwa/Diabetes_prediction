# Diabetes Prediction Web App

## Overview
This repository contains a machine learning model and web application for predicting diabetes in female patients based on various factors such as age, no of pregnancies, BMI, Glucose level, Insulin level, Blood pressure, Skin thickness, and Diabetes prediction function. The model is trained on a dataset of diabetes records from a healthcare center and aims to provide accurate diabetes prediction via a simple web app for non-technical users.

## Dataset
The dataset used for training and testing the model is included in the `data` directory. It consists of a CSV file named `insurance.numbers.csv`, containing the following columns:
- `Pregnancies`: No of pregnancies the individual has had including past and present.
- `Glucose`: Level of glucose presently.
- `Blood Pressure`: The individual's current blood pressure.
-  `Skin thickness`: Skin thickness of the individual.
- `BMI`: The individual's Body Mass Index (BMI).
- `Insulin level`: Level of insulin in an individual.
- `Diabetes prediction function`: Age of the individual in years.
- `Age`: Age of the individual in years.


## Model Development
The model development process involves several steps, including data preprocessing, feature engineering, model selection, training, evaluation, and optimization. Here's a brief overview of each step:

### Data Preprocessing
- Handle missing values: Check for missing values in the dataset and either impute or remove them.
- Scale numerical features: Normalize or standardize numerical features (e.g., age, BMI) to ensure uniformity in their scales.

### Feature Engineering
- Explore feature distributions: Analyze the distribution of features and identify any outliers or anomalies.
- Create new features: Derive additional features, such as interaction terms or polynomial features, to capture complex relationships if necessary.

### Model Selection
- Choose algorithms: Select suitable machine learning algorithms for regression tasks, such as linear regression, decision trees, or ensemble methods.
- Cross-validation: Perform cross-validation to assess the performance of each model and mitigate overfitting.

### Model Training and Evaluation
- Train-test split: Split the dataset into training and testing sets to evaluate model generalization.
- Train models: Train multiple regression models on the training data using different algorithms.
- Evaluate performance: Evaluate each model's performance on the test set using appropriate metrics such as mean absolute error (MAE), mean squared error (MSE), or R-squared.

### Model Optimization
- Hyperparameter tuning: Fine-tune model hyperparameters using grid or randomized search techniques to improve performance.
- Regularization: Apply regularization techniques (e.g., L1 or L2 regularization) to prevent overfitting and enhance model robustness.

## Model Deployment
Once the model is trained and optimized, it can be deployed in various ways, including:
- Integrating into a web application or mobile app for real-time predictions.
- Exposing as an API endpoint for remote inference.
- Packaging as a standalone application for local use.
For this project, I deployed this model in form of a web application using Streamlit.

# Steps in model deployment using Streamlit
1. Create a Python virtual environment
2. Run Spyder
3. import dependencies and libraries (Numpy, Pickle, Streamlit)
4. Streamlit run on your terminal

   
## Requirements
Ensure you have the following dependencies installed to run the code:
- Python 3.10
- Jupyter Notebook
- Scikit-learn
- Pandas
- NumPy
- Pickle
- Streamlit
  
## Findings
Based on the model prediction, Older females with high blood levels tend to be diabetic and the number of pregnancies has little or no effect on diabetes diagnosis in a female. 

## Conclusion
SVM was used and the accuracy score for this model was 79%. 


## Usage
To use the model:
1. Clone this repository to your local machine.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Open the Jupyter Notebook `diabetes.ipynb`.
4. Follow the instructions in the notebook to preprocess the data, train the model, and make predictions.

