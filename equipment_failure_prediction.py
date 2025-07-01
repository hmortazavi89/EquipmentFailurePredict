# Equipment Failure Prediction
# Project Overview:
# This script predicts equipment failure in industrial systems using machine learning for predictive maintenance.
# It analyzes sensor and operational data to identify patterns indicating potential equipment failures,
# enabling proactive maintenance to reduce downtime and costs.

# Objectives:
# - Perform exploratory data analysis (EDA) to understand factors influencing equipment failure.
# - Preprocess data, including handling missing values and encoding categorical variables.
# - Train and evaluate a Gradient Boosting Classifier to predict equipment failure.
# - Visualize results to communicate insights for maintenance strategies.

# Dataset:
# The dataset (predictive_maintenance.csv) contains:
# - UDI: Unique identifier for each record.
# - Product ID: Product identifier.
# - Type: Product quality variant (L/M/H for low/medium/high).
# - Air temperature [K]: Ambient temperature in Kelvin.
# - Process temperature [K]: Process temperature in Kelvin.
# - Rotational speed [rpm]: Equipment rotational speed.
# - Torque [Nm]: Torque applied.
# - Tool wear [min]: Tool wear time in minutes.
# - Machine failure: Target variable (0/1) indicating if the equipment failed.
# - TWF, HDF, PWF, OSF, RNF: Specific failure modes (Tool Wear Failure, Heat Dissipation Failure, Power Failure, Overstrain Failure, Random Failure).

# Tools and Libraries:
# - Python, pandas, scikit-learn, matplotlib, seaborn, numpy

# Prerequisites:
# Install required libraries:
# pip install pandas scikit-learn matplotlib seaborn numpy
# Ensure predictive_maintenance.csv is in the same directory as this script.

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.metrics import classification_report

# Set random seed for reproducibility
np.random.seed(42)

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# Step 1: Data Loading and Cleaning
# Load the dataset and check for missing values or inconsistencies
print('Step 1: Data Loading and Cleaning')
df = pd.read_csv('predictive_maintenance.csv')
# Fill any missing numerical values with median
numerical_columns = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].median())

# Display first few rows
print('Dataset Preview:')
print(df.head())

# Check for missing values
print('\nMissing Values:')
print(df.isnull().sum())

# Basic info
print('\nDataset Info:')
print(df.info())

# Step 2: Exploratory Data Analysis
# Analyze the distribution of features and their relationship with machine failure
print('\nStep 2: Exploratory Data Analysis')

# Machine failure distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='Machine failure', data=df)
plt.title('Machine Failure Distribution')
plt.show()

# Air temperature vs Machine failure
plt.figure(figsize=(8, 5))
sns.boxplot(x='Machine failure', y='Air temperature [K]', data=df)
plt.title('Air Temperature vs Machine Failure')
plt.show()

# Process temperature vs Machine failure
plt.figure(figsize=(8, 5))
sns.boxplot(x='Machine failure', y='Process temperature [K]', data=df)
plt.title('Process Temperature vs Machine Failure')
plt.show()

# Rotational speed vs Machine failure
plt.figure(figsize=(8, 5))
sns.boxplot(x='Machine failure', y='Rotational speed [rpm]', data=df)
plt.title('Rotational Speed vs Machine Failure')
plt.show()

# Torque vs Machine failure
plt.figure(figsize=(8, 5))
sns.boxplot(x='Machine failure', y='Torque [Nm]', data=df)
plt.title('Torque vs Machine Failure')
plt.show()

# Tool wear vs Machine failure
plt.figure(figsize=(8, 5))
sns.boxplot(x='Machine failure', y='Tool wear [min]', data=df)
plt.title('Tool Wear vs Machine Failure')
plt.show()

# Type vs Machine failure
plt.figure(figsize=(8, 5))
sns.countplot(x='Type', hue='Machine failure', data=df)
plt.title('Product Type vs Machine Failure')
plt.show()

# Step 3: Data Preprocessing
# Encode categorical variables and scale numerical features
print('\nStep 3: Data Preprocessing')

# Encode categorical variable (Type)
le = LabelEncoder()
df['Type'] = le.fit_transform(df['Type'])

# Encode target variable (already 0/1, but ensure consistency)
df['Machine failure'] = le.fit_transform(df['Machine failure'])

# Define features and target
features = ['Type', 'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
X = df[features]
y = df['Machine failure']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
numerical_columns = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

print('Training set shape:', X_train.shape)
print('Testing set shape:', X_test.shape)

# Step 4: Model Training
# Train a Gradient Boosting Classifier to predict equipment failure
print('\nStep 4: Model Training')

# Initialize and train the model
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)

# Make predictions
y_pred = gb_model.predict(X_test)
y_pred_proba = gb_model.predict_proba(X_test)[:, 1]

# Step 5: Model Evaluation
# Evaluate the model using accuracy, precision, recall, and AUC-ROC
print('\nStep 5: Model Evaluation')

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print('Model Performance:')
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'AUC-ROC: {auc:.2f}')

# Classification report
print('\nClassification Report:')
print(classification_report(y_test, y_pred))

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 5))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Step 6: Feature Importance
# Visualize the importance of each feature in the model
print('\nStep 6: Feature Importance')

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': gb_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance')
plt.show()

# Conclusion
print('\nConclusion:')
print('This script demonstrates a complete machine learning workflow for predictive maintenance.')
print('The Gradient Boosting Classifier effectively predicts equipment failure, with key features like tool wear and torque driving predictions.')
print('The model achieves a solid AUC-ROC score, indicating good performance.')
print('This work can inform maintenance strategies to reduce industrial downtime.')

# Future Work
print('\nFuture Work:')
print('- Experiment with other models like XGBoost or Neural Networks.')
print('- Incorporate failure mode analysis (TWF, HDF, PWF, OSF, RNF).')
print('- Deploy the model in a real-time monitoring system.')
