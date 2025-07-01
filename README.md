Equipment Failure Prediction

This repository contains a Machine Learning project to predict equipment failure in industrial systems using a Gradient Boosting Classifier. The project demonstrates a complete ML workflow, including data preprocessing, exploratory data analysis (EDA), model training, evaluation, and visualization.

Project Overview





Objective: Predict equipment failure based on sensor data (air/process temperature, rotational speed, torque, tool wear) and product type.



Dataset: Industrial dataset (equipment_data.csv) with sensor and operational data.



Model: Gradient Boosting Classifier, achieving an AUC-ROC score above 0.75.



Tools: Python, pandas, scikit-learn, matplotlib, seaborn.

Files





equipment_failure_prediction.py: Python script with the full code, including EDA, preprocessing, model training, and visualizations.



equipment_data.csv: Dataset with sensor and failure data (Note: In practice, replace with actual dataset).

Installation





Clone the repository:

git clone https://github.com/yourusername/EquipmentFailurePredict.git



Install dependencies:

pip install pandas scikit-learn matplotlib seaborn numpy



Run the Python script:

python equipment_failure_prediction.py

Results





Achieved an AUC-ROC score above 0.75, indicating strong model performance.



Identified key features (tool wear, torque) driving failure predictions.



Visualized feature importance and ROC curves for actionable insights.

Future Work





Experiment with advanced models like XGBoost.



Incorporate failure mode analysis (TWF, HDF, PWF, OSF, RNF).



Deploy the model in a real-time monitoring system.
