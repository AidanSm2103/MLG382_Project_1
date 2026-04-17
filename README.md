# MLG382_Project_1

📌 Background

BC Analytics is a health-tech startup that partners with clinics, insurers, and wellness programs to improve patient outcomes and reduce long-term healthcare costs.

One of the key challenges is the early identification and management of diabetes risk. Currently, healthcare providers often:
  - React too late to warning signs
  - Lack visibility into lifestyle drivers of risk
  - Do not have tools for data-driven decision-making during consultations

🎯 Project Objective

This project delivers a data-driven decision support system that:
  - Predicts patient diabetes risk (classification)
  - Identifies key lifestyle factors influencing risk
  - Segments patients into meaningful groups
  - Provides actionable recommendations via a web dashboard
  
🧠 Project Framework – CRISP-DM
1. Business Understanding
    - The goal is to assist healthcare professionals in identifying diabetes risk early and recommending interventions.

2. Data Understanding
    - Lifestyle factors (activity, diet, alcohol)
    - Health indicators (BMI, blood pressure, history)
    - Demographics

4. Data Preparation
    - Missing values handled
    - Categorical variables encoded 
    - Feature scaling applied 
    - Class imbalance handled using SMOTE
    - Feature set aligned and stored for deployment

6. Modeling
  - Risk Classification (Primary Task)
  - Models used:
      - Decision Tree
      - Random Forest
      - XGBoost (final selected model)
  
  - Evaluation metrics:
    - Accuracy
    - F1 Score (weighted)
  
  - Patient Segmentation
      - Algorithm: K-Means (k = 3)
      - Groups patients into:
      - Low-risk lifestyle group
      - Moderate-risk group
      - High-risk group
  
  - Key Driver Analysis
    - Method: SHAP (Shapley Additive Explanations)
    
  - Identifies key contributors to diabetes risk such as:
    - BMI
    - Physical activity
    - Lifestyle indicators

5. Evaluation
    - Cross-validation used for model comparison
    - XGBoost and Random Forest showed best performance
    - Clustering evaluated using Silhouette Score

7. Deployment
    - A Dash web application was built and deployed to provide:
      - Risk prediction
      - Patient segmentation
      - Lifestyle recommendations
  
🌐 Web Application
  - Features:
      - Input patient data (BMI, Age, Activity)
      - Predict diabetes risk level
      - Assign patient to a cluster
      - Generate recommendations

🛠️ Tech Stack
    - Python
    - Pandas, NumPy
    - Scikit-learn
    - XGBoost
    - SHAP
    - Dash (for web app)
    - Joblib (model persistence)

