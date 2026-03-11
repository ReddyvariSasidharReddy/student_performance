# Student Performance Prediction System
A machine learning project that predicts students' math scores using demographic and academic features with 89% accuracy (Gradient Boosting model: RMSE 6.21, R² 0.891).

Project Details
Dataset: 1000 students from Kaggle with features - Gender, Race/Ethnicity, Parental Education, Lunch Type, Test Preparation Course, Reading Score, Writing Score (Target: Math Score)

Technologies: Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn (Linear Regression, Random Forest, Gradient Boosting), Pickle/Joblib for model saving, Google Colab

Key Steps: Data preprocessing (Label Encoding, StandardScaler), Feature engineering (total score, average score), Train-test split (80-20), Model training, Evaluation (RMSE, R²), Hyperparameter tuning

Feature Importance: Writing Score (32%), Reading Score (28%), Test Prep Course (15%), Lunch Type (12%), Parental Education (8%), Race/Ethnicity (3%), Gender (2%)

Key Insights: Reading & writing scores are strongest predictors, Test prep improves scores by 5-7 points, Lunch type (socioeconomic) significantly impacts performance, Gender shows minimal impact

Sample Predictions:

High performer (Reading 85, Writing 82) → Math 84-87

Average student (Reading 65, Writing 60) → Math 62-65

At-risk student (Reading 45, Writing 40) → Math 43-46

Installation:

bash
git clone https://github.com/yourusername/student-performance-prediction
pip install pandas numpy matplotlib seaborn scikit-learn joblib
Usage:

python
student = {'gender':'female','reading score':85,'writing score':82,'test preparation course':'completed'}
prediction = predict_math_score(student)  # Returns predicted math score
Model Saving: pickle.dump(model, open('models/model.pkl', 'wb'))

Project Structure: data/, models/, src/, requirements.txt, README.md

Future Work: Web app with Flask/Streamlit, REST API, Deep learning models
