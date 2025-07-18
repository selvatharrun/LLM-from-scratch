# import pandas as pd
# from sklearn.linear_model import LinearRegression
# import joblib

# # Sample dataset
# data = {
#     'Area': [1000, 1500, 1800, 2400, 3000],
#     'Price': [100000, 150000, 180000, 240000, 300000]
# }
# df = pd.DataFrame(data)

# # Train model
# model = LinearRegression()
# model.fit(df[['Area']], df['Price'])

# # Save model
# joblib.dump(model, 'house_price_model.pkl')
# print("Model saved!")

# -------------------------------
# 📁 student_data.csv (prepare this as a CSV file)
# -------------------------------
# Area: Hours_Studied, Attendance (%), Previous_Score, Final_Score
# Add this as a CSV file in the same directory

# """
# Hours_Studied,Attendance,Previous_Score,Final_Score
# 2,80,50,55
# 4,85,60,65
# 5,75,70,70
# 7,90,80,85
# 1,60,45,48
# 3,70,55,60
# 8,95,90,95
# 6,88,75,80
# """

# -------------------------------
# 📄 train_model.py (trains and saves model)
# -------------------------------

import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
df = pd.read_csv('C:\\Users\\user\\OneDrive\\Documents\\LLM-from-scratch\\SRET\\DS\\student_data.csv')

# Create target for classification
df['Pass_Fail'] = df['Final_Score'].apply(lambda x: 'Pass' if x >= 50 else 'Fail')

# Regression Model
X_reg = df[['Hours_Studied', 'Attendance', 'Previous_Score']]
y_reg = df['Final_Score']
reg_model = LinearRegression()
reg_model.fit(X_reg, y_reg)
joblib.dump(reg_model, 'regression_model.pkl')

# Classification Model
le = LabelEncoder()
y_cls = le.fit_transform(df['Pass_Fail'])
cls_model = LogisticRegression()
cls_model.fit(X_reg, y_cls)
joblib.dump(cls_model, 'classification_model.pkl')
joblib.dump(le, 'label_encoder.pkl')

print("✅ Models trained and saved")

