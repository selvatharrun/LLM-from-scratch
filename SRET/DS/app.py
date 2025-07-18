# # import streamlit as st
# # import pandas as pd
# # import matplotlib.pyplot as plt
# # import seaborn as sns

# # # Set page config
# # st.set_page_config(page_title="Data Science Dashboard", layout="wide")

# # # App title
# # st.title("📊 Smart Data Insights - Streamlit App")

# # # File uploader
# # uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

# # # If a file is uploaded
# # if uploaded_file is not None:
# #     # Read the data
# #     df = pd.read_csv(uploaded_file)

# #     # Display preview
# #     st.subheader("🔍 Dataset Preview")
# #     st.dataframe(df.head())

# #     # Show basic info
# #     st.subheader("📌 Dataset Summary")
# #     st.write(f"**Number of rows:** {df.shape[0]}")
# #     st.write(f"**Number of columns:** {df.shape[1]}")
# #     st.write("**Data types:**")
# #     st.write(df.dtypes)

# #     # Show basic statistics
# #     st.subheader("📈 Summary Statistics")
# #     st.write(df.describe())

# #     # Sidebar filters
# #     st.sidebar.header("🔧 Filters & Visualization Settings")

# #     # Select a column to plot
# #     numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# #     if numeric_columns:
# #         selected_col = st.sidebar.selectbox("Choose a numeric column to visualize", numeric_columns)

# #         # Histogram
# #         st.subheader(f"📊 Histogram of {selected_col}")
# #         fig1, ax1 = plt.subplots()
# #         sns.histplot(df[selected_col], kde=True, ax=ax1, color="skyblue")
# #         st.pyplot(fig1)

# #         # Boxplot
# #         st.subheader(f"📦 Boxplot of {selected_col}")
# #         fig2, ax2 = plt.subplots()
# #         sns.boxplot(y=df[selected_col], ax=ax2, color="lightgreen")
# #         st.pyplot(fig2)

# #     # Correlation heatmap
# #     if len(numeric_columns) >= 2:
# #         st.subheader("🔗 Correlation Heatmap")
# #         fig3, ax3 = plt.subplots(figsize=(10, 6))
# #         sns.heatmap(df[numeric_columns].corr(), annot=True, cmap="coolwarm", ax=ax3)
# #         st.pyplot(fig3)

# #     # Optional: Add ML model prediction here
# #     st.markdown("---")
# #     st.subheader("🤖 Model Placeholder")
# #     st.write("You can add a trained machine learning model here to make predictions.")

# # else:
# #     st.info("👈 Please upload a CSV file to begin.")

# import streamlit as st
# import joblib

# # Load model
# model = joblib.load('house_price_model.pkl')

# # Title
# st.title("🏠 House Price Predictor")
# st.markdown("Enter the area of the house (in square feet) and get a price prediction!")

# # Input from user
# area = st.number_input("Enter Area (sq ft)", min_value=500, max_value=5000, step=100)

# # Predict button
# if st.button("Predict Price"):
#     prediction = model.predict([[area]])
#     st.success(f"Estimated Price: ₹{int(prediction[0]):,}")

# -------------------------------
# 📄 app.py (Streamlit App)
# -------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load models
reg_model = joblib.load('regression_model.pkl')
cls_model = joblib.load('classification_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

df = pd.read_csv('C:\\Users\\user\\OneDrive\\Documents\\LLM-from-scratch\\SRET\\DS\\student_data.csv')
df['Pass_Fail'] = df['Final_Score'].apply(lambda x: 'Pass' if x >= 50 else 'Fail')  # 👈 Fix

st.set_page_config(page_title="Student Score Predictor", layout="centered")
st.title("🎓 Student Performance Predictor")

# Sidebar user input
st.sidebar.header("📌 Input Student Details")
hours = st.sidebar.slider("Hours Studied", 0, 10, 5)
attendance = st.sidebar.slider("Attendance (%)", 50, 100, 75)
previous = st.sidebar.slider("Previous Exam Score", 0, 100, 60)

# Predictionss
if st.sidebar.button("🔍 Predict"):
    input_data = np.array([[hours, attendance, previous]])
    predicted_score = reg_model.predict(input_data)[0]
    predicted_label = cls_model.predict(input_data)[0]
    predicted_status = label_encoder.inverse_transform([predicted_label])[0]

    st.subheader("📈 Predicted Results")
    st.success(f"🎯 Predicted Final Score: {round(predicted_score, 2)}")
    if predicted_status == "Pass":
        st.info("✅ Student is likely to Pass")
    else:
        st.warning("❌ Student is likely to Fail")

    # Store prediction history
    if 'history' not in st.session_state:
        st.session_state.history = []
    st.session_state.history.append({
        'Hours': hours,
        'Attendance': attendance,
        'Previous Score': previous,
        'Predicted Score': round(predicted_score, 2),
        'Prediction': predicted_status
    })

# Prediction History
if 'history' in st.session_state:
    st.write("### 🧾 Prediction History")
    st.dataframe(pd.DataFrame(st.session_state.history))

# Visualization
st.write("### 📊 EDA - Area vs Final Score")
fig, ax = plt.subplots()
sns.scatterplot(data=df, x='Hours_Studied', y='Final_Score', hue='Pass_Fail', ax=ax)
plt.title("Study Hours vs Final Score")
st.pyplot(fig)

# Correlation heatmap
st.write("### 🔗 Correlation Heatmap")
fig2, ax2 = plt.subplots()
# Drop non-numeric columns like 'Pass_Fail'
numeric_df = df.drop(columns=['Pass_Fail'])  # or use df.select_dtypes

sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax2)
st.pyplot(fig2)

st.caption("Built by 3rd Year Engineering Student 🧑‍🎓")
