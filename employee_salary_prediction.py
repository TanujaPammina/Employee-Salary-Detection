import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns

# Set Streamlit page config
st.set_page_config(page_title="Employee Salary Prediction", layout="centered")

# Title
st.title("💼 Employee Salary Prediction")
st.markdown("Predict salary based on years of experience using Linear Regression.")

# Load dataset
@st.cache_data
def load_data():
    data = pd.read_csv("salary_data.csv")
    return data

data = load_data()

# Train model
X = data[['YearsExperience']]
y = data['Salary']
model = LinearRegression()
model.fit(X, y)

# Sidebar for input
st.sidebar.header("Enter Employee Details:")
years_exp = st.sidebar.number_input("Years of Experience", min_value=0.0, max_value=50.0, step=0.1)

# Predict salary
if years_exp:
    predicted_salary = model.predict([[years_exp]])[0]
    st.success(f"💰 Predicted Salary for {years_exp} years of experience: ₹ {int(predicted_salary):,}")

# Display data
with st.expander("📊 View Dataset"):
    st.dataframe(data)

# Plot
fig, ax = plt.subplots()
sns.scatterplot(x='YearsExperience', y='Salary', data=data, color='blue', label='Data Points')
sns.lineplot(x=data['YearsExperience'], y=model.predict(X), color='red', label='Regression Line')
ax.set_title("Experience vs Salary")
ax.set_xlabel("Years of Experience")
ax.set_ylabel("Salary")
ax.legend()
st.pyplot(fig)

# Footer
st.markdown("---")
st.caption("Developed with ❤️ using Streamlit")
