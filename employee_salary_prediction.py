import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns

# Streamlit page configuration
st.set_page_config(page_title="Employee Salary Prediction", layout="centered")

# App title and description
st.title("💼 Employee Salary Prediction")
st.markdown("Predict salary based on years of experience using Linear Regression.")

# File uploader
uploaded_file = st.file_uploader("📁 Upload your salary_data.csv file", type="csv")

# Load and process data
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

if uploaded_file is not None:
    data = load_data(uploaded_file)

    # Train linear regression model
    X = data[['YearsExperience']]
    y = data['Salary']
    model = LinearRegression()
    model.fit(X, y)

    # Sidebar input for prediction
    st.sidebar.header("Enter Employee Details:")
    years_exp = st.sidebar.number_input("Years of Experience", min_value=0.0, max_value=50.0, step=0.1)

    if years_exp:
        predicted_salary = model.predict([[years_exp]])[0]
        st.success(f"💰 Predicted Salary for {years_exp} years of experience: ₹ {int(predicted_salary):,}")

    # Show dataset in an expandable section
    with st.expander("📊 View Dataset"):
        st.dataframe(data)

    # Plot the regression
    fig, ax = plt.subplots()
    sns.scatterplot(x='YearsExperience', y='Salary', data=data, color='blue', label='Data Points')
    sns.lineplot(x=data['YearsExperience'], y=model.predict(X), color='red', label='Regression Line')
    ax.set_title("Experience vs Salary")
    ax.set_xlabel("Years of Experience")
    ax.set_ylabel("Salary")
    ax.legend()
    st.pyplot(fig)

else:
    st.warning("⚠️ Please upload a `salary_data.csv` file to continue.")

# Footer
st.markdown("---")
st.caption("Developed with ❤️ using Streamlit")
# Replace this line
st.success(f"💰 Predicted Salary for {years_exp} years of experience: ₹ {int(predicted_salary):,}")

# With this
st.success(f"💰 Predicted Salary for {round(years_exp, 2)} years of experience: ₹ {int(predicted_salary):,}")

