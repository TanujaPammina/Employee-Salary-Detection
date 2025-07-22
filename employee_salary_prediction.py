import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Set page config
st.set_page_config(page_title="Employee Salary Prediction", layout="centered")

st.title("💼 Employee Salary Prediction")
st.caption("Predict salary based on years of experience using Linear Regression.")

# Upload CSV file
uploaded_file = st.file_uploader("📂 Upload your salary_data.csv file", type=["csv"])

# Function to train model and make prediction
def train_model(data):
    X = data[['YearsExperience']]
    y = data['Salary']
    model = LinearRegression()
    model.fit(X, y)
    return model

if uploaded_file is not None:
    # Read uploaded CSV
    try:
        data = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")

        # Show sample data
        with st.expander("📊 View Dataset"):
            st.dataframe(data)

        # Sidebar input
        st.sidebar.header("Enter Employee Details:")
        years_exp = st.sidebar.number_input("Years of Experience", min_value=0.0, max_value=50.0, step=0.1)

        # Train and predict
        model = train_model(data)

        if years_exp:
            predicted_salary = model.predict([[years_exp]])[0]
            st.success(f"💰 Predicted Salary for {years_exp} years of experience: ₹ {int(predicted_salary):,}")

        # Plotting
        fig, ax = plt.subplots()
        sns.scatterplot(x='YearsExperience', y='Salary', data=data, color='blue', label='Data Points')
        sns.lineplot(x=data['YearsExperience'], y=model.predict(data[['YearsExperience']]), color='red', label='Regression Line')
        ax.set_title("Experience vs Salary")
        ax.set_xlabel("Years of Experience")
        ax.set_ylabel("Salary")
        ax.legend()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Error: {e}")

else:
    st.warning("⚠️ Please upload a `salary_data.csv` file to continue.")

# Footer
st.markdown("---")
st.caption("Developed with ❤️ using Streamlit")
