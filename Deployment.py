import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb

# Load the saved XGBoost model
xg_reg2 = xgb.XGBRegressor()
xg_reg2.load_model("xgboost_model.json")
test = pd.read_csv('test_dataset.csv')

# Columns of interest with their descriptions
columns_info = {
    "ExterQual": {
        "desc": "Quality of the outer material",
        "values": {1: "Poor", 2: "Regular", 3: "Good", 4: "Excelent"},
        "dtype": "float64",
    },
    "Qual_LivArea": {"desc": "Adjusted living area", "range": (733, 50950), "dtype": "int64"},
    "TotalSF": {"desc": "Total area in square feet", "range": (612, 10190), "dtype": "float64"},
    "OverallQual": {
        "desc": "Overall quality of the house",
        "values": {1: "Very low", 2: "Low", 3: "Below average", 4: "Average", 5: "Above average", 6: "Good", 7: "Very Good", 8: "Excelent", 9: "Luxury", 10: "Maximum luxury"},
        "dtype": "int64",
    },
    "BsmtQual": {
        "desc": "Basement height",
        "values": {-1: "No basement", 1: "Poor", 2: "Regular", 3: "Good", 4: "Excelent", 5: "Superior"},
        "dtype": "float64",
    },
    "KitchenQual": {
        "desc": "Kitchen quality",
        "values": {1: "Poor", 2: "Regular", 3: "Good", 4: "Excelent"},
        "dtype": "float64",
    },
    "Age": {"desc": "Age of the house in years", "range": (0, 129), "dtype": "int64"},
    "TotalBaths": {"desc": "Total number of bathrooms", "range": (1, 7), "dtype": "float64"},
    "Fireplaces": {"desc": "Number of chimneys", "range": (0, 4), "dtype": "int64"},
    "YearBuilt": {"desc": "Year of construction", "range": (1879, 2010), "dtype": "int64"},
    "GrLivArea": {"desc": "Living area in square feet", "range": (407, 5095), "dtype": "int64"},
    "GarageArea": {"desc": "Garage area in square feet", "range": (0, 1488), "dtype": "float64"},
}

st.title("House Price Prediction")
st.write("Modify the parameters and get the estimated house price!")

# Get the first record of the dataset
sample = test.iloc[0].copy()

# User input form
with st.form("house_form"):
    st.subheader("Enter House Details")
    
    for col, info in columns_info.items():
        if "values" in info:
            sample[col] = st.selectbox(f"{info['desc']} ({col})", options=list(info["values"].keys()), format_func=lambda x: info["values"][x])
        else:
            sample[col] = st.slider(f"{info['desc']} ({col})", info["range"][0], info["range"][1], int(sample[col]))
    
    submitted = st.form_submit_button("Predict Price")

if submitted:
    sample_df = pd.DataFrame([sample.to_dict()])
    predicted_price = xg_reg2.predict(sample_df)
    predicted_price = np.expm1(predicted_price)
    st.success(f"The estimated price of the house is: ${predicted_price[0]:,.2f}")
