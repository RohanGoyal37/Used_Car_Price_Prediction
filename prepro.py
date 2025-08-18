# Improved Preprocessing
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
import numpy as np

# Import Dataset
df = pd.read_csv('Car details v3.csv')

@st.cache
def load_data():
    data = df.copy()

    # Extract brand from name
    data['brand'] = data['name'].str.split().str[0]

    # Convert engine to numeric (remove 'CC')
    data['engine'] = data['engine'].str.replace('CC', '', regex=False)
    data['engine'] = pd.to_numeric(data['engine'], errors='coerce')

    # Convert max_power to numeric (remove 'bhp')
    data['max_power'] = data['max_power'].str.replace('bhp', '', regex=False)
    data['max_power'] = pd.to_numeric(data['max_power'], errors='coerce')

    # Convert mileage to numeric (remove 'kmpl', 'km/kg')
    data['mileage'] = data['mileage'].str.replace('kmpl', '', regex=False)
    data['mileage'] = data['mileage'].str.replace('km/kg', '', regex=False)
    data['mileage'] = pd.to_numeric(data['mileage'], errors='coerce')

    # Handle missing values
    for col in ['seats', 'mileage', 'max_power', 'engine']:
        if data[col].isnull().any():
            data[col].fillna(data[col].median(), inplace=True)
    for col in ['fuel', 'seller_type', 'owner', 'transmission', 'brand']:
        if data[col].isnull().any():
            data[col].fillna(data[col].mode()[0], inplace=True)

    # Remove outliers in selling_price (keep 1st to 99th percentile)
    lower = data['selling_price'].quantile(0.01)
    upper = data['selling_price'].quantile(0.99)
    data = data[(data['selling_price'] >= lower) & (data['selling_price'] <= upper)]

    # Encode categorical features
    cat_cols = data.select_dtypes(include=['object']).columns
    for col in cat_cols:
        data[col] = LabelEncoder().fit_transform(data[col])

    data['seats'] = data['seats'].astype(int)

    # Features and target
    y = data['selling_price']
    X = data.drop(['selling_price', 'name', 'torque'], axis=1, errors='ignore')
    return X, y, data

# Creating a function for DataFrame
@st.cache
def DataFrame():
    return load_data()[2]