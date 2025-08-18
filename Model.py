#importing Libraries
import joblib
import streamlit as st
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_squared_log_error
from math import sqrt
from prepro import load_data

@st.cache
def load_trained_model():
    return joblib.load("xgb_model.pkl")

def model(df, feature):
    """This function loads the trained model and returns the prediction and metrics."""
    model = load_trained_model()
    X, y, _ = load_data()
    pred_price = model.predict(feature)[0]

    # Calculate metrics on the full data (optional, for display)
    y_pred = model.predict(X)
    rsquare_score = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    import numpy as np
    if np.any(y <= -1) or np.any(y_pred <= -1):
        msle = None  # or float('nan')
    else:
        msle = mean_squared_log_error(y, y_pred)
    rmse = sqrt(mean_squared_error(y, y_pred))
    acc = model.score(X, y)

    return model, acc*100, pred_price, rsquare_score*100, mae, msle, rmse