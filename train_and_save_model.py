import pandas as pd
from xgboost import XGBRegressor
import joblib
from prepro import load_data

# Train and save the model
X, y, _ = load_data()
xgb = XGBRegressor(max_depth=4, n_estimators=1000, learning_rate=0.1, booster='gbtree')
xgb.fit(X, y)
joblib.dump(xgb, "xgb_model.pkl")
print("Model trained and saved as xgb_model.pkl")
