# Cross-validation for model evaluation
from prepro import load_data
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
import numpy as np

X, y, _ = load_data()
model = XGBRegressor(max_depth=4, n_estimators=1000, learning_rate=0.1, booster='gbtree')

# 5-fold cross-validation (change cv=5 to cv=10 for 10-fold)
scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print(f"Cross-validated R^2 scores: {scores}")
print(f"Mean R^2: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
