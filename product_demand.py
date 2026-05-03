# 1. IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# 2. LOAD DATASET
df = pd.read_csv('retail_demand.csv')
print('First 5 rows:\n', df.head())
print('\nColumns:\n', df.columns)

# 3. DATA PREPROCESSING
df = df.dropna()                           # Remove missing values
df = pd.get_dummies(df, drop_first=True)   # Encode categorical features
print('\nAfter Encoding:\n', df.head())

# 4. SELECT FEATURES AND TARGET
target = 'Units_Sold'
X = df.drop(target, axis=1)
y = df[target]
print('\nTarget Column:', target)

# 5. TRAIN-TEST SPLIT (80:20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 6. MODEL TRAINING
model = LinearRegression()
model.fit(X_train, y_train)

# 7. PREDICTION
y_pred = model.predict(X_test)

# 8. EVALUATION
print('\nModel Performance:')
print('MAE: ', metrics.mean_absolute_error(y_test, y_pred))
print('MSE: ', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R2:  ', metrics.r2_score(y_test, y_pred))

# 9. SCATTER PLOT
plt.figure()
plt.scatter(y_test, y_pred, alpha=0.6, color='#2E75B6')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Units Sold')
plt.ylabel('Predicted Units Sold')
plt.title('Actual vs Predicted Product Demand')
plt.show()

# 10. LINE PLOT (30 samples)
plt.figure()
plt.plot(range(30), y_test[:30], 'b-o', label='Actual')
plt.plot(range(30), y_pred[:30], 'r--s', label='Predicted')
plt.legend()
plt.title('Actual vs Predicted Demand Comparison')
plt.show()

# 11. RESIDUAL PLOT
residuals = y_test - y_pred
plt.figure()
plt.scatter(y_pred, residuals, alpha=0.5, color='#70AD47')
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()
