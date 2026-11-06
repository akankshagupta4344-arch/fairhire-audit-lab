import numpy as np
import pandas as pd

print("=" * 50)
print("VIEWING PREPROCESSED DATA")
print("=" * 50)

# Load the data
X_train = np.load('X_train.npy')
X_test = np.load('X_test.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

print(f"\nX_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# Show first 5 rows of training data
print(f"\nFirst 5 rows of X_train:")
print(X_train[:5])

print(f"\nFirst 5 rows of y_train:")
print(y_train[:5])

# Load the preprocessor to see feature names
import joblib
preprocessor = joblib.load('preprocessor.joblib')

# Get feature names (optional - if you want to see column names)
print("\n" + "=" * 50)
print("Data loaded successfully!")
print("=" * 50)