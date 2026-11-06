# ========================================
# EDA AI BIAS PROJECT - DATA PREPROCESSING
# ========================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import joblib

print("=" * 50)
print("STEP 1: Loading the dataset...")
print("=" * 50)

# Load the CSV file
df = pd.read_csv('adult.csv')

# Show basic info
print(f"\nDataset shape: {df.shape} rows, {df.shape[1]} columns")
print(f"\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\n" + "=" * 50)
print("STEP 2: Handling missing values...")
print("=" * 50)

# Replace '?' with NaN (proper missing value marker)
df.replace('?', np.nan, inplace=True)

# Show missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Drop rows with missing values
df.dropna(inplace=True)
print(f"\nAfter dropping missing values: {df.shape} rows remaining")

print("\n" + "=" * 50)
print("STEP 3: Defining Target Variable...")
print("=" * 50)

# Create binary target: 1 if income > 50K, 0 otherwise
y = np.where(df['income'] == ' >50K', 1, 0)
print(f"Target distribution:")
print(pd.Series(y).value_counts())

# Drop target and unnecessary columns
X = df.drop(['income', 'fnlwgt', 'education'], axis=1)
print(f"\nFeatures shape: {X.shape}")

print("\n" + "=" * 50)
print("STEP 4: Encoding Categorical Variables...")
print("=" * 50)

# Identify categorical columns
cat_cols = X.select_dtypes(include=['object']).columns.tolist()
print(f"Categorical columns: {cat_cols}")

# Identify numerical columns (FIXED: changed 'education-num' to 'educational-num')
num_cols = ['age', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']
print(f"Numerical columns to scale: {num_cols}")

# Create a preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_cols),
        ('num', StandardScaler(), num_cols)
    ],
    remainder='drop'
)

# Apply preprocessing to the data
X_processed = preprocessor.fit_transform(X)
print(f"\nProcessed feature shape: {X_processed.shape}")

print("\n" + "=" * 50)
print("STEP 5: Splitting into Train/Test Sets...")
print("=" * 50)

X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

print("\n" + "=" * 50)
print("STEP 6: Saving preprocessed data...")
print("=" * 50)

# Save the preprocessor so we can use it later
joblib.dump(preprocessor, 'preprocessor.joblib')

# Save the processed data as numpy files
np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)

print("✅ All files saved successfully!")
print("\nSaved files:")
print("  - preprocessor.joblib")
print("  - X_train.npy, X_test.npy")
print("  - y_train.npy, y_test.npy")
print("\n" + "=" * 50)
print("PREPROCESSING COMPLETE!")
print("=" * 50)