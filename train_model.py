import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb
import joblib
import os

# Load the dataset
df = pd.read_csv('dataset.csv')

# Drop the ID column as it's not useful for modeling
df = df.drop(columns=['ID'])

# Define features and target
X = df.drop(columns=['default.payment.next.month'])
y = df['default.payment.next.month']

# Identify categorical columns
categorical_cols = ['SEX', 'EDUCATION', 'MARRIAGE']

# Create a column transformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), categorical_cols)
    ],
    remainder='passthrough'  # Keep numerical columns as is
)

# Create a pipeline with preprocessor and XGBoost
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', xgb.XGBClassifier(objective='binary:logistic', n_estimators=100, random_state=42))
])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train the model
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'xgb_model.pkl')

# Save the train and test data for later use in the app
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

# Create a sample new_data.csv for demonstration (using a subset of X_test)
new_data = X_test.sample(n=1000, random_state=42)
new_data.to_csv('new_data.csv', index=False)

print("Model trained and saved successfully.")
print("Training and test data saved.")
print("Sample new_data.csv created for demonstration.")