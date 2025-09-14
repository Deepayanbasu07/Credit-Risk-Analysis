import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load the training data"""
    df_train = pd.read_csv('train/train.csv')
    return df_train

def preprocess_data(df_train):
    """Preprocess the data following the notebook steps"""
    # Feature Engineering
    df_train['DTI_ratio'] = df_train['yearly_debt_payments'] / df_train['net_yearly_income']
    df_train['outstanding_balance'] = df_train['credit_limit'] * (df_train['credit_limit_used(%)'] / 100)

    # Credit score bucketing
    score_bins = [0, 580, 670, 740, 800, float('inf')]
    score_labels = ['Poor', 'Fair', 'Good', 'Very Good', 'Excellent']
    df_train['credit_score_bucket'] = pd.cut(df_train['credit_score'], bins=score_bins, labels=score_labels, right=False)

    # Age grouping
    age_bins = [0, 29, 45, 60, float('inf')]
    age_labels = ['Young_Adult', 'Middle_Aged', 'Senior_Adult', 'Elderly']
    df_train['age_group'] = pd.cut(df_train['age'], bins=age_bins, labels=age_labels, right=False)

    # Encoding
    df_train['owns_car'] = df_train['owns_car'].map({'Y': 1, 'N': 0})
    df_train['owns_house'] = df_train['owns_house'].map({'Y': 1, 'N': 0})
    df_train['gender'] = df_train['gender'].map({'M': 1, 'F': 0})

    # Handle missing values
    median_debt_by_occupation = df_train.groupby('occupation_type')['yearly_debt_payments'].transform('median')
    df_train['yearly_debt_payments'] = df_train['yearly_debt_payments'].fillna(median_debt_by_occupation)

    median_employment = df_train.groupby('occupation_type')['no_of_days_employed'].transform('median')
    df_train['no_of_days_employed'] = df_train['no_of_days_employed'].fillna(median_employment)

    median_children = df_train.groupby('total_family_members')['no_of_children'].transform('median')
    df_train['no_of_children'] = df_train['no_of_children'].fillna(median_children)

    median_cars = df_train.groupby('occupation_type')['owns_car'].transform("median")
    df_train['owns_car'] = df_train['owns_car'].fillna(median_cars)

    median_dti_by_bucket = df_train.groupby('credit_score_bucket')['DTI_ratio'].transform('median')
    df_train['DTI_ratio'] = df_train['DTI_ratio'].fillna(median_dti_by_bucket)

    # Drop leaky features and unnecessary columns
    df_train = df_train.drop(['default_in_last_6months', 'name'], axis=1)

    # Drop rows with remaining NaN
    df_train = df_train.dropna().reset_index(drop=True)

    # One-hot encoding with consistent columns
    df_train = pd.get_dummies(df_train, columns=['occupation_type'], drop_first=True)

    # Log transformations
    df_train['log_income'] = np.log1p(df_train['net_yearly_income'])
    df_train['log_no_of_days_employed'] = np.log1p(df_train['no_of_days_employed'])
    df_train['log_credit_limit'] = np.log1p(df_train['credit_limit'])

    # Feature engineering
    df_train['income_per_person'] = df_train['net_yearly_income'] / (df_train['total_family_members'] + 1)
    df_train['credit_utilization'] = df_train['credit_limit_used(%)'] / 100

    # Select final features for training data (includes target)
    selected_features_train = [
        "gender", "owns_car", "owns_house", "no_of_children",
        "total_family_members", "migrant_worker", "prev_defaults",
        "yearly_debt_payments", "outstanding_balance",
        "log_income", "log_no_of_days_employed", "log_credit_limit",
        "occupation_type_Cleaning staff", "occupation_type_Cooking staff",
        "occupation_type_Core staff", "occupation_type_Drivers",
        "occupation_type_HR staff", "occupation_type_High skill tech staff",
        "occupation_type_IT staff", "occupation_type_Laborers",
        "occupation_type_Low-skill Laborers", "occupation_type_Managers",
        "occupation_type_Medicine staff", "occupation_type_Private service staff",
        "occupation_type_Realty agents", "occupation_type_Sales staff",
        "occupation_type_Secretaries", "occupation_type_Security staff",
        "occupation_type_Waiters/barmen staff",
        "age_group", "credit_score_bucket", "credit_card_default"
    ]

    df_train = df_train[selected_features_train]

    # One-hot encode categorical variables
    df_train = pd.get_dummies(df_train, columns=["age_group", "credit_score_bucket"], drop_first=True)

    # Convert boolean columns to int
    bool_cols = df_train.select_dtypes(include='bool').columns
    df_train[bool_cols] = df_train[bool_cols].astype(int)

    return df_train

def train_and_save_models():
    """Train all models and save them"""
    print("Loading data...")
    df_train = load_data()

    print("Preprocessing data...")
    df_train_processed = preprocess_data(df_train)

    # Prepare features and target
    X_train = df_train_processed.drop(columns=["credit_card_default"])
    y_train = df_train_processed["credit_card_default"]

    print("Training models...")
    # Handle class imbalance
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Train models
    lr_model = LogisticRegression(random_state=42, solver='liblinear')
    rf_model = RandomForestClassifier(random_state=42)
    gb_model = GradientBoostingClassifier(random_state=42)
    xgb_model = xgb.XGBClassifier(random_state=42)

    lr_model.fit(X_train_resampled, y_train_resampled)
    rf_model.fit(X_train_resampled, y_train_resampled)
    gb_model.fit(X_train_resampled, y_train_resampled)
    xgb_model.fit(X_train_resampled, y_train_resampled)

    print("Saving models...")
    models_dir = 'models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    joblib.dump(lr_model, os.path.join(models_dir, 'lr_model.pkl'))
    joblib.dump(rf_model, os.path.join(models_dir, 'rf_model.pkl'))
    joblib.dump(gb_model, os.path.join(models_dir, 'gb_model.pkl'))
    joblib.dump(xgb_model, os.path.join(models_dir, 'xgb_model.pkl'))
    joblib.dump(X_train_resampled, os.path.join(models_dir, 'X_train_resampled.pkl'))
    joblib.dump(y_train_resampled, os.path.join(models_dir, 'y_train_resampled.pkl'))

    print("Models saved successfully!")

if __name__ == "__main__":
    train_and_save_models()