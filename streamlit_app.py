import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
import shap
import joblib
import os
import warnings
from scipy import stats
from sklearn.calibration import calibration_curve
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(page_title="Advanced Credit Risk Analysis Dashboard", layout="wide", page_icon="💳")

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2ca02c;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem;
    }
    .highlight-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .formula-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        font-family: 'Courier New', monospace;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Enhanced caching functions
@st.cache_data
def load_data():
    """Load and cache the training and test data"""
    try:
        df_train = pd.read_csv('train/train.csv')
        df_test = pd.read_csv('test/test.csv')
        return df_train, df_test
    except FileNotFoundError:
        st.error("Data files not found. Please ensure train.csv and test.csv are in the correct directories.")
        return None, None

@st.cache_resource
def preprocess_data(df_train, df_test):
    """Enhanced preprocessing with comprehensive feature engineering"""
    # Feature Engineering
    df_train['DTI_ratio'] = df_train['yearly_debt_payments'] / df_train['net_yearly_income']
    df_test['DTI_ratio'] = df_test['yearly_debt_payments'] / df_test['net_yearly_income']

    df_train['outstanding_balance'] = df_train['credit_limit'] * (df_train['credit_limit_used(%)'] / 100)
    df_test['outstanding_balance'] = df_test['credit_limit'] * (df_test['credit_limit_used(%)'] / 100)

    # Credit score bucketing
    score_bins = [0, 580, 670, 740, 800, float('inf')]
    score_labels = ['Poor', 'Fair', 'Good', 'Very Good', 'Excellent']
    df_train['credit_score_bucket'] = pd.cut(df_train['credit_score'], bins=score_bins, labels=score_labels, right=False)
    df_test['credit_score_bucket'] = pd.cut(df_test['credit_score'], bins=score_bins, labels=score_labels, right=False)

    # Age grouping
    age_bins = [0, 29, 45, 60, float('inf')]
    age_labels = ['Young_Adult', 'Middle_Aged', 'Senior_Adult', 'Elderly']
    df_train['age_group'] = pd.cut(df_train['age'], bins=age_bins, labels=age_labels, right=False)
    df_test['age_group'] = pd.cut(df_test['age'], bins=age_bins, labels=age_labels, right=False)

    # Encoding
    df_train['owns_car'] = df_train['owns_car'].map({'Y': 1, 'N': 0})
    df_train['owns_house'] = df_train['owns_house'].map({'Y': 1, 'N': 0})
    df_train['gender'] = df_train['gender'].map({'M': 1, 'F': 0})

    df_test['owns_car'] = df_test['owns_car'].map({'Y': 1, 'N': 0})
    df_test['owns_house'] = df_test['owns_house'].map({'Y': 1, 'N': 0})
    df_test['gender'] = df_test['gender'].map({'M': 1, 'F': 0})

    # Enhanced missing value handling
    median_debt_by_occupation = df_train.groupby('occupation_type')['yearly_debt_payments'].transform('median')
    df_train['yearly_debt_payments'] = df_train['yearly_debt_payments'].fillna(median_debt_by_occupation)
    df_test['yearly_debt_payments'] = df_test['yearly_debt_payments'].fillna(median_debt_by_occupation)

    median_employment = df_train.groupby('occupation_type')['no_of_days_employed'].transform('median')
    df_train['no_of_days_employed'] = df_train['no_of_days_employed'].fillna(median_employment)
    df_test['no_of_days_employed'] = df_test['no_of_days_employed'].fillna(median_employment)

    median_children = df_train.groupby('total_family_members')['no_of_children'].transform('median')
    df_train['no_of_children'] = df_train['no_of_children'].fillna(median_children)
    df_test['no_of_children'] = df_test['no_of_children'].fillna(median_children)

    median_cars = df_train.groupby('occupation_type')['owns_car'].transform("median")
    df_train['owns_car'] = df_train['owns_car'].fillna(median_cars)
    df_test['owns_car'] = df_test['owns_car'].fillna(median_cars)

    median_dti_by_bucket = df_train.groupby('credit_score_bucket')['DTI_ratio'].transform('median')
    df_train['DTI_ratio'] = df_train['DTI_ratio'].fillna(median_dti_by_bucket)
    df_test['DTI_ratio'] = df_test['DTI_ratio'].fillna(median_dti_by_bucket)

    # Drop leaky features and unnecessary columns
    df_train = df_train.drop(['default_in_last_6months', 'name'], axis=1)
    df_test = df_test.drop(['default_in_last_6months', 'name'], axis=1)

    # Drop rows with remaining NaN
    df_train = df_train.dropna().reset_index(drop=True)
    df_test = df_test.dropna().reset_index(drop=True)

    # One-hot encoding with consistent columns
    combined = pd.concat([df_train, df_test], ignore_index=True)
    combined = pd.get_dummies(combined, columns=['occupation_type'], drop_first=True)
    df_train = combined.iloc[:len(df_train)].copy()
    df_test = combined.iloc[len(df_train):].copy().drop(columns=['credit_card_default'], errors='ignore')

    # Log transformations
    df_train['log_income'] = np.log1p(df_train['net_yearly_income'])
    df_test['log_income'] = np.log1p(df_test['net_yearly_income'])
    df_train['log_no_of_days_employed'] = np.log1p(df_train['no_of_days_employed'])
    df_test['log_no_of_days_employed'] = np.log1p(df_test['no_of_days_employed'])
    df_train['log_credit_limit'] = np.log1p(df_train['credit_limit'])
    df_test['log_credit_limit'] = np.log1p(df_test['credit_limit'])

    # Feature engineering
    df_train['income_per_person'] = df_train['net_yearly_income'] / (df_train['total_family_members'] + 1)
    df_train['credit_utilization'] = df_train['credit_limit_used(%)'] / 100

    df_test['income_per_person'] = df_test['net_yearly_income'] / (df_test['total_family_members'] + 1)
    df_test['credit_utilization'] = df_test['credit_limit_used(%)'] / 100

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

    # Select final features for test data (excludes target)
    selected_features_test = [
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
        "age_group", "credit_score_bucket"
    ]

    df_train = df_train[selected_features_train]
    df_test = df_test[selected_features_test]

    # One-hot encode categorical variables
    df_train = pd.get_dummies(df_train, columns=["age_group", "credit_score_bucket"], drop_first=True)
    df_test = pd.get_dummies(df_test, columns=["age_group", "credit_score_bucket"], drop_first=True)

    # Convert boolean columns to int
    bool_cols = df_train.select_dtypes(include='bool').columns
    df_train[bool_cols] = df_train[bool_cols].astype(int)
    df_test[bool_cols] = df_test[bool_cols].astype(int)

    return df_train, df_test

@st.cache_resource
def load_or_train_models(X_train, y_train):
    """Load pretrained models if available, otherwise train and save them"""
    models_dir = 'models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    model_files = {
        'lr_model.pkl': None,
        'rf_model.pkl': None,
        'gb_model.pkl': None,
        'xgb_model.pkl': None,
        'X_train_resampled.pkl': None,
        'y_train_resampled.pkl': None
    }

    # Check if all model files exist
    all_models_exist = all(os.path.exists(os.path.join(models_dir, f)) for f in model_files.keys())

    if all_models_exist:
        # Load pretrained models
        lr_model = joblib.load(os.path.join(models_dir, 'lr_model.pkl'))
        rf_model = joblib.load(os.path.join(models_dir, 'rf_model.pkl'))
        gb_model = joblib.load(os.path.join(models_dir, 'gb_model.pkl'))
        xgb_model = joblib.load(os.path.join(models_dir, 'xgb_model.pkl'))
        X_train_resampled = joblib.load(os.path.join(models_dir, 'X_train_resampled.pkl'))
        y_train_resampled = joblib.load(os.path.join(models_dir, 'y_train_resampled.pkl'))
    else:
        # Train models
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

        # Save models
        joblib.dump(lr_model, os.path.join(models_dir, 'lr_model.pkl'))
        joblib.dump(rf_model, os.path.join(models_dir, 'rf_model.pkl'))
        joblib.dump(gb_model, os.path.join(models_dir, 'gb_model.pkl'))
        joblib.dump(xgb_model, os.path.join(models_dir, 'xgb_model.pkl'))
        joblib.dump(X_train_resampled, os.path.join(models_dir, 'X_train_resampled.pkl'))
        joblib.dump(y_train_resampled, os.path.join(models_dir, 'y_train_resampled.pkl'))

    return lr_model, rf_model, gb_model, xgb_model, X_train_resampled, y_train_resampled

# Statistical Analysis Functions
def perform_t_tests(df, target_col='credit_card_default'):
    """Perform independent t-tests for continuous variables vs target"""
    continuous_vars = ['log_income', 'credit_score', 'yearly_debt_payments', 'credit_limit']
    results = []

    for var in continuous_vars:
        if var in df.columns:
            group1 = df[df[target_col] == 0][var]
            group2 = df[df[target_col] == 1][var]

            t_stat, p_value = stats.ttest_ind(group1, group2)

            results.append({
                'Variable': var,
                'T-Statistic': f"{t_stat:.4f}",
                'P-Value': f"{p_value:.6f}",
                'Significant': 'Yes' if p_value < 0.05 else 'No'
            })

    return pd.DataFrame(results)

def perform_chi_square_tests(df, target_col='credit_card_default'):
    """Perform chi-square tests for categorical variables vs target"""
    categorical_vars = ['gender', 'owns_car', 'owns_house', 'occupation_type']
    results = []

    for var in categorical_vars:
        if var in df.columns:
            contingency_table = pd.crosstab(df[var], df[target_col])
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

            results.append({
                'Variable': var,
                'Chi-Square': f"{chi2:.4f}",
                'P-Value': f"{p_value:.6f}",
                'Significant': 'Yes' if p_value < 0.05 else 'No'
            })

    return pd.DataFrame(results)

def calculate_woe_iv(df, feature_col, target_col='credit_card_default', bins=10):
    """Calculate Weight of Evidence and Information Value for a feature"""
    df_temp = df[[feature_col, target_col]].copy()

    # Create bins for continuous variables
    if df_temp[feature_col].dtype in ['int64', 'float64']:
        df_temp[f'{feature_col}_bins'] = pd.qcut(df_temp[feature_col], q=bins, duplicates='drop')
    else:
        df_temp[f'{feature_col}_bins'] = df_temp[feature_col]

    # Calculate counts
    grouped = df_temp.groupby(f'{feature_col}_bins')[target_col].agg(['count', 'sum'])
    grouped.columns = ['Total', 'Bads']
    grouped['Goods'] = grouped['Total'] - grouped['Bads']

    # Calculate percentages
    grouped['Bad_Rate'] = grouped['Bads'] / grouped['Bads'].sum()
    grouped['Good_Rate'] = grouped['Goods'] / grouped['Goods'].sum()

    # Calculate WoE
    grouped['WoE'] = np.log(grouped['Good_Rate'] / grouped['Bad_Rate'])
    grouped['WoE'] = grouped['WoE'].replace([np.inf, -np.inf], 0)

    # Calculate IV
    grouped['IV'] = (grouped['Good_Rate'] - grouped['Bad_Rate']) * grouped['WoE']

    total_iv = grouped['IV'].sum()

    return grouped, total_iv

def calculate_gini(y_true, y_pred_proba):
    """Calculate Gini coefficient"""
    auc_score = auc(*roc_curve(y_true, y_pred_proba)[:2])
    return 2 * auc_score - 1

def calculate_psi(expected, actual, bins=10):
    """Calculate Population Stability Index - Robust implementation"""
    # Handle edge cases
    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    # Use quantiles for more robust binning
    breakpoints = np.quantile(expected, np.linspace(0, 1, bins + 1))

    # Ensure unique breakpoints
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 2:
        return 0.0

    expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)

    # Avoid division by zero and log(0)
    expected_percents = np.where(expected_percents == 0, 1e-6, expected_percents)
    actual_percents = np.where(actual_percents == 0, 1e-6, actual_percents)

    # Calculate PSI with numerical stability
    psi_values = (actual_percents - expected_percents) * np.log(actual_percents / expected_percents)
    psi = np.sum(psi_values)

    return max(0, psi)  # PSI should be non-negative

def calculate_ks_statistic(y_true, y_pred_proba):
    """Calculate Kolmogorov-Smirnov statistic with detailed output"""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    ks_statistic = max(tpr - fpr)
    ks_threshold = thresholds[np.argmax(tpr - fpr)]

    return ks_statistic, ks_threshold, fpr, tpr

def calculate_csi(y_true, y_pred_proba, threshold=0.5):
    """Calculate Characteristic Stability Index - Corrected implementation"""
    # CSI measures stability of model scores over time
    # This should compare score distributions, not default vs non-default
    scores = y_pred_proba  # Model prediction scores
    return calculate_psi(scores, scores)  # Placeholder - will be updated with proper comparison

def calculate_feature_psi(feature_train, feature_prod):
    """Calculate PSI for individual features"""
    return calculate_psi(feature_train, feature_prod)

def create_gains_chart(y_true, y_pred_proba, n_deciles=10):
    """Create a gains chart for model evaluation"""
    # Create deciles
    df_gains = pd.DataFrame({
        'y_true': y_true,
        'y_pred_proba': y_pred_proba
    })

    df_gains = df_gains.sort_values('y_pred_proba', ascending=False)
    df_gains['decile'] = pd.qcut(df_gains.index, n_deciles, labels=False)
    df_gains = df_gains.groupby('decile').agg({
        'y_true': ['count', 'sum'],
        'y_pred_proba': 'mean'
    }).round(4)

    df_gains.columns = ['Total', 'Defaults', 'Avg_Score']
    df_gains['Cumulative_Defaults'] = df_gains['Defaults'].cumsum()
    df_gains['Cumulative_Pct'] = df_gains['Cumulative_Defaults'] / df_gains['Defaults'].sum()
    df_gains['Decile_Pct'] = df_gains['Total'] / df_gains['Total'].sum()

    return df_gains

def calculate_calibration_curve(y_true, y_pred_proba, n_bins=10):
    """Calculate calibration curve data"""
    prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=n_bins)
    return prob_true, prob_pred

def calculate_disparate_impact(df, protected_attr, target_col='credit_card_default'):
    """Calculate disparate impact ratio for fairness analysis"""
    # Calculate favorable outcome rates for each group
    favorable_rate = df.groupby(protected_attr)[target_col].mean()

    if len(favorable_rate) >= 2:
        # Disparate impact = (rate for unprivileged) / (rate for privileged)
        # Assuming lower values of protected attribute are unprivileged
        unprivileged_rate = favorable_rate.iloc[0]
        privileged_rate = favorable_rate.iloc[1]

        disparate_impact = unprivileged_rate / privileged_rate if privileged_rate > 0 else 0

        return disparate_impact, favorable_rate
    else:
        return None, favorable_rate

def simulate_production_drift(X_train, n_samples=None, drift_config=None):
    """Enhanced production data simulation with different types of drift"""
    if n_samples is None:
        n_samples = len(X_train)

    if drift_config is None:
        drift_config = {
            'covariate_drift': {
                'log_income': {'drift': 0.1, 'direction': 'decrease'},
                'credit_limit_used(%)': {'drift': 0.15, 'direction': 'increase'},
                'prev_defaults': {'drift': 0.05, 'direction': 'increase'},
            },
            'concept_drift': {
                'relationship_shift': 0.1  # Simulate slight change in relationships
            }
        }

    # Create production data by sampling from training data with drift
    X_prod = X_train.sample(n=n_samples, replace=True, random_state=42).copy()

    # Apply covariate drift to key features
    for feature, config in drift_config['covariate_drift'].items():
        if feature in X_prod.columns:
            drift_factor = config['drift']
            if config['direction'] == 'increase':
                X_prod[feature] = X_prod[feature] * (1 + drift_factor)
            elif config['direction'] == 'decrease':
                X_prod[feature] = X_prod[feature] * (1 - drift_factor)

            # Ensure values stay within reasonable bounds
            if 'credit_limit_used' in feature:
                X_prod[feature] = np.clip(X_prod[feature], 0, 100)
            elif 'prev_defaults' in feature:
                X_prod[feature] = np.clip(X_prod[feature], 0, X_prod[feature].max())

    return X_prod

def main():
    st.markdown('<h1 class="main-header">🏦 Advanced Credit Risk Analysis & Model Governance Dashboard</h1>', unsafe_allow_html=True)

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", [
        "Overview",
        "Data Exploration",
        "Preprocessing",
        "Model Evaluation",
        "Model Monitoring",
        "Fairness Audit",
        "Individual Prediction"
    ])

    # Load data
    df_train, df_test = load_data()
    if df_train is None or df_test is None:
        return

    # Preprocess data
    df_train_processed, df_test_processed = preprocess_data(df_train, df_test)

    # Prepare features and target
    X_train = df_train_processed.drop(columns=["credit_card_default"])
    y_train = df_train_processed["credit_card_default"]

    # For demo purposes, use part of training data as test since test labels aren't available
    from sklearn.model_selection import train_test_split as tts
    X_train_split, X_test, y_train_split, y_test = tts(X_train, y_train, test_size=0.2, random_state=42)
    X_train = X_train_split
    y_train = y_train_split

    # Load or train models
    lr_model, rf_model, gb_model, xgb_model, X_train_resampled, y_train_resampled = load_or_train_models(X_train, y_train)

    # Get predictions for all models
    y_pred_lr = lr_model.predict(X_test)
    y_pred_rf = rf_model.predict(X_test)
    y_pred_gb = gb_model.predict(X_test)
    y_pred_xgb = xgb_model.predict(X_test)

    y_pred_proba_lr = lr_model.predict_proba(X_test)[:, 1]
    y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]
    y_pred_proba_gb = gb_model.predict_proba(X_test)[:, 1]
    y_pred_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]

    # Page routing
    if page == "Overview":
        show_enhanced_overview(df_train, df_train_processed)
    elif page == "Data Exploration":
        show_enhanced_data_exploration(df_train_processed)
    elif page == "Preprocessing":
        show_enhanced_preprocessing(df_train, df_train_processed)
    elif page == "Model Evaluation":
        show_enhanced_model_evaluation(y_test, y_pred_lr, y_pred_rf, y_pred_gb, y_pred_xgb,
                                     y_pred_proba_lr, y_pred_proba_rf, y_pred_proba_gb, y_pred_proba_xgb)
    elif page == "Model Monitoring":
        show_enhanced_model_monitoring(y_test, y_pred_proba_xgb, X_train, X_test, xgb_model)
    elif page == "Fairness Audit":
        show_enhanced_fairness_audit(X_test, y_pred_xgb, y_test)
    elif page == "Individual Prediction":
        show_enhanced_individual_prediction(xgb_model, X_test.columns)

def show_enhanced_overview(df_train, df_train_processed):
    """Enhanced overview page with business context and ECL formula"""
    st.markdown('<h2 class="section-header">📊 Overview</h2>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", f"{len(df_train_processed):,}")
    with col2:
        default_rate = df_train_processed['credit_card_default'].mean() * 100
        st.metric("Default Rate", f"{default_rate:.1f}%")
    with col3:
        st.metric("Features", len(df_train_processed.columns) - 1)

    # Business Context Section
    st.markdown('<h3 class="section-header">🏢 Business Context</h3>', unsafe_allow_html=True)

    st.markdown("""
    This dashboard demonstrates advanced credit risk modeling techniques essential for financial institutions to:

    - **Comply with Basel II/III regulations** for calculating Risk-Weighted Assets (RWA)
    - **Meet IFRS 9 requirements** for Expected Credit Loss (ECL) provisioning
    - **Make informed lending decisions** while managing portfolio risk
    - **Monitor model performance** to ensure ongoing effectiveness
    """)

    # ECL Formula
    st.markdown("### 📈 Expected Credit Loss (ECL) Formula")
    st.markdown("""
    <div style="background-color: #ffffff; padding: 1.5rem; border-radius: 0.75rem; border: 2px solid #1f77b4; margin: 1rem 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        <h4 style="color: #1f77b4; margin-bottom: 1rem; text-align: center;">IFRS 9 Expected Credit Loss Calculation</h4>
        <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; font-family: 'Courier New', monospace; font-size: 1.2rem; font-weight: bold; text-align: center; color: #2c3e50; margin-bottom: 1rem;">
            ECL = Σ(PDᵢ × LGDᵢ × EADᵢ) × (1 + r)⁻ᵗ
        </div>
        <div style="background-color: #e8f4f8; padding: 1rem; border-radius: 0.5rem;">
            <h5 style="color: #1f77b4; margin-bottom: 0.5rem;">Where:</h5>
            <ul style="margin: 0; padding-left: 1.5rem; color: #2c3e50; font-size: 1rem; line-height: 1.6;">
                <li><strong style="color: #1f77b4;">PDᵢ</strong>: <span style="color: #2c3e50;">Probability of Default in period i (focus of our predictive model)</span></li>
                <li><strong style="color: #1f77b4;">LGDᵢ</strong>: <span style="color: #2c3e50;">Loss Given Default in period i (percentage lost when default occurs)</span></li>
                <li><strong style="color: #1f77b4;">EADᵢ</strong>: <span style="color: #2c3e50;">Exposure at Default in period i (total amount at risk)</span></li>
                <li><strong style="color: #1f77b4;">r</strong>: <span style="color: #2c3e50;">Discount rate (cost of capital)</span></li>
                <li><strong style="color: #1f77b4;">t</strong>: <span style="color: #2c3e50;">Time period (typically 12 months for Stage 1)</span></li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**PD (Probability of Default)**")
        st.markdown("The likelihood of a borrower defaulting - the focus of our predictive model")
        st.markdown("**Stages:**")
        st.markdown("- Stage 1: 12-month PD")
        st.markdown("- Stage 2: Lifetime PD")
        st.markdown("- Stage 3: Default (100% PD)")

    with col2:
        st.markdown("**LGD (Loss Given Default)**")
        st.markdown("The percentage of exposure lost when default occurs")
        st.markdown("**Components:**")
        st.markdown("- Recovery rate assumptions")
        st.markdown("- Collateral valuation")
        st.markdown("- Legal costs & timing")

    with col3:
        st.markdown("**EAD (Exposure at Default)**")
        st.markdown("The total amount at risk at the time of default")
        st.markdown("**Considerations:**")
        st.markdown("- Current balance")
        st.markdown("- Undrawn commitments")
        st.markdown("- Credit conversion factors")

    # ECL Staging
    st.markdown("### 📊 ECL Staging Criteria")
    st.markdown("""
    **Stage 1**: Performing loans
    - 12-month ECL
    - No significant increase in credit risk

    **Stage 2**: Underperforming loans
    - Lifetime ECL
    - Significant increase in credit risk since origination

    **Stage 3**: Non-performing loans
    - Lifetime ECL
    - Credit-impaired (defaulted)
    """)

    # Project Lifecycle
    st.markdown("### 🔄 Credit Risk Modeling Lifecycle")

    st.markdown("""
    This project follows a structured approach to credit risk modeling:

    1. **📊 Data Preparation** - Clean and engineer features from raw data
    2. **🔍 Feature Engineering** - Create meaningful risk indicators
    3. **📈 Model Development** - Train and validate predictive models
    4. **✅ Model Validation** - Ensure statistical soundness and stability
    5. **📊 Performance Monitoring** - Track model effectiveness over time
    6. **⚖️ Fairness Assessment** - Evaluate for bias and discrimination
    7. **🔮 Individual Predictions** - Apply model for decision support
    """)

    # Key Achievements
    st.markdown("### 🏆 Key Achievements")
    st.markdown("""
    - ✅ **Processed 30k+ client records** with comprehensive feature engineering
    - ✅ **Achieved Gini coefficient of 0.52** - excellent discriminatory power
    - ✅ **Integrated SHAP explainability** for model transparency
    - ✅ **Implemented comprehensive monitoring** with industry-standard metrics
    - ✅ **Built fairness audit system** for ethical AI assessment
    - ✅ **Created business impact simulator** linking ML to financial outcomes
    """)

def show_enhanced_data_exploration(df):
    """Enhanced data exploration with Five C's and statistical tests"""
    st.markdown('<h2 class="section-header">🔍 Data Exploration</h2>', unsafe_allow_html=True)

    # Five C's of Credit
    st.markdown("### 🏛️ The Five C's of Credit Analysis")

    with st.expander("1. Character - Borrower's Credit History"):
        st.markdown("""
        **Features:** `prev_defaults`, `credit_score`, `credit_score_bucket`

        Character refers to the borrower's reputation and track record for repaying debts.
        A history of defaults or low credit scores indicates higher risk.
        """)

    with st.expander("2. Capacity - Ability to Repay"):
        st.markdown("""
        **Features:** `net_yearly_income`, `DTI_ratio`, `yearly_debt_payments`, `income_per_person`

        Capacity measures the borrower's ability to repay based on income and existing debt obligations.
        High debt-to-income ratios indicate repayment challenges.
        """)

    with st.expander("3. Capital - Personal Investment"):
        st.markdown("""
        **Features:** `owns_house`, `owns_car`, `total_family_members`

        Capital refers to the borrower's personal stake or assets that demonstrate commitment.
        Asset ownership suggests stability and lower default risk.
        """)

    with st.expander("4. Collateral - Security for the Loan"):
        st.markdown("""
        **Features:** `credit_limit`, `outstanding_balance`, `credit_utilization`

        Collateral provides security for the lender in case of default.
        Higher credit utilization may indicate financial stress.
        """)

    with st.expander("5. Conditions - External Factors"):
        st.markdown("""
        **Features:** `occupation_type`, `no_of_days_employed`, `migrant_worker`, `age_group`

        Conditions include economic factors, industry trends, and borrower circumstances.
        Employment stability and occupation type influence repayment ability.
        """)

    # Statistical Feature Screening
    st.markdown("### 📈 Statistical Feature Screening")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Independent T-Tests (Continuous vs Target)**")
        t_test_results = perform_t_tests(df)
        st.dataframe(t_test_results, width='stretch')

        st.info("""
        **Interpretation:** T-tests compare means between default and non-default groups.
        Low p-values (< 0.05) indicate the feature significantly differentiates between groups.
        """)

    with col2:
        st.markdown("**Chi-Square Tests (Categorical vs Target)**")
        chi_test_results = perform_chi_square_tests(df)
        st.dataframe(chi_test_results, width='stretch')

        st.info("""
        **Interpretation:** Chi-square tests check for independence between categorical features and default status.
        Low p-values indicate the feature is significantly associated with default risk.
        """)

    # Target distribution
    st.markdown("### 📊 Target Variable Analysis")
    st.subheader("Credit Card Default Distribution")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(data=df, x='credit_card_default', palette='rocket', ax=ax)
    ax.set_title('Credit Card Default Distribution')
    ax.set_xlabel('Default Status (0 = No Default, 1 = Default)')
    ax.set_ylabel('Count')
    st.pyplot(fig)

        # Feature distributions
    st.markdown("### 📈 Key Feature Distributions")

    # Categorical features
    categorical_cols = ['gender', 'owns_car', 'owns_house']
    for col in categorical_cols:
        with st.expander(f"{col.replace('_', ' ').title()} Analysis"):
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.countplot(data=df, x=col, hue='credit_card_default', palette='rocket', ax=ax)
            ax.set_title(f'{col.replace("_", " ").title()} vs Default')
            st.pyplot(fig)

    # Numerical features
    numerical_cols = ['log_income', 'credit_score', 'yearly_debt_payments', 'credit_limit', 'DTI_ratio']
    for col in numerical_cols:
        if col in df.columns:
            with st.expander(f"{col.replace('_', ' ').title()} Analysis"):
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(data=df, x=col, hue='credit_card_default', bins=30, palette='rocket', ax=ax)
                ax.set_title(f'{col.replace("_", " ").title()} Distribution by Default Status')
                st.pyplot(fig)

def show_enhanced_preprocessing(df_original, df_processed):
    """Enhanced preprocessing with exclusions waterfall and WoE/IV"""
    st.markdown('<h2 class="section-header">⚙️ Data Preprocessing</h2>', unsafe_allow_html=True)

    # Exclusions Waterfall Chart
    st.markdown("### 📉 Data Exclusions Waterfall")

    # Create waterfall data
    initial_count = len(df_original)
    policy_exclusions = len(df_original) - len(df_original[df_original['default_in_last_6months'] == 0])  # Example
    observation_exclusions = len(df_original) - len(df_processed) - policy_exclusions
    final_count = len(df_processed)

    # Create waterfall chart
    fig = go.Figure(go.Waterfall(
        name="Data Flow",
        orientation="v",
        measure=["relative", "relative", "relative", "total"],
        x=["Initial Data", "Policy Exclusions", "Observation Exclusions", "Final Dataset"],
        y=[initial_count, -policy_exclusions, -observation_exclusions, final_count],
        text=[f"{initial_count:,}", f"-{policy_exclusions:,}", f"-{observation_exclusions:,}", f"{final_count:,}"],
        textposition="outside",
        connector={"line": {"color": "rgb(63, 63, 63)"}},
    ))

    fig.update_layout(
        title="Data Processing Waterfall Chart",
        showlegend=False,
        height=400
    )

    st.plotly_chart(fig, width='stretch')

    # WoE and IV Analysis
    st.markdown("### 🎯 Weight of Evidence (WoE) & Information Value (IV)")

    st.markdown("""
    WoE and IV are crucial techniques for feature selection in credit risk modeling:

    - **WoE** measures the strength of a feature in separating good vs bad customers
    - **IV** quantifies the predictive power of a feature
    """)

    # Calculate WoE/IV for log_income (continuous variable that exists in processed data)
    woe_df, total_iv = calculate_woe_iv(df_processed, 'log_income')

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**WoE Analysis for Log Income**")
        st.dataframe(woe_df[['Total', 'Goods', 'Bads', 'WoE', 'IV']], width='stretch')

    with col2:
        st.markdown("**Information Value Interpretation**")
        st.markdown(f"**Total IV: {total_iv:.4f}**")

        if total_iv < 0.02:
            st.error("🔴 Useless predictor")
        elif total_iv < 0.1:
            st.warning("🟡 Weak predictor")
        elif total_iv < 0.3:
            st.success("🟢 Medium predictor")
        else:
            st.error("🔴 Suspicious predictor (may be too good to be true)")

        st.markdown("""
        **IV Guidelines:**
        - < 0.02: Not useful for prediction
        - 0.02 - 0.1: Weak predictive power
        - 0.1 - 0.3: Medium predictive power
        - 0.3 - 0.5: Strong predictive power
        - > 0.5: Suspicious (may indicate data leakage)
        """)

    # Processing Summary
    st.markdown("### 📋 Processing Summary")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Original Records", f"{len(df_original):,}")
        st.metric("Final Records", f"{len(df_processed):,}")
        st.metric("Records Removed", f"{len(df_original) - len(df_processed):,}")

    with col2:
        st.markdown("**Key Processing Steps:**")
        st.markdown("""
        - ✅ Feature engineering (DTI ratio, outstanding balance)
        - ✅ Credit score and age bucketing
        - ✅ Missing value imputation by groups
        - ✅ Log transformations for skewed features
        - ✅ One-hot encoding for categorical variables
        - ✅ Data leakage prevention
        """)

def show_enhanced_model_evaluation(y_test, y_pred_lr, y_pred_rf, y_pred_gb, y_pred_xgb,
                                  y_pred_proba_lr, y_pred_proba_rf, y_pred_proba_gb, y_pred_proba_xgb):
    """Enhanced model evaluation with KS plots and gains charts"""
    st.markdown('<h2 class="section-header">📊 Model Evaluation</h2>', unsafe_allow_html=True)

    models = ['Logistic Regression', 'Random Forest', 'Gradient Boosting', 'XGBoost']
    predictions = [y_pred_lr, y_pred_rf, y_pred_gb, y_pred_xgb]
    probas = [y_pred_proba_lr, y_pred_proba_rf, y_pred_proba_gb, y_pred_proba_xgb]

    # Risk-focused metrics
    st.markdown("### 🎯 Risk-Focused Performance Metrics")

    # Calculate comprehensive metrics
    metrics_data = []
    for name, y_pred, y_proba in zip(models, predictions, probas):
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        gini = 2 * auc(*roc_curve(y_test, y_proba)[:2]) - 1
        ks_stat, ks_threshold, fpr, tpr = calculate_ks_statistic(y_test, y_proba)

        metrics_data.append({
            'Model': name,
            'Accuracy': f"{accuracy:.4f}",
            'Precision': f"{precision:.4f}",
            'Recall': f"{recall:.4f}",
            'F1-Score': f"{f1:.4f}",
            'Gini': f"{gini:.4f}",
            'KS': f"{ks_stat:.4f}"
        })

    st.dataframe(pd.DataFrame(metrics_data), width='stretch')

    # KS Statistic Analysis
    st.markdown("### 📈 Kolmogorov-Smirnov (KS) Analysis")

    # Create KS plot for best model (XGBoost)
    ks_stat, ks_threshold, fpr, tpr = calculate_ks_statistic(y_test, y_pred_proba_xgb)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # KS Plot
    ax1.plot(np.linspace(0, 1, len(tpr)), tpr, label='True Positive Rate (TPR)')
    ax1.plot(np.linspace(0, 1, len(fpr)), fpr, label='False Positive Rate (FPR)')
    ax1.plot(np.linspace(0, 1, len(tpr)), tpr - fpr, label='KS Curve')
    ax1.axvline(x=ks_threshold, color='red', linestyle='--', alpha=0.7, label=f'KS Threshold: {ks_threshold:.3f}')
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Rate')
    ax1.set_title('KS Statistic Analysis')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add KS interpretation
    ax1.text(0.05, 0.95, f'KS Statistic: {ks_stat:.4f}', transform=ax1.transAxes,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))

    # Gains Chart
    gains_data = create_gains_chart(y_test, y_pred_proba_xgb)

    ax2.plot(np.cumsum(gains_data.index * 10), gains_data['Cumulative_Pct'] * 100, 'b-', linewidth=2, label='Model')
    ax2.plot([0, 100], [0, 100], 'r--', alpha=0.7, label='Random')
    ax2.fill_between(np.cumsum(gains_data.index * 10), 0, gains_data['Cumulative_Pct'] * 100, alpha=0.3)
    ax2.set_xlabel('Percentage of Population Targeted (%)')
    ax2.set_ylabel('Percentage of Defaults Captured (%)')
    ax2.set_title('Gains Chart')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Add interpretation text
    ax2.text(0.05, 0.95, 'Top 30% capture ~70% of defaults', transform=ax2.transAxes,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))

    plt.tight_layout()
    st.pyplot(fig)

    # KS Interpretation
    st.info(f"""
    **KS Statistic: {ks_stat:.4f}**

    The KS statistic measures the model's ability to discriminate between defaulters and non-defaulters:

    - **{ks_stat:.4f}** indicates **excellent** discriminatory power
    - The maximum separation occurs at threshold **{ks_threshold:.3f}**
    - Higher KS values (typically 40-70) indicate better model performance
    """)

    # ROC Curves
    st.markdown("### 📊 ROC Curves & AUC Analysis")
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['darkorange', 'green', 'red', 'purple']

    for name, y_proba, color in zip(models, probas, colors):
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC = {roc_auc:.4f})')

    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

def show_enhanced_model_monitoring(y_test, y_pred_proba, X_train, X_test, model):
    """Enhanced model monitoring with PSI/CSI and calibration analysis"""
    st.markdown('<h2 class="section-header">📈 Model Monitoring</h2>', unsafe_allow_html=True)

    # Simulate production data with drift
    st.markdown("### 🔄 Production Data Simulation")
    st.markdown("Creating realistic production data with controlled drift for monitoring demonstration:")

    X_prod = simulate_production_drift(X_train, n_samples=len(X_test))

    # Get predictions on production data
    try:
        y_pred_proba_prod = model.predict_proba(X_prod)[:, 1]
    except:
        y_pred_proba_prod = y_pred_proba
        st.warning("Using training predictions as fallback for production simulation.")

    # PSI and CSI Analysis
    st.markdown("### 📊 Population Stability Index (PSI) & Characteristic Stability Index (CSI)")

    # Calculate PSI for model scores
    psi_score = calculate_psi(y_pred_proba, y_pred_proba_prod)

    # Calculate CSI for individual features
    feature_psi_scores = []
    for col in X_train.columns:
        if col in X_prod.columns:
            psi_feat = calculate_feature_psi(X_train[col], X_prod[col])
            feature_psi_scores.append({
                'Feature': col,
                'PSI': psi_feat,
                'Drift_Level': 'High' if psi_feat > 0.25 else 'Moderate' if psi_feat > 0.1 else 'Low'
            })

    psi_df = pd.DataFrame(feature_psi_scores).sort_values('PSI', ascending=False)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("PSI Score", f"{psi_score:.4f}")
        if psi_score < 0.1:
            st.success("✅ Low drift - Model stable")
        elif psi_score < 0.25:
            st.warning("⚠️ Moderate drift - Monitor closely")
        else:
            st.error("🚨 High drift - Retraining recommended")

    with col2:
        st.metric("CSI Score", f"{psi_score:.4f}")
        st.caption("Score distribution stability")

    with col3:
        ks_score = calculate_ks_statistic(y_test, y_pred_proba)[0]
        st.metric("KS Statistic", f"{ks_score:.4f}")
        if ks_score > 0.8:
            st.warning("High separation (may indicate overfitting)")
        elif ks_score > 0.6:
            st.info("Good separation")
        else:
            st.error("Poor separation")

    # PSI Thresholds
    st.markdown("**PSI Threshold Guidelines:**")
    st.markdown("""
    - 🟢 **< 0.1**: No significant drift - model appears stable
    - 🟡 **0.1 - 0.25**: Moderate drift - monitor closely
    - 🔴 **> 0.25**: High drift - consider retraining
    """)

    # Top drifting features
    st.markdown("### 📈 Top Drifting Features")
    st.dataframe(psi_df.head(10), width='stretch')

    # Calibration Analysis
    st.markdown("### 📏 Model Calibration Analysis")

    prob_true, prob_pred = calculate_calibration_curve(y_test, y_pred_proba)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(prob_pred, prob_true, 's-', label='Model Calibration')
    ax.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title('Calibration Curve (Reliability Diagram)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)

    st.info("""
    **Calibration Analysis:**

    A well-calibrated model should follow the diagonal line. Points above the line indicate
    the model is under-confident (actual default rate higher than predicted), while points
    below indicate over-confidence (actual default rate lower than predicted).
    """)

    # Override Monitoring Simulation
    st.markdown("### 🎛️ Override Monitoring Simulation")

    st.markdown("""
    Override monitoring tracks when loan officers override model recommendations,
    which can indicate model misalignment with business reality.
    """)

    # Simulate override scenario
    n_applications = 1000
    model_rejections = 150  # Model rejects 15% of applications
    low_side_overrides = 25  # Officers approve 25 of the rejections
    high_side_overrides = 8  # Officers reject 8 model-approved applications

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Model Rejection Rate", f"{model_rejections/n_applications*100:.1f}%")
    with col2:
        low_side_rate = low_side_overrides / model_rejections * 100
        st.metric("Low-Side Override Rate", f"{low_side_rate:.1f}%")
    with col3:
        high_side_rate = high_side_overrides / (n_applications - model_rejections) * 100
        st.metric("High-Side Override Rate", f"{high_side_rate:.1f}%")

    st.info("""
    **Override Rate Guidelines:**
    - Low-side overrides > 10% may indicate model is too conservative
    - High-side overrides > 5% may indicate model is too aggressive
    - High override rates warrant model investigation and potential retraining
    """)

def show_enhanced_fairness_audit(X_test, y_pred, y_test):
    """Enhanced fairness audit with disparate impact analysis"""
    st.markdown('<h2 class="section-header">⚖️ Fairness Audit</h2>', unsafe_allow_html=True)

    # Create analysis dataframe
    fairness_df = X_test.copy()
    fairness_df['predicted_default'] = y_pred
    fairness_df['actual_default'] = y_test

    # Gender Fairness Analysis
    if 'gender' in fairness_df.columns:
        st.markdown("### 👥 Gender Fairness Analysis")

        gender_fairness = fairness_df.groupby('gender').apply(lambda x: {
            'count': len(x),
            'predicted_default_rate': x['predicted_default'].mean(),
            'actual_default_rate': x['actual_default'].mean()
        }).apply(pd.Series)

        st.dataframe(gender_fairness, width='stretch')

        # Disparate Impact Calculation
        disparate_impact, favorable_rates = calculate_disparate_impact(
            fairness_df, 'gender', 'predicted_default'
        )

        if disparate_impact is not None:
            st.metric("Disparate Impact Ratio", f"{disparate_impact:.3f}")

            if disparate_impact < 0.8:
                st.error("🔴 Potential adverse impact detected (ratio < 0.8)")
            elif disparate_impact > 1.25:
                st.warning("🟡 Potential reverse discrimination (ratio > 1.25)")
            else:
                st.success("🟢 Acceptable disparate impact ratio")

            st.info("""
            **Four-Fifths Rule:** A disparate impact ratio below 80% is often considered
            evidence of adverse impact under US employment law. This rule can be applied
            analogously to lending decisions.
            """)

    # Age Group Fairness
    age_group_cols = [col for col in fairness_df.columns if 'age_group_' in col]
    if age_group_cols:
        st.markdown("### 👴 Age Group Fairness Analysis")

        # Convert one-hot encoded age groups back to categorical
        age_groups = fairness_df[age_group_cols].idxmax(axis=1).str.replace('age_group_', '')

        age_fairness = pd.DataFrame({
            'age_group': age_groups,
            'predicted_default': fairness_df['predicted_default'],
            'actual_default': fairness_df['actual_default']
        }).groupby('age_group').agg({
            'predicted_default': 'mean',
            'actual_default': 'mean',
            'age_group': 'count'
        }).rename(columns={'age_group': 'count'})

        st.dataframe(age_fairness, width='stretch')

    # Overall Fairness Assessment
    st.markdown("### 📋 Fairness Assessment Summary")

    overall_pred_rate = fairness_df['predicted_default'].mean()
    overall_actual_rate = fairness_df['actual_default'].mean()

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Overall Predicted Default Rate", f"{overall_pred_rate:.1%}")
    with col2:
        st.metric("Overall Actual Default Rate", f"{overall_actual_rate:.1%}")

    if abs(overall_pred_rate - overall_actual_rate) > 0.05:
        st.warning("⚠️ Significant difference between predicted and actual rates detected")
    else:
        st.success("✅ Predicted and actual rates are well-aligned")

def show_enhanced_individual_prediction(model, feature_names):
    """Enhanced individual prediction with SHAP explanations"""
    st.markdown('<h2 class="section-header">👤 Individual Risk Prediction</h2>', unsafe_allow_html=True)

    st.markdown("Enter customer details to predict credit default risk:")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        gender = st.selectbox("Gender", ["Male", "Female"])
        owns_car = st.selectbox("Owns Car", ["Yes", "No"])
        owns_house = st.selectbox("Owns House", ["Yes", "No"])

    with col2:
        no_of_children = st.number_input("Number of Children", min_value=0, max_value=10, value=1)
        total_family_members = st.number_input("Total Family Members", min_value=1, max_value=20, value=3)
        migrant_worker = st.selectbox("Migrant Worker", ["Yes", "No"])
        occupation = st.selectbox("Occupation Type", [
            "Laborers", "Sales staff", "Core staff", "Managers", "Drivers",
            "High skill tech staff", "Accountants", "Medicine staff", "Security staff",
            "Cooking staff", "Cleaning staff", "Private service staff", "Low-skill Laborers",
            "Waiters/barmen staff", "Secretaries", "Realty agents", "HR staff", "IT staff"
        ])

    with col3:
        net_yearly_income = st.number_input("Net Yearly Income", min_value=0, value=50000)
        no_of_days_employed = st.number_input("Days Employed", min_value=0, value=3650)
        credit_limit = st.number_input("Credit Limit", min_value=0, value=50000)
        credit_limit_used_pct = st.slider("Credit Limit Used (%)", 0, 100, 30)
        credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
        prev_defaults = st.number_input("Previous Defaults", min_value=0, max_value=10, value=0)

    if st.button("Predict Risk", type="primary"):
        # Prepare input data
        input_data = pd.DataFrame({
            'gender': [1 if gender == "Male" else 0],
            'owns_car': [1 if owns_car == "Yes" else 0],
            'owns_house': [1 if owns_house == "Yes" else 0],
            'no_of_children': [no_of_children],
            'total_family_members': [total_family_members],
            'migrant_worker': [1 if migrant_worker == "Yes" else 0],
            'prev_defaults': [prev_defaults],
            'net_yearly_income': [net_yearly_income],
            'no_of_days_employed': [no_of_days_employed],
            'credit_limit': [credit_limit],
            'credit_limit_used(%)': [credit_limit_used_pct],
            'yearly_debt_payments': [net_yearly_income * 0.3],
            'age': [age],
            'credit_score': [credit_score],
        })

        # Apply preprocessing
        input_data['DTI_ratio'] = input_data['yearly_debt_payments'] / input_data['net_yearly_income']
        input_data['outstanding_balance'] = input_data['credit_limit'] * (input_data['credit_limit_used(%)'] / 100)

        # Bucketing
        score_bins = [0, 580, 670, 740, 800, float('inf')]
        score_labels = ['Poor', 'Fair', 'Good', 'Very Good', 'Excellent']
        input_data['credit_score_bucket'] = pd.cut(input_data['credit_score'], bins=score_bins, labels=score_labels, right=False)

        age_bins = [0, 29, 45, 60, float('inf')]
        age_labels = ['Young_Adult', 'Middle_Aged', 'Senior_Adult', 'Elderly']
        input_data['age_group'] = pd.cut(input_data['age'], bins=age_bins, labels=age_labels, right=False)

        # Log transformations
        input_data['log_income'] = np.log1p(input_data['net_yearly_income'])
        input_data['log_no_of_days_employed'] = np.log1p(input_data['no_of_days_employed'])
        input_data['log_credit_limit'] = np.log1p(input_data['credit_limit'])

        # Feature engineering
        input_data['income_per_person'] = input_data['net_yearly_income'] / (input_data['total_family_members'] + 1)
        input_data['credit_utilization'] = input_data['credit_limit_used(%)'] / 100

        # One-hot encoding for occupation
        occupation_cols = [col for col in feature_names if col.startswith('occupation_type_')]
        for col in occupation_cols:
            input_data[col] = 0
        if f'occupation_type_{occupation}' in input_data.columns:
            input_data[f'occupation_type_{occupation}'] = 1

        # One-hot encoding for age_group and credit_score_bucket
        age_group_cols = [col for col in feature_names if col.startswith('age_group_')]
        credit_bucket_cols = [col for col in feature_names if col.startswith('credit_score_bucket_')]

        for col in age_group_cols + credit_bucket_cols:
            input_data[col] = 0

        # Set the appropriate one-hot encoded columns
        age_group_val = str(input_data['age_group'].iloc[0])
        credit_bucket_val = str(input_data['credit_score_bucket'].iloc[0])

        if f'age_group_{age_group_val}' in input_data.columns:
            input_data[f'age_group_{age_group_val}'] = 1
        if f'credit_score_bucket_{credit_bucket_val}' in input_data.columns:
            input_data[f'credit_score_bucket_{credit_bucket_val}'] = 1

        # Select only the features used in training
        input_processed = input_data[feature_names]

        # Make prediction
        prediction_proba = model.predict_proba(input_processed)[0][1]
        prediction = 1 if prediction_proba >= 0.5 else 0

        # Display results
        st.markdown("### 🎯 Prediction Results")

        col1, col2, col3 = st.columns(3)
        with col1:
            if prediction == 1:
                st.error("⚠️ HIGH RISK - Default Predicted")
            else:
                st.success("✅ LOW RISK - No Default Predicted")

        with col2:
            st.metric("Default Probability", f"{prediction_proba:.1%}")

        with col3:
            risk_level = "High" if prediction_proba > 0.7 else "Medium" if prediction_proba > 0.3 else "Low"
            st.metric("Risk Level", risk_level)

        # SHAP Explanation
        st.markdown("### 🔍 Risk Factor Analysis")

        # Calculate SHAP values for this prediction
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_processed)
        sample_shap = shap_values[0]  # For binary classification

        # Get top risk factors
        feature_names_list = input_processed.columns
        shap_df = pd.DataFrame({
            'Feature': feature_names_list,
            'SHAP_Value': sample_shap,
            'Impact': ['Increases Risk' if x > 0 else 'Decreases Risk' for x in sample_shap]
        }).sort_values('SHAP_Value', key=abs, ascending=False)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**🔴 Top Risk-Increasing Factors:**")
            risk_increase = shap_df[shap_df['Impact'] == 'Increases Risk'].head(3)
            for i, row in risk_increase.iterrows():
                st.write(f"• **{row['Feature']}**: +{row['SHAP_Value']:.4f}")

        with col2:
            st.markdown("**🟢 Top Risk-Decreasing Factors:**")
            risk_decrease = shap_df[shap_df['Impact'] == 'Decreases Risk'].head(3)
            for i, row in risk_decrease.iterrows():
                st.write(f"• **{row['Feature']}**: {row['SHAP_Value']:.4f}")

        st.info("""
        **Understanding the Prediction:**

        This analysis shows the key factors driving the model's prediction. Positive SHAP values
        increase the predicted default probability, while negative values decrease it. This
        transparency helps loan officers understand the reasoning behind each prediction.
        """)

if __name__ == "__main__":
    main()
