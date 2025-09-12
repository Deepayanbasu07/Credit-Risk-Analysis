import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
from metrics import calculate_gini, calculate_ks, calculate_psi, calculate_csi, get_psi_csi_table, plot_distribution_shift

# Load the trained model and test data
@st.cache_resource
def load_model():
    return joblib.load('xgb_model.pkl')

@st.cache_data
def load_data():
    X_test = pd.read_csv('X_test.csv')
    y_test = pd.read_csv('y_test.csv').values.ravel()
    return X_test, y_test

model = load_model()
X_test, y_test = load_data()

# Get predictions on test data
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba > 0.5).astype(int)

# Sidebar navigation
st.sidebar.title("Credit Risk Model Monitoring Dashboard")

# Navigation buttons
if st.sidebar.button("Home"):
    st.session_state.page = "Home"
    st.rerun()
if st.sidebar.button("Model Performance"):
    st.session_state.page = "Model Performance"
    st.rerun()
if st.sidebar.button("Stability Analysis"):
    st.session_state.page = "Stability Analysis"
    st.rerun()
if st.sidebar.button("Business Impact Analysis"):
    st.session_state.page = "Business Impact Analysis"
    st.rerun()
if st.sidebar.button("Fairness & Bias Audit"):
    st.session_state.page = "Fairness & Bias Audit"
    st.rerun()

page = st.session_state.get('page', "Home")

# Home Page
if page == "Home":
    st.title("Credit Risk Model Monitoring Dashboard")
    st.write("""
    This dashboard provides comprehensive monitoring for a credit risk prediction model.
    It includes performance metrics, stability analysis, business impact assessment, and fairness audits.
    """)
    st.write("**Purpose:** Monitor model performance, detect drifts, evaluate business implications, and ensure fairness.")
    if st.button("Get Started"):
        st.session_state.page = "Model Performance"
        st.rerun()

# Model Performance Page
elif page == "Model Performance":
    st.title("Model Performance")
    st.write("Key performance metrics on the test dataset:")

    auc = roc_auc_score(y_test, y_pred_proba)
    gini = calculate_gini(y_test, y_pred_proba)
    ks = calculate_ks(y_test, y_pred_proba)

    col1, col2, col3 = st.columns(3)
    col1.metric("AUC", f"{auc:.3f}")
    col2.metric("Gini Coefficient", f"{gini:.3f}")
    col3.metric("KS Statistic", f"{ks:.3f}")

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm = ConfusionMatrixDisplay(cm, display_labels=['No Default', 'Default']).plot(cmap='Blues').figure_
    st.pyplot(fig_cm)

    st.subheader("ROC Curve")
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    fig_roc, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend()
    st.pyplot(fig_roc)

# Stability Analysis Page
elif page == "Stability Analysis":
    st.title("Stability Analysis (PSI/CSI)")
    st.write("Monitor model stability by comparing predictions and feature distributions.")

    uploaded_file = st.file_uploader("Upload new CSV file for monitoring", type="csv")
    if uploaded_file is not None:
        new_df = pd.read_csv(uploaded_file)
        st.write("Using uploaded data.")
    else:
        new_df = pd.read_csv('new_data.csv')
        st.write("Using default new_data.csv for demonstration.")

    # Ensure new_df has the same columns
    if set(new_df.columns) != set(X_test.columns):
        st.error("Uploaded file must have the same columns as the training data.")
    else:
        new_pred_proba = model.predict_proba(new_df)[:, 1]

        # PSI on predictions
        psi_pred = calculate_psi(y_pred_proba, new_pred_proba)
        st.metric("PSI on Predictions", f"{psi_pred:.3f}")

        # Select feature for CSI
        feature = st.selectbox("Select Feature for CSI", new_df.columns.tolist())
        csi = calculate_csi(X_test[feature], new_df[feature])
        st.metric(f"CSI for {feature}", f"{csi:.3f}")

        # Table with breakdown
        st.subheader("PSI/CSI Breakdown by Decile")
        table = get_psi_csi_table(X_test[feature], new_df[feature])
        st.dataframe(table)

        # Bar chart
        st.subheader("Distribution Shift Visualization")
        fig_shift = plot_distribution_shift(X_test[feature], new_df[feature], f"Distribution Shift for {feature}")
        st.pyplot(fig_shift)

# Business Impact Analysis Page
elif page == "Business Impact Analysis":
    st.title("Business Impact Analysis")
    st.write("Analyze business metrics based on classification threshold.")

    threshold = st.slider("Classification Threshold", 0.0, 1.0, 0.5, 0.01)
    y_pred_thresh = (y_pred_proba > threshold).astype(int)

    approval_rate = 1 - y_pred_thresh.mean()
    default_rate = (y_pred_thresh * y_test).mean() / y_pred_thresh.mean() if y_pred_thresh.mean() > 0 else 0

    # Assumptions: Profit from good loan = 10, Loss from bad loan = 100
    profit_per_loan = (1 - y_test) * y_pred_thresh * 10 - y_test * y_pred_thresh * 100
    total_profit = profit_per_loan.sum()

    col1, col2, col3 = st.columns(3)
    col1.metric("Approval Rate", f"{approval_rate:.3f}")
    col2.metric("Default Rate", f"{default_rate:.3f}")
    col3.metric("Estimated Profit/Loss", f"{total_profit:.2f}")

    st.subheader("What-If Analysis: Simulate Data Drift")
    numeric_features = X_test.select_dtypes(include=[np.number]).columns.tolist()
    feature_whatif = st.selectbox("Select Feature for Drift Simulation", numeric_features)
    mean_adj = st.slider("Adjust Mean by (additive)", -2.0, 2.0, 0.0, 0.1)
    std_adj = st.slider("Adjust Std by (multiplicative)", 0.5, 2.0, 1.0, 0.1)

    # Simulate drifted data
    drifted_feature = X_test[feature_whatif] * std_adj + mean_adj
    X_drifted = X_test.copy()
    X_drifted[feature_whatif] = drifted_feature
    pred_proba_drifted = model.predict_proba(X_drifted)[:, 1]
    pred_drifted = (pred_proba_drifted > threshold).astype(int)

    approval_drifted = 1 - pred_drifted.mean()
    default_drifted = (pred_drifted * y_test).mean() / pred_drifted.mean() if pred_drifted.mean() > 0 else 0
    profit_drifted = ((1 - y_test) * pred_drifted * 10 - y_test * pred_drifted * 100).sum()

    st.write(f"**Drifted Approval Rate:** {approval_drifted:.3f}")
    st.write(f"**Drifted Default Rate:** {default_drifted:.3f}")
    st.write(f"**Drifted Profit/Loss:** {profit_drifted:.2f}")

# Fairness & Bias Audit Page
elif page == "Fairness & Bias Audit":
    st.title("Fairness & Bias Audit")
    st.write("Evaluate fairness across sensitive attributes.")

    sensitive_attr = st.selectbox("Select Sensitive Attribute", ['SEX', 'MARRIAGE'])
    groups = sorted(X_test[sensitive_attr].unique())

    approval_rates = {}
    tpr_rates = {}
    for group in groups:
        mask = X_test[sensitive_attr] == group
        pred_group = y_pred[mask]
        true_group = y_test[mask]
        approval_rates[group] = 1 - pred_group.mean()
        tp = ((pred_group == 1) & (true_group == 1)).sum()
        fn = ((pred_group == 0) & (true_group == 1)).sum()
        tpr_rates[group] = tp / (tp + fn) if (tp + fn) > 0 else 0

    st.subheader("Demographic Parity: Approval Rates by Group")
    fig_dp, ax = plt.subplots()
    ax.bar([str(k) for k in approval_rates.keys()], list(approval_rates.values()))
    ax.set_ylabel('Approval Rate')
    ax.set_title('Approval Rates by Group')
    st.pyplot(fig_dp)

    st.subheader("Equal Opportunity: True Positive Rates by Group")
    fig_eo, ax = plt.subplots()
    ax.bar([str(k) for k in tpr_rates.keys()], list(tpr_rates.values()))
    ax.set_ylabel('True Positive Rate')
    ax.set_title('True Positive Rates by Group')
    st.pyplot(fig_eo)