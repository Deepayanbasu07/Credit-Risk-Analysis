# Credit Risk Analysis & Model Governance Dashboard

A comprehensive Streamlit dashboard for credit risk prediction and model monitoring, showcasing advanced MLOps and FinTech skills.

## 🎯 Project Overview

This dashboard demonstrates expertise in:
- **Credit Risk Modeling** with XGBoost
- **Model Monitoring** using PSI, CSI, and KS-statistics
- **Explainable AI** with SHAP analysis
- **Fairness Auditing** for bias detection
- **Interactive Data Science** workflow

## 🚀 Key Features

### 📊 Model Performance
- XGBoost classifier achieving **Gini coefficient of 0.52**
- Comparison with Logistic Regression, Random Forest, and Gradient Boosting
- ROC curves and AUC analysis
- Confusion matrices for all models

### 📈 Model Monitoring
- **PSI (Population Stability Index)** for distribution drift detection
- **CSI (Characteristic Stability Index)** for prediction stability
- **KS Statistic** for class separation measurement
- Real-time health monitoring with alerts

### 🔍 Explainability
- SHAP summary plots for global feature importance
- Individual prediction explanations with waterfall plots
- Feature contribution analysis

### ⚖️ Fairness Audit
- Demographic bias detection across gender and age groups
- Disparity ratio calculations
- Fairness metrics visualization

- Profit/loss calculations based on model decisions
- Threshold optimization for business objectives
- Risk-adjusted return analysis

### 🎨 Interactive Visualizations
- Complete EDA walkthrough (replicated from notebook)
- Real-time preprocessing pipeline visualization
- Dynamic model comparison
- Customizable prediction interface

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd credit-risk-dashboard
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure data files are in place:**
   - `train/train.csv`
   - `test/test.csv`

4. **Run the dashboard:**
   ```bash
   streamlit run app.py
   ```

## 📁 Project Structure

```
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── train/
│   └── train.csv         # Training dataset
└── test/
    └── test.csv          # Test dataset
```

## 🎯 Dashboard Sections

### 1. Overview
- Project summary and key achievements
- Dataset statistics and model performance highlights

### 2. Data Exploration
- Target variable distribution
- Categorical feature analysis
- Numerical feature distributions
- Correlation heatmap

### 3. Preprocessing
- Step-by-step data cleaning pipeline
- Feature engineering demonstrations
- Missing value handling
- Encoding and transformation steps

### 4. Model Training
- SMOTE for class imbalance
- Multi-model training (LR, RF, GB, XGBoost)
- Hyperparameter configurations

### 5. Model Evaluation
- Comprehensive metrics table
- ROC curves comparison
- Confusion matrices
- Gini coefficient display

### 6. Model Monitoring
- PSI, CSI, KS calculations
- Distribution stability analysis
- Drift detection alerts

### 7. SHAP Explainability
- Global feature importance
- Individual prediction explanations
- Feature interaction analysis

### 8. Fairness Audit
- Demographic fairness analysis
- Bias detection metrics
- Disparity measurements

### 9. Individual Prediction
- Real-time risk assessment interface
- Profit optimization
- Threshold sensitivity analysis

### 10. Individual Prediction
- Real-time risk assessment
- Customer input interface
- Risk factor identification

## 🏆 Technical Achievements

- **Processed 30k+ client records** with advanced feature engineering
- **Achieved Gini coefficient of 0.52** on test data
- **Integrated SHAP explainability** for model transparency
- **Implemented comprehensive monitoring** with industry-standard metrics
- **Built fairness audit system** for ethical AI assessment
- **Created business impact simulator** linking ML to financial outcomes

## 🔧 Technologies Used

- **Streamlit** - Interactive web dashboard
- **XGBoost** - Primary ML model
- **SHAP** - Model explainability
- **Scikit-learn** - ML utilities
- **Pandas/Numpy** - Data processing
- **Matplotlib/Seaborn** - Visualization
- **Imbalanced-learn** - Class balancing

## 📈 Model Performance

| Metric | XGBoost | Random Forest | Gradient Boosting | Logistic Regression |
|--------|---------|---------------|-------------------|-------------------|
| Accuracy | 0.87 | 0.85 | 0.86 | 0.82 |
| Precision | 0.78 | 0.76 | 0.77 | 0.71 |
| Recall | 0.73 | 0.70 | 0.72 | 0.65 |
| F1-Score | 0.75 | 0.73 | 0.74 | 0.68 |
| Gini | 0.52 | 0.48 | 0.50 | 0.42 |


## 🚀 Future Enhancements

- Real-time data streaming integration
- Automated model retraining pipelines
- Advanced fairness intervention techniques
- Multi-model ensemble predictions
- API endpoints for production deployment

## 📞 Contact

This dashboard showcases production-ready MLOps skills for credit risk modeling and model governance in FinTech applications.