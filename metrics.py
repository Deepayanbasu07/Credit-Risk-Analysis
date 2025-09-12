import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt

def calculate_gini(y_true, y_scores):
    """
    Calculate Gini coefficient from true labels and predicted scores.
    Gini = 2 * AUC - 1
    """
    auc = roc_auc_score(y_true, y_scores)
    gini = 2 * auc - 1
    return gini

def calculate_ks(y_true, y_scores):
    """
    Calculate Kolmogorov-Smirnov statistic.
    KS = max difference between cumulative good and bad rates.
    """
    # Sort by scores descending
    data = pd.DataFrame({'y_true': y_true, 'y_scores': y_scores})
    data = data.sort_values('y_scores', ascending=False)
    
    # Cumulative good (non-default) and bad (default)
    data['cum_good'] = (1 - data['y_true']).cumsum() / (1 - data['y_true']).sum()
    data['cum_bad'] = data['y_true'].cumsum() / data['y_true'].sum()
    
    ks = (data['cum_bad'] - data['cum_good']).max()
    return ks

def calculate_psi(expected, actual, bins=10):
    """
    Calculate Population Stability Index (PSI) between expected and actual distributions.
    """
    # Bin the data
    expected_bins = pd.cut(expected, bins=bins, duplicates='drop')
    actual_bins = pd.cut(actual, bins=bins, duplicates='drop')
    
    # Calculate proportions
    expected_prop = (expected_bins.value_counts() / len(expected_bins)).sort_index()
    actual_prop = (actual_bins.value_counts() / len(actual_bins)).sort_index()
    
    # Align indices
    all_bins = expected_prop.index.union(actual_prop.index)
    expected_prop = expected_prop.reindex(all_bins, fill_value=0)
    actual_prop = actual_prop.reindex(all_bins, fill_value=0)
    
    # Avoid division by zero
    expected_prop = expected_prop.replace(0, 1e-10)
    
    psi = ((actual_prop - expected_prop) * np.log(actual_prop / expected_prop)).sum()
    return psi

def calculate_csi(feature_expected, feature_actual, bins=10):
    """
    Calculate Characteristic Stability Index (CSI) for a feature.
    Similar to PSI but for feature distributions.
    """
    return calculate_psi(feature_expected, feature_actual, bins)

def get_psi_csi_table(expected, actual, bins=10):
    """
    Get a table with PSI/CSI breakdown by decile.
    """
    # Bin the data
    expected_bins = pd.cut(expected, bins=bins, duplicates='drop')
    actual_bins = pd.cut(actual, bins=bins, duplicates='drop')
    
    # Calculate proportions
    expected_prop = (expected_bins.value_counts() / len(expected_bins)).sort_index()
    actual_prop = (actual_bins.value_counts() / len(actual_bins)).sort_index()
    
    # Align indices
    all_bins = expected_prop.index.union(actual_prop.index)
    expected_prop = expected_prop.reindex(all_bins, fill_value=0)
    actual_prop = actual_prop.reindex(all_bins, fill_value=0)
    
    # Calculate PSI per bin
    psi_per_bin = (actual_prop - expected_prop) * np.log((actual_prop + 1e-10) / (expected_prop + 1e-10))
    
    table = pd.DataFrame({
        'Bin': all_bins,
        'Expected %': expected_prop * 100,
        'Actual %': actual_prop * 100,
        'PSI Contribution': psi_per_bin
    })
    
    return table

def plot_distribution_shift(expected, actual, title="Distribution Shift"):
    """
    Plot bar chart for distribution shift.
    """
    bins = 10
    expected_bins = pd.cut(expected, bins=bins, duplicates='drop')
    actual_bins = pd.cut(actual, bins=bins, duplicates='drop')
    
    expected_prop = (expected_bins.value_counts() / len(expected_bins)).sort_index()
    actual_prop = (actual_bins.value_counts() / len(actual_bins)).sort_index()
    
    all_bins = expected_prop.index.union(actual_prop.index)
    expected_prop = expected_prop.reindex(all_bins, fill_value=0)
    actual_prop = actual_prop.reindex(all_bins, fill_value=0)
    
    fig, ax = plt.subplots()
    width = 0.35
    x = np.arange(len(all_bins))
    ax.bar(x - width/2, expected_prop, width, label='Expected')
    ax.bar(x + width/2, actual_prop, width, label='Actual')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{i.left:.2f}-{i.right:.2f}' for i in all_bins], rotation=45)
    ax.set_ylabel('Proportion')
    ax.set_title(title)
    ax.legend()
    return fig