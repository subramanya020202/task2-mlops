from scipy.stats import ks_2samp, anderson_ksamp, wasserstein_distance, chi2_contingency
from scipy.spatial.distance import jensenshannon
import pandas as pd
import numpy as np

def run_anderson_darling(reference_data, current_data):
    """
    Runs the Anderson-Darling test to check if two numeric datasets have the same distribution.

    Parameters:
        reference_data (pd.Series): Historical dataset for comparison.
        current_data (pd.Series): Current dataset to be analyzed for drift.

    Returns:
        tuple: A tuple containing the Anderson-Darling statistic and the test name.
    """
    try:
        ad_stat, _, _ = anderson_ksamp([reference_data, current_data])
        return ad_stat, "Anderson-Darling Test"
    except Exception as e:
        return None, f"Anderson-Darling Test Failed: {e}"

def run_ks_test(reference_data, current_data):
    """
    Runs the Kolmogorov-Smirnov test to compare the distributions of two datasets.

    Parameters:
        reference_data (pd.Series): Historical dataset for comparison.
        current_data (pd.Series): Current dataset to be analyzed for drift.

    Returns:
        tuple: A tuple containing the Kolmogorov-Smirnov statistic and the test name.
    """
    ks_stat, _ = ks_2samp(reference_data, current_data)
    return ks_stat, "Kolmogorov-Smirnov Test"

def run_wasserstein_distance(reference_data, current_data):
    """
    Computes the Wasserstein distance to measure the difference between two numeric distributions.

    Parameters:
        reference_data (pd.Series): Historical dataset for comparison.
        current_data (pd.Series): Current dataset to be analyzed for drift.

    Returns:
        tuple: A tuple containing the normalized Wasserstein distance and the test name.
    """
    threshold = 0.1
    norm = max(np.std(reference_data), 0.001)
    wd_norm_value = wasserstein_distance(reference_data, current_data) / norm
    return wd_norm_value, "Wasserstein Distance"

def run_js_divergence(reference_data, current_data):
    """
    Computes the Jensen-Shannon divergence to quantify the difference between two probability distributions.

    Parameters:
        reference_data (pd.Series): Historical dataset for comparison.
        current_data (pd.Series): Current dataset to be analyzed for drift.

    Returns:
        tuple: A tuple containing the Jensen-Shannon divergence and the test name.
    """
    ref_prob = reference_data.value_counts(normalize=True)
    curr_prob = current_data.value_counts(normalize=True)
    all_categories = ref_prob.index.union(curr_prob.index)
    ref_prob = ref_prob.reindex(all_categories, fill_value=0)
    curr_prob = curr_prob.reindex(all_categories, fill_value=0)
    js_divergence = jensenshannon(ref_prob, curr_prob)
    return js_divergence, "Jensen-Shannon Divergence"

def run_chi_squared_test(reference_data, current_data):
    """
    Runs the Chi-Squared test to analyze the independence between two categorical datasets.

    Parameters:
        reference_data (pd.Series): Historical dataset for comparison.
        current_data (pd.Series): Current dataset to be analyzed for drift.

    Returns:
        tuple: A tuple containing the Chi-Squared statistic and the test name.
    """
    contingency_table = pd.crosstab(reference_data, current_data)
    chi2_stat, _, _, _ = chi2_contingency(contingency_table)
    return chi2_stat, "Chi-Squared Test"