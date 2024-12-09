import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, entropy, wasserstein_distance
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

# 1. Mean and Variance Shift
def mean_variance_shift(reference_data, current_data):
    """
    Computes the mean and variance shift between the reference and current data.
    
    Parameters:
        reference_data (pd.Series): Historical dataset for comparison.
        current_data (pd.Series): Current dataset to be analyzed for drift.
    
    Returns:
        dict: A dictionary containing the mean and variance shifts.
    """
    mean_ref, mean_curr = reference_data.mean(), current_data.mean()
    var_ref, var_curr = reference_data.var(), current_data.var()

    return {
        "Mean Shift": abs(mean_ref - mean_curr),
        "Variance Shift": abs(var_ref - var_curr)
    }

# 2. Cumulative Distribution Shift
def cumulative_distribution_shift(reference_data, current_data):
    """
    Compares the cumulative distributions of reference and current data using Wasserstein distance.
    
    Parameters:
        reference_data (pd.Series): Historical dataset for comparison.
        current_data (pd.Series): Current dataset to be analyzed for drift.
    
    Returns:
        float: Wasserstein distance between the cumulative distributions.
    """
    return wasserstein_distance(np.sort(reference_data), np.sort(current_data))

# 3. Kullback-Leibler Divergence
def kullback_leibler_divergence(reference_data, current_data):
    """
    Computes the Kullback-Leibler Divergence between two distributions.
    
    Parameters:
        reference_data (pd.Series): Historical dataset for comparison.
        current_data (pd.Series): Current dataset to be analyzed for drift.
    
    Returns:
        float: KL divergence between the two distributions.
    """
    ref_prob = reference_data.value_counts(normalize=True)
    curr_prob = current_data.value_counts(normalize=True)

    # Align the two distributions
    all_categories = ref_prob.index.union(curr_prob.index)
    ref_prob = ref_prob.reindex(all_categories, fill_value=0)
    curr_prob = curr_prob.reindex(all_categories, fill_value=0)

    return np.sum(ref_prob * np.log(ref_prob / curr_prob))

# 4. Covariate Shift Detection
def covariate_shift(reference_data, current_data):
    """
    Compares the conditional distributions of features with respect to a target variable.
    
    Parameters:
        reference_data (pd.DataFrame): Historical dataset for comparison.
        current_data (pd.DataFrame): Current dataset to be analyzed for drift.
    
    Returns:
        float: The Wasserstein distance for each feature conditioned on the target variable.
    """
    # Assuming we have a target variable
    target_column = "target"  # You should modify this to your actual target column
    ref_features = reference_data.drop(columns=[target_column])
    curr_features = current_data.drop(columns=[target_column])

    drift_scores = {}
    for column in ref_features.columns:
        drift_scores[column] = wasserstein_distance(
            np.sort(ref_features[column]), np.sort(curr_features[column])
        )
    
    return drift_scores

# 5. Feature Correlation Drift
def feature_correlation_drift(reference_data, current_data):
    """
    Detects changes in feature correlations between the reference and current data.
    
    Parameters:
        reference_data (pd.DataFrame): Historical dataset for comparison.
        current_data (pd.DataFrame): Current dataset to be analyzed for drift.
    
    Returns:
        dict: A dictionary containing correlation shifts between feature pairs.
    """
    ref_corr = reference_data.corr()
    curr_corr = current_data.corr()

    drift_scores = {}
    for column in ref_corr.columns:
        for other_column in ref_corr.columns:
            if column != other_column:
                ref_corr_value = ref_corr.at[column, other_column]
                curr_corr_value = curr_corr.at[column, other_column]
                drift_scores[f"{column}-{other_column}"] = abs(ref_corr_value - curr_corr_value)
    
    return drift_scores

# 6. Entropy Shift
def entropy_shift(reference_data, current_data):
    """
    Measures the change in entropy between the reference and current data distributions.
    
    Parameters:
        reference_data (pd.Series): Historical dataset for comparison.
        current_data (pd.Series): Current dataset to be analyzed for drift.
    
    Returns:
        float: The shift in entropy.
    """
    ref_entropy = entropy(reference_data.value_counts(normalize=True))
    curr_entropy = entropy(current_data.value_counts(normalize=True))
    return abs(ref_entropy - curr_entropy)

# 7. Outlier Ratio
def outlier_ratio(reference_data, current_data):
    """
    Measures the ratio of outliers in the current data compared to the reference data.
    
    Parameters:
        reference_data (pd.Series): Historical dataset for comparison.
        current_data (pd.Series): Current dataset to be analyzed for drift.
    
    Returns:
        float: The ratio of outliers in the current data relative to the reference data.
    """
    def get_outliers(data):
        z_scores = (data - data.mean()) / data.std()
        return np.sum(np.abs(z_scores) > 3)  # Outliers defined as Z > 3 or Z < -3

    ref_outliers = get_outliers(reference_data)
    curr_outliers = get_outliers(current_data)

    return curr_outliers / ref_outliers if ref_outliers > 0 else float('inf')

# 8. Maximum Mean Discrepancy (MMD)
def maximum_mean_discrepancy(reference_data, current_data):
    """
    Measures the Maximum Mean Discrepancy (MMD) between two datasets using a Gaussian kernel.
    
    Parameters:
        reference_data (pd.Series): Historical dataset for comparison.
        current_data (pd.Series): Current dataset to be analyzed for drift.
    
    Returns:
        float: The MMD score between the reference and current datasets.
    """
    def gaussian_kernel(x, y, bandwidth=1.0):
        return np.exp(-np.linalg.norm(x - y) ** 2 / (2 * bandwidth ** 2))
    
    ref_data = reference_data.values[:, None]
    curr_data = current_data.values[:, None]
    
    ref_kernel = np.mean([gaussian_kernel(x, y) for x in ref_data for y in ref_data])
    curr_kernel = np.mean([gaussian_kernel(x, y) for x in curr_data for y in curr_data])
    cross_kernel = np.mean([gaussian_kernel(x, y) for x in ref_data for y in curr_data])
    
    return ref_kernel + curr_kernel - 2 * cross_kernel

# 9. Proportion of Unique Values (Categorical)
def proportion_unique_values(reference_data, current_data):
    """
    Measures the change in the proportion of unique values in categorical columns.
    
    Parameters:
        reference_data (pd.Series): Historical dataset for comparison.
        current_data (pd.Series): Current dataset to be analyzed for drift.
    
    Returns:
        float: The ratio of change in unique values.
    """
    unique_ref = reference_data.nunique()
    unique_curr = current_data.nunique()
    return abs(unique_ref - unique_curr) / max(unique_ref, unique_curr)

# 10. Classifier-based Drift Detection
def classifier_based_drift(reference_data, current_data):
    """
    Uses a classifier to detect drift by measuring the model's ability to distinguish between reference and current data.
    
    Parameters:
        reference_data (pd.DataFrame): Historical dataset for comparison.
        current_data (pd.DataFrame): Current dataset to be analyzed for drift.
    
    Returns:
        float: The accuracy of the classifier in distinguishing between reference and current data.
    """
    # Label the datasets as reference (0) and current (1)
    ref_labels = np.zeros(len(reference_data))
    curr_labels = np.ones(len(current_data))

    combined_data = pd.concat([reference_data, current_data], axis=0)
    combined_labels = np.concatenate([ref_labels, curr_labels])

    # Train a simple classifier (e.g., Logistic Regression)
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    X_train, X_test, y_train, y_test = train_test_split(combined_data, combined_labels, test_size=0.3)

    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    return accuracy_score(y_test, y_pred)

# 11. Jensen-Shannon Divergence
def jensen_shannon_divergence(reference_data, current_data):
    """
    Computes the Jensen-Shannon Divergence to quantify the difference between two categorical distributions.
    
    Parameters:
        reference_data (pd.Series): Historical dataset for comparison.
        current_data (pd.Series): Current dataset to be analyzed for drift.
    
    Returns:
        float: The Jensen-Shannon divergence between the reference and current data.
    """
    ref_prob = reference_data.value_counts(normalize=True)
    curr_prob = current_data.value_counts(normalize=True)

    # Align both distributions
    all_categories = ref_prob.index.union(curr_prob.index)
    ref_prob = ref_prob.reindex(all_categories, fill_value=0)
    curr_prob = curr_prob.reindex(all_categories, fill_value=0)

    return jensenshannon(ref_prob, curr_prob)
