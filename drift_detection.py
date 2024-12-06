import pandas as pd
from tests import (
    run_anderson_darling,
    run_ks_test,
    run_wasserstein_distance,
    run_js_divergence,
    run_chi_squared_test,
)

class DriftDetection:
    """
    Class for detecting drift between reference data and current data
    using statistical tests based on data type and size.
    """
    def __init__(self, reference_data, current_data):
        """
        Initialize the DriftDetection class with reference and current datasets.
        :param reference_data: DataFrame representing historical data
        :param current_data: DataFrame representing the latest data to compare
        """
        self.reference_data = reference_data
        self.current_data = current_data

    def detect_drift(self):
        """
        Perform drift detection by applying statistical tests to each column
        based on its data type and properties.
        :return: A dictionary containing drift metrics and test results for each column.
        """
        reference_data = self.reference_data
        current_data = self.current_data
        results = {}
        threshold = 0.1
        is_large_data = len(reference_data) > 1000

        # Iterate over each column in the reference dataset
        for column in reference_data.columns:
            try:
                # Check if the column is numeric
                if pd.api.types.is_numeric_dtype(reference_data[column]):
                    # Numerical columns
                    if reference_data[column].nunique() > 5:
                        if is_large_data:
                            # Use Wasserstein distance for large datasets
                            drift_metric, test_name = run_wasserstein_distance(
                                reference_data[column], current_data[column]
                            )
                        else:
                            # Use Anderson-Darling test for small datasets
                            drift_metric, test_name = run_anderson_darling(
                                reference_data[column], current_data[column]
                            )
                    else:
                        # Use Jensen-Shannon divergence for low cardinality numeric data
                        drift_metric, test_name = run_js_divergence(
                            reference_data[column], current_data[column]
                        )
                else:
                    # Categorical columns
                    if reference_data[column].nunique() <= 2:
                        # Use Jensen-Shannon divergence for binary categories
                        drift_metric, test_name = run_js_divergence(
                            reference_data[column], current_data[column]
                        )
                    else:
                        # Use Chi-squared test for categorical data with more than two categories
                        drift_metric, test_name = run_chi_squared_test(
                            reference_data[column], current_data[column]
                        )

                # Store results
                results[column] = {
                    "Test Name": test_name,
                    "Drift Metric": drift_metric,
                    "Threshold Breach": drift_metric > threshold if is_large_data else None,
                }

            except Exception as e:
                results[column] = {"Test Name": "Failed", "Error": str(e)}

        return results