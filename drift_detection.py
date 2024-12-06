import pandas as pd
from tests import (
    run_anderson_darling,
    run_ks_test,
    run_wasserstein_distance,
    run_js_divergence,
    run_chi_squared_test,
)

class DriftDetection:
    def __init__(self, reference_data, current_data):
      
        self.reference_data = reference_data
        self.current_data = current_data

    def detect_drift(self):
      
        reference_data = self.reference_data
        current_data = self.current_data
        results = {}
        threshold = 0.1
        is_large_data = len(reference_data) > 1000

        for column in reference_data.columns:
            try:
                if pd.api.types.is_numeric_dtype(reference_data[column]):
                    # Numerical columns
                    if reference_data[column].nunique() > 5:
                        if is_large_data:
                            drift_metric, test_name = run_wasserstein_distance(
                                reference_data[column], current_data[column]
                            )
                        else:
                            drift_metric, test_name = run_anderson_darling(
                                reference_data[column], current_data[column]
                            )
                    else:
                        drift_metric, test_name = run_js_divergence(
                            reference_data[column], current_data[column]
                        )
                else:
                    # Categorical columns
                    if reference_data[column].nunique() <= 2:
                        drift_metric, test_name = run_js_divergence(
                            reference_data[column], current_data[column]
                        )
                    else:
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