import pandas as pd
from drift_detection import DriftDetection
from plotting import generate_drift_report
import custom_metrics  # Import the custom metrics


def main():
    # Load datasets
    reference_data = pd.read_csv("cleaned_reference_data.csv")
    current_data = pd.read_csv("cleaned_current_data.csv")

    # Prompt user to exclude specific columns
    print("Available columns:", list(reference_data.columns))
    excluded_columns = input(
        "Enter the columns to exclude from drift detection, separated by commas (or press Enter to skip): "
    ).split(",")
    excluded_columns = [col.strip() for col in excluded_columns if col.strip()]

    # Filter datasets to exclude specified columns
    reference_data = reference_data.drop(columns=excluded_columns, errors="ignore")
    current_data = current_data.drop(columns=excluded_columns, errors="ignore")

    # Perform drift detection with the default detection function
    custom1 = DriftDetection(reference_data, current_data)

    drift_results = custom1.detect_drift()

    # Display default drift detection results
    print("\nDrift Detection Results (default methods):")
    for column, result in drift_results.items():
        print(f"{column}: {result}")

    # Now, calculate additional custom metrics for drift detection
    print("\nCustom Metrics Results:")
    for column in reference_data.columns:
        ref_data = reference_data[column]
        curr_data = current_data[column]

        # Call custom metrics for each column
        try:
            # Example custom metrics you might want to calculate
            mean_var_shift = custom_metrics.mean_variance_shift(ref_data, curr_data)
            print(f"Mean and Variance Shift for {column}: {mean_var_shift}")

            cum_dist_shift = custom_metrics.cumulative_distribution_shift(ref_data, curr_data)
            print(f"Cumulative Distribution Shift for {column}: {cum_dist_shift}")

            kl_divergence = custom_metrics.kullback_leibler_divergence(ref_data, curr_data)
            print(f"Kullback-Leibler Divergence for {column}: {kl_divergence}")

            # Add other custom metrics as needed
            entropy_diff = custom_metrics.entropy_shift(ref_data, curr_data)
            print(f"Entropy Shift for {column}: {entropy_diff}")
            
            outlier_ratio = custom_metrics.outlier_ratio(ref_data, curr_data)
            print(f"Outlier Ratio for {column}: {outlier_ratio}")

        except Exception as e:
            print(f"Could not compute custom metrics for {column} due to: {e}")

    # Generate drift report
    generate_drift_report(reference_data, current_data, drift_results)

if __name__ == "__main__":
    main()
