import pandas as pd
from drift_detection import DriftDetection
from plotting import generate_drift_report


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

    # Initialize drift detector and perform drift detection
    drift_detector = DriftDetection(reference_data, current_data)
    drift_results = drift_detector.detect_drift()

    # Display results
    for column, result in drift_results.items():
        print(f"{column}: {result}")

    # Generate drift report
    generate_drift_report(reference_data, current_data, drift_results)

if __name__ == "__main__":
    main()