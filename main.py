import pandas as pd
from drift_detection import DriftDetection
from plotting import generate_drift_report

def main():
    reference_data = pd.read_csv("cleaned_reference_data.csv")
    current_data = pd.read_csv("cleaned_current_data.csv")
   
    drift_detector = DriftDetection(reference_data, current_data)
   
    drift_results = drift_detector.detect_drift()

    for column, result in drift_results.items():
        print(f"{column}: {result}")

    generate_drift_report(reference_data, current_data, drift_results)
    
if __name__ == "__main__":
    main()