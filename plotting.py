import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import plotly.graph_objects as go

def prepare_data(data):
   
    data['Date'] = pd.to_datetime(data['Date'])
    data['Year-Month'] = data['Date'].dt.to_period('M')
    monthly_mean = data.groupby('Year-Month').mean()
    monthly_mean.index = monthly_mean.index.astype(str)
    return monthly_mean

def generate_drift_plot(reference_mean, current_mean, column, plot_dir):
    
    mean_val = reference_mean[column].mean()
    std_val = reference_mean[column].std()

    plt.figure(figsize=(10, 6))
    plt.fill_between(current_mean.index, mean_val + std_val, mean_val - std_val, 
                     color='green', alpha=0.2, label='Reference Mean ± SD Range')
    plt.plot(current_mean.index, current_mean[column], 
             label=f'Monthly Average {column}', marker='o', color='red')
    plt.axhline(mean_val, color='green', linestyle='-', linewidth=1, label='Mean')
    plt.axhline(mean_val + std_val, color='blue', linestyle='--', linewidth=1, label='Mean + SD')
    plt.axhline(mean_val - std_val, color='blue', linestyle='--', linewidth=1, label='Mean - SD')
    plt.title(f'Monthly Average {column} with SD Bands', fontsize=16)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel(f'{column}', fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.legend()
    plt.tight_layout()

    plot_path = os.path.join(plot_dir, f'{column}_plot.png')
    plt.savefig(plot_path)
    plt.close()

    return plot_path

def generate_distribution_plot(reference_mean, current_mean, column, plot_dir):
 
    if not np.issubdtype(reference_mean[column].dtype, np.number):
        print(f"Skipping non-numeric column: {column}")
        return None

    min_val = min(current_mean[column].min(), reference_mean[column].min())
    max_val = max(current_mean[column].max(), reference_mean[column].max())
    bins = np.linspace(min_val, max_val, 40)

    current_counts, _ = np.histogram(current_mean[column], bins=bins)
    reference_counts, _ = np.histogram(reference_mean[column], bins=bins)

    bin_midpoints = 0.5 * (bins[:-1] + bins[1:])

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=bin_midpoints,
        y=current_counts * 50,
        name="Current",
        marker_color='red'
    ))
    fig.add_trace(go.Bar(
        x=bin_midpoints,
        y=reference_counts * 50,
        name="Reference",
        marker_color='gray'
    ))

    fig.update_layout(
        title=f"Data Distribution for {column}: Current vs. Reference",
        xaxis_title=f"{column} Range",
        yaxis_title="Count",
        barmode="group",
        template="plotly_dark",
        legend=dict(title="Group")
    )

    fig_path = os.path.join(plot_dir, f"{column}_data_distribution.html")
    fig.write_html(fig_path)

    return fig_path

def generate_drift_report(reference_data, current_data, drift_results):
    
    plot_dir = 'plots'
    os.makedirs(plot_dir, exist_ok=True)

    reference_mean_monthly = prepare_data(reference_data)
    current_mean_monthly = prepare_data(current_data)

    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Data Drift Report</title>
        <style>
            body { font-family: Arial, sans-serif; }
            .container { width: 80%; margin: auto; }
            .column { margin-bottom: 40px; }
            .column h2 { border-bottom: 1px solid #ddd; padding-bottom: 10px; }
            .column img { width: 100%; height: auto; }
            .column iframe { width: 100%; height: 400px; border: none; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Data Drift Report</h1>
            <p>Algorithm Used: Statistical Drift Detection</p>
    """

    for column in reference_mean_monthly.select_dtypes(include=[np.number]).columns:
        try:
            plot_path = generate_drift_plot(reference_mean_monthly, current_mean_monthly, column, plot_dir)
            distribution_path = generate_distribution_plot(reference_mean_monthly, current_mean_monthly, column, plot_dir)

            if plot_path and distribution_path:
                test_name = drift_results.get(column, {}).get('Test Name', 'N/A')
                drift_metric = drift_results.get(column, {}).get('Drift Metric', 'N/A')
                threshold_breach = drift_results.get(column, {}).get('Threshold Breach', 'N/A')

                html_content += f"""
                <div class="column">
                    <h2>Column: {column}</h2>
                    <p>Test Name: {test_name}</p>
                    <p>Drift Metric: {drift_metric}</p>
                    <p>Threshold Breach: {threshold_breach}</p>
                    <img src="{plot_path}" alt="Data Drift in {column}">
                    <h3>Data Distribution (Current vs. Reference)</h3>
                    <iframe src="{distribution_path}"></iframe>
                </div>
                """
        except Exception as e:
            print(f"Error processing column {column}: {e}")

    html_content += """
        </div>
    </body>
    </html>
    """

    html_report_path = 'data_drift_report.html'
    with open(html_report_path, 'w') as file:
        file.write(html_content)

    print(f"Report saved as {html_report_path}")