import os
import shutil
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import base64
from io import BytesIO

def prepare_data(data):
    """
    Prepare data for drift analysis by converting date and extracting monthly means.
    """
    data['Date'] = pd.to_datetime(data['Date'])
    data['Year-Month'] = data['Date'].dt.to_period('M')
    monthly_mean = data.groupby('Year-Month').mean()
    monthly_mean.index = monthly_mean.index.astype(str)
    return monthly_mean

def generate_drift_plot(reference_mean, current_mean, column):
    """
    Generate drift plot for a specific column and return as HTML for interactivity.
    """
    mean_val = reference_mean[column].mean()
    std_val = reference_mean[column].std()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=current_mean.index,
        y=[mean_val + std_val] * len(current_mean.index),
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip',
    ))
    fig.add_trace(go.Scatter(
        x=current_mean.index,
        y=[mean_val - std_val] * len(current_mean.index),
        mode='lines',
        fill='tonexty',
        fillcolor='rgba(0, 255, 0, 0.2)',
        line=dict(width=0),
        name='Mean ± SD Range',
        hoverinfo='skip',
    ))
    fig.add_trace(go.Scatter(
        x=current_mean.index,
        y=current_mean[column],
        mode='lines+markers',
        line=dict(color='red', width=2),
        marker=dict(size=6),
        name=f'Monthly Average {column}',
        hovertemplate='%{y:.2f}<extra></extra>',
    ))
    fig.add_trace(go.Scatter(
        x=current_mean.index,
        y=[mean_val] * len(current_mean.index),
        mode='lines',
        line=dict(color='green', width=1),
        name='Mean',
        hoverinfo='skip',
    ))
    fig.add_trace(go.Scatter(
        x=current_mean.index,
        y=[mean_val + std_val] * len(current_mean.index),
        mode='lines',
        line=dict(color='blue', dash='dash', width=1),
        name='Mean + SD',
        hoverinfo='skip',
    ))
    fig.add_trace(go.Scatter(
        x=current_mean.index,
        y=[mean_val - std_val] * len(current_mean.index),
        mode='lines',
        line=dict(color='blue', dash='dash', width=1),
        name='Mean - SD',
        hoverinfo='skip',
    ))

    fig.update_layout(
        title=f'Monthly Average {column} with SD Bands',
        xaxis_title='Month',
        yaxis_title=f'{column}',
        template='plotly_white',
        legend=dict(title="Legend"),
        xaxis=dict(tickangle=45),
        hovermode='x unified',
    )

    # Generate the Plotly figure as an HTML div (interactive)
    fig_html = pio.to_html(fig, full_html=False)
    return fig_html

def generate_distribution_plot(reference_mean, current_mean, column):
    """
    Generate distribution plot using Plotly and return as HTML for interactivity.
    """
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
        y=current_counts,
        name="Current",
        marker_color='red'
    ))
    fig.add_trace(go.Bar(
        x=bin_midpoints,
        y=reference_counts,
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

    # Generate the Plotly figure as an HTML div (interactive)
    fig_html = pio.to_html(fig, full_html=False)
    return fig_html

def generate_drift_report(reference_data, current_data, drift_results):
    """
    Generate an HTML report based on drift analysis with embedded interactive plots.
    """
    reference_mean_monthly = prepare_data(reference_data)
    current_mean_monthly = prepare_data(current_data)

    drift_detected = []
    no_drift_detected = []

    # Classify columns based on drift detection
    for column in reference_mean_monthly.select_dtypes(include=[np.number]).columns:
        if drift_results.get(column, {}).get('Threshold Breach', False):
            drift_detected.append(column)
        else:
            no_drift_detected.append(column)

    # HTML Report Header
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Data Drift Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; }}
            .container {{ width: 80%; margin: auto; }}
            .column {{ margin-bottom: 40px; }}
            .column h2 {{ border-bottom: 1px solid #ddd; padding-bottom: 10px; }}
            .column img {{ width: 100%; height: auto; }}
            table {{ width: 100%; border-collapse: collapse; margin-bottom: 40px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Data Drift Report</h1>
            <p>Algorithm Used: Statistical Drift Detection</p>
            <h2>Summary of Drift Detection</h2>
            <table>
                <thead>
                    <tr>
                        <th>Drift Detected</th>
                        <th>No Drift Detected</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>{', '.join(drift_detected)}</td>
                        <td>{', '.join(no_drift_detected)}</td>
                    </tr>
                </tbody>
            </table>
    """

    # Generate report for each column
    for column in reference_mean_monthly.select_dtypes(include=[np.number]).columns:
        try:
            drift_plot_html = generate_drift_plot(reference_mean_monthly, current_mean_monthly, column)
            distribution_plot_html = generate_distribution_plot(reference_mean_monthly, current_mean_monthly, column)

            if drift_plot_html and distribution_plot_html:
                test_name = drift_results.get(column, {}).get('Test Name', 'N/A')
                drift_metric = drift_results.get(column, {}).get('Drift Metric', 'N/A')
                threshold_breach = drift_results.get(column, {}).get('Threshold Breach', 'N/A')

                html_content += f"""
                <div class="column">
                    <h2>Column: {column}</h2>
                    <p>Test Name: {test_name}</p>
                    <p>Drift Metric: {drift_metric}</p>
                    <p>Threshold Breach: {threshold_breach}</p>
                    <h3>Drift Plot</h3>
                    {drift_plot_html}
                    <h3>Data Distribution (Current vs. Reference)</h3>
                    {distribution_plot_html}
                </div>
                """
        except Exception as e:
            print(f"Error processing column {column}: {e}")

    html_content += """
        </div>
    </body>
    </html>
    """

    # Save the report as an HTML file
    html_report_path = 'data_drift_report.html'
    with open(html_report_path, 'w') as file:
        file.write(html_content)

    print(f"Report saved as {html_report_path}")

    # Clean up: Delete the 'plots' folder after generating the report
    plots_folder = 'plots'
    if os.path.exists(plots_folder):
        shutil.rmtree(plots_folder)
    return html_report_path