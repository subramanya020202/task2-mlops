# import os
# import shutil
# import pandas as pd
# import numpy as np
# import plotly.graph_objects as go
# import plotly.io as pio
# import base64
# from io import BytesIO

# def prepare_data(data):
#     """
#     Prepare data for drift analysis by converting date and extracting monthly means.
#     """
#     data['Date'] = pd.to_datetime(data['Date'])
#     data['Year-Month'] = data['Date'].dt.to_period('M')
#     monthly_mean = data.groupby('Year-Month').mean()
#     monthly_mean.index = monthly_mean.index.astype(str)
#     return monthly_mean

# def generate_drift_plot(reference_mean, current_mean, column):
#     """
#     Generate drift plot for a specific column and return as HTML for interactivity.
#     """
#     mean_val = reference_mean[column].mean()
#     std_val = reference_mean[column].std()

#     fig = go.Figure()

#     fig.add_trace(go.Scatter(
#         x=current_mean.index,
#         y=[mean_val + std_val] * len(current_mean.index),
#         mode='lines',
#         line=dict(width=0),
#         showlegend=False,
#         hoverinfo='skip',
#     ))
#     fig.add_trace(go.Scatter(
#         x=current_mean.index,
#         y=[mean_val - std_val] * len(current_mean.index),
#         mode='lines',
#         fill='tonexty',
#         fillcolor='rgba(0, 255, 0, 0.2)',
#         line=dict(width=0),
#         name='Mean ± SD Range',
#         hoverinfo='skip',
#     ))
#     fig.add_trace(go.Scatter(
#         x=current_mean.index,
#         y=current_mean[column],
#         mode='lines+markers',
#         line=dict(color='red', width=2),
#         marker=dict(size=6),
#         name=f'Monthly Average {column}',
#         hovertemplate='%{y:.2f}<extra></extra>',
#     ))
#     fig.add_trace(go.Scatter(
#         x=current_mean.index,
#         y=[mean_val] * len(current_mean.index),
#         mode='lines',
#         line=dict(color='green', width=1),
#         name='Mean',
#         hoverinfo='skip',
#     ))
#     fig.add_trace(go.Scatter(
#         x=current_mean.index,
#         y=[mean_val + std_val] * len(current_mean.index),
#         mode='lines',
#         line=dict(color='blue', dash='dash', width=1),
#         name='Mean + SD',
#         hoverinfo='skip',
#     ))
#     fig.add_trace(go.Scatter(
#         x=current_mean.index,
#         y=[mean_val - std_val] * len(current_mean.index),
#         mode='lines',
#         line=dict(color='blue', dash='dash', width=1),
#         name='Mean - SD',
#         hoverinfo='skip',
#     ))

#     fig.update_layout(
#         title=f'Monthly Average {column} with SD Bands',
#         xaxis_title='Month',
#         yaxis_title=f'{column}',
#         template='plotly_white',
#         legend=dict(title="Legend"),
#         xaxis=dict(tickangle=45),
#         hovermode='x unified',
#     )

#     # Generate the Plotly figure as an HTML div (interactive)
#     fig_html = pio.to_html(fig, full_html=False)
#     return fig_html

# def generate_distribution_plot(reference_mean, current_mean, column):
#     """
#     Generate distribution plot using Plotly and return as HTML for interactivity.
#     """
#     if not np.issubdtype(reference_mean[column].dtype, np.number):
#         print(f"Skipping non-numeric column: {column}")
#         return None

#     min_val = min(current_mean[column].min(), reference_mean[column].min())
#     max_val = max(current_mean[column].max(), reference_mean[column].max())
#     bins = np.linspace(min_val, max_val, 40)

#     current_counts, _ = np.histogram(current_mean[column], bins=bins)
#     reference_counts, _ = np.histogram(reference_mean[column], bins=bins)

#     bin_midpoints = 0.5 * (bins[:-1] + bins[1:])

#     fig = go.Figure()
#     fig.add_trace(go.Bar(
#         x=bin_midpoints,
#         y=current_counts,
#         name="Current",
#         marker_color='red'
#     ))
#     fig.add_trace(go.Bar(
#         x=bin_midpoints,
#         y=reference_counts,
#         name="Reference",
#         marker_color='gray'
#     ))

#     fig.update_layout(
#         title=f"Data Distribution for {column}: Current vs. Reference",
#         xaxis_title=f"{column} Range",
#         yaxis_title="Count",
#         barmode="group",
#         template="plotly_dark",
#         legend=dict(title="Group")
#     )

#     # Generate the Plotly figure as an HTML div (interactive)
#     fig_html = pio.to_html(fig, full_html=False)
#     return fig_html

# def generate_drift_report(reference_data, current_data, drift_results):
#     """
#     Generate an HTML report based on drift analysis with embedded interactive plots.
#     """
#     reference_mean_monthly = prepare_data(reference_data)
#     current_mean_monthly = prepare_data(current_data)

#     drift_detected = []
#     no_drift_detected = []

#     # Classify columns based on drift detection
#     for column in reference_mean_monthly.select_dtypes(include=[np.number]).columns:
#         if drift_results.get(column, {}).get('Threshold Breach', False):
#             drift_detected.append(column)
#         else:
#             no_drift_detected.append(column)

#     # HTML Report Header
#     html_content = f"""
#     <!DOCTYPE html>
#     <html>
#     <head>
#         <title>Data Drift Report</title>
#         <style>
#             body {{ font-family: Arial, sans-serif; }}
#             .container {{ width: 80%; margin: auto; }}
#             .column {{ margin-bottom: 40px; }}
#             .column h2 {{ border-bottom: 1px solid #ddd; padding-bottom: 10px; }}
#             .column img {{ width: 100%; height: auto; }}
#             table {{ width: 100%; border-collapse: collapse; margin-bottom: 40px; }}
#             th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
#             th {{ background-color: #f2f2f2; }}
#         </style>
#     </head>
#     <body>
#         <div class="container">
#             <h1>Data Drift Report</h1>
#             <p>Algorithm Used: Statistical Drift Detection</p>
#             <h2>Summary of Drift Detection</h2>
#             <table>
#                 <thead>
#                     <tr>
#                         <th>Drift Detected</th>
#                         <th>No Drift Detected</th>
#                     </tr>
#                 </thead>
#                 <tbody>
#                     <tr>
#                         <td>{', '.join(drift_detected)}</td>
#                         <td>{', '.join(no_drift_detected)}</td>
#                     </tr>
#                 </tbody>
#             </table>
#     """

#     # Generate report for each column
#     for column in reference_mean_monthly.select_dtypes(include=[np.number]).columns:
#         try:
#             drift_plot_html = generate_drift_plot(reference_mean_monthly, current_mean_monthly, column)
#             distribution_plot_html = generate_distribution_plot(reference_mean_monthly, current_mean_monthly, column)

#             if drift_plot_html and distribution_plot_html:
#                 test_name = drift_results.get(column, {}).get('Test Name', 'N/A')
#                 drift_metric = drift_results.get(column, {}).get('Drift Metric', 'N/A')
#                 threshold_breach = drift_results.get(column, {}).get('Threshold Breach', 'N/A')

#                 html_content += f"""
#                 <div class="column">
#                     <h2>Column: {column}</h2>
#                     <p>Test Name: {test_name}</p>
#                     <p>Drift Metric: {drift_metric}</p>
#                     <p>Threshold Breach: {threshold_breach}</p>
#                     <h3>Drift Plot</h3>
#                     {drift_plot_html}
#                     <h3>Data Distribution (Current vs. Reference)</h3>
#                     {distribution_plot_html}
#                 </div>
#                 """
#         except Exception as e:
#             print(f"Error processing column {column}: {e}")

#     html_content += """
#         </div>
#     </body>
#     </html>
#     """

#     # Save the report as an HTML file
#     html_report_path = 'data_drift_report.html'
#     with open(html_report_path, 'w') as file:
#         file.write(html_content)

#     print(f"Report saved as {html_report_path}")

#     # Clean up: Delete the 'plots' folder after generating the report
#     plots_folder = 'plots'
#     if os.path.exists(plots_folder):
#         shutil.rmtree(plots_folder)
#     return html_report_path


import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio


def prepare_data(data):
    """
    Dynamically prepare data for drift analysis:
    - Detects a column with date-like data if it exists.
    - Converts it to datetime, extracts monthly means, and groups by 'Year-Month'.
    - If no date-like column exists, groups by the existing index and calculates means.
    """
    date_col = None
    for col in data.columns:
        if pd.api.types.is_datetime64_any_dtype(data[col]) or data[col].astype(str).str.match(r'\d{4}-\d{2}-\d{2}').any():
            date_col = col
            break
    
    if date_col:
        # Process the identified date-like column
        data[date_col] = pd.to_datetime(data[date_col], errors='coerce')
        if data[date_col].notna().any():
            data['Year-Month'] = data[date_col].dt.to_period('M')
            monthly_mean = data.groupby('Year-Month').mean()
            monthly_mean.index = monthly_mean.index.astype(str)
            return monthly_mean
    
    # If no date-like column, group by existing index and calculate means
    if data.index.name is None:
        data.index.name = 'Index'  # Temporarily name the index if unnamed
    grouped_data = data.groupby(data.index).mean()
    return grouped_data




def generate_drift_plot(reference_mean, current_mean, column):
    """
    Generate an drift plot with enhanced aesthetics and automatic handling of dates or index values.
    If no dates are present in the index, fallback to integer indexing with spaced tick values.
    """
    import numpy as np

    # Check if index values are datetime-like; fallback to integer indexing if not
    if pd.api.types.is_datetime64_any_dtype(current_mean.index):
        # Handle datetime indices safely
        x_values = current_mean.index
        is_datetime = True
    else:
        # Fallback for non-date index, use integer positions
        x_values = current_mean.index
        is_datetime = False

    # Calculate mean and standard deviation from reference_mean
    mean_val = reference_mean[column].mean()
    std_val = reference_mean[column].std()

    # Filter current_mean dynamically based on index or date range
    start_date = x_values.min()
    current_mean_filtered = current_mean[current_mean.index >= start_date]

    # Elegant color palette
    colors = {
        'background': 'rgba(240, 248, 255, 0.6)',  # Light blue background
        'current_line': 'rgb(59, 130, 246)',      # Vibrant blue
        'mean_line': 'rgb(34, 197, 94)',          # Soft green
        'sd_lines': 'rgb(168, 85, 247)',          # Soft purple
        'sd_shading': 'rgba(173, 216, 230, 0.3)'  # Soft light blue shading
    }

    fig = go.Figure()

    # Shading for Standard deviation range
    fig.add_trace(go.Scatter(
        x=x_values.tolist() + x_values.tolist()[::-1],
        y=[mean_val + std_val] * len(x_values) + [mean_val - std_val] * len(x_values),
        fill='toself',
        fillcolor=colors['sd_shading'],
        line=dict(width=0),
        name='± Standard Deviation Range',
        hoverinfo='skip'
    ))

    # Current monthly average line
    fig.add_trace(go.Scatter(
        x=x_values,
        y=current_mean_filtered[column],
        mode='lines+markers',
        line=dict(color=colors['current_line'], width=3, shape='spline', smoothing=1.3),
        marker=dict(size=8, symbol='diamond', color=colors['current_line'], line=dict(width=2, color='white')),
        name=f'Monthly Average {column}',
        hovertemplate='Index: %{x}<br>Value: %{y:.2f}<extra></extra>'
    ))

    # Mean line
    fig.add_trace(go.Scatter(
        x=x_values,
        y=[mean_val] * len(x_values),
        mode='lines',
        line=dict(color=colors['mean_line'], width=2, dash='dot'),
        name='Reference Mean',
        hoverinfo='skip'
    ))

    # Standard deviation lines (without shaded area)
    fig.add_trace(go.Scatter(
        x=x_values,
        y=[mean_val + std_val] * len(x_values),
        mode='lines',
        line=dict(color=colors['sd_lines'], width=1.5, dash='dash'),
        name='Mean + SD',
        hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=x_values,
        y=[mean_val - std_val] * len(x_values),
        mode='lines',
        line=dict(color=colors['sd_lines'], width=1.5, dash='dash'),
        name='Mean - SD',
        hoverinfo='skip'
    ))

    # Dynamic x-axis ticks
    if is_datetime:
        tickvals = x_values  # Use all datetime ticks
        ticktext = [str(x) for x in x_values]
    else:
        step = max(1, len(x_values) // 10)  # Dynamically determine step size for ticks
        tickvals = x_values[::step]
        ticktext = [str(x) for x in tickvals]

    # Layout with modern design
    fig.update_layout(
        title={
            'text': f'Drift Analysis: {column} Trend',
            'font': {'size': 20, 'color': 'rgba(0,0,0,0.7)'}
        },
        xaxis_title='Index',
        yaxis_title=column,
        template='plotly_white',
        plot_bgcolor=colors['background'],
        hovermode='x unified',
        xaxis=dict(
            tickmode='array',
            tickvals=tickvals,
            ticktext=ticktext,
            tickangle=45
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='rgba(0,0,0,0.1)',
            borderwidth=1
        ),
        margin=dict(l=50, r=50, t=80, b=50)
    )

    # Ensure Plotly JS is included in the HTML report
    return pio.to_html(fig, full_html=False, include_plotlyjs='cdn')



def generate_distribution_plot(reference_mean, current_mean, column):
    """
    Generate distribution plot with modern aesthetics.
    """
    if not np.issubdtype(reference_mean[column].dtype, np.number):
        return None

    # Elegant color palette
    colors = {
        'current': 'rgb(59, 130, 246)',   # Vibrant blue
        'reference': 'rgb(168, 85, 247)', # Soft purple
        'background': 'rgba(240, 248, 255, 0.6)'  # Light blue background
    }

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
        name="Current Data",
        marker_color=colors['current'],
        opacity=0.7
    ))
    fig.add_trace(go.Bar(
        x=bin_midpoints,
        y=reference_counts,
        name="Reference Data",
        marker_color=colors['reference'],
        opacity=0.7
    ))

    fig.update_layout(
        title={
            'text': f'Distribution Analysis: {column}',
            'font': {'size': 20, 'color': 'rgba(0,0,0,0.7)'}
        },
        xaxis_title=f'{column} Range',
        yaxis_title='Frequency',
        template='plotly_white',
        plot_bgcolor=colors['background'],
        barmode='overlay',
        legend=dict(
            orientation="h", 
            yanchor="bottom", 
            y=1.02, 
            xanchor="right", 
            x=1,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='rgba(0,0,0,0.1)',
            borderwidth=1
        ),
        margin=dict(l=50, r=50, t=80, b=50)
    )

    return pio.to_html(fig, full_html=False, include_plotlyjs='cdn')

def generate_drift_report(reference_data, current_data, drift_results):
    """
    Generate  HTML report with interactive visualizations.
    """
    reference_mean_monthly = prepare_data(reference_data)
    current_mean_monthly = prepare_data(current_data)

    # Classify columns based on drift detection
    drift_detected = []
    no_drift_detected = []

    # Classify columns based on threshold breach
    for column in reference_mean_monthly.select_dtypes(include=[np.number]).columns:
        if drift_results.get(column, {}).get('Threshold Breach', False):
            drift_detected.append(column)
        else:
            no_drift_detected.append(column)

    # HTML Report Header with summary table
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Data Drift Report</title>
        <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
        <style>
            body {{ font-family: 'Inter', sans-serif; }}
            .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
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
            <h1 class="text-3xl font-bold text-center my-6 text-gray-800">Data Drift Report</h1>
            <p class="text-xl font-semibold mb-4">Algorithm Used: Statistical Drift Detection</p>
            <h2 class="text-2xl font-semibold mb-4">Summary of Drift Detection</h2>
            <table>
                <thead>
                    <tr>
                        <th class="px-4 py-2">Data Drift Detected</th>
                        <th class="px-4 py-2">No Data Drift Detected</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td class="px-4 py-2">{', '.join(drift_detected)}</td>
                        <td class="px-4 py-2">{', '.join(no_drift_detected)}</td>
                    </tr>
                </tbody>
            </table>
    """

    # Process each numeric column
    for column in reference_mean_monthly.select_dtypes(include=[np.number]).columns:
        try:
            drift_plot_html = generate_drift_plot(reference_mean_monthly, current_mean_monthly, column)
            distribution_plot_html = generate_distribution_plot(reference_mean_monthly, current_mean_monthly, column)

            if drift_plot_html and distribution_plot_html:
                drift_info = drift_results.get(column, {})
                
                html_content += f"""
                <div class="column">
                    <h2 class="text-xl font-semibold text-gray-700">Column: {column}</h2>
                    <div class="grid grid-cols-3 gap-4 mt-2 text-sm">
                        <div>
                            <strong>Test:</strong> {drift_info.get('Test Name', 'N/A')}
                        </div>
                        <div>
                            <strong>Drift Metric:</strong> {drift_info.get('Drift Metric', 'N/A')}
                        </div>
                        <div>
                            <strong>Threshold Breach:</strong> {drift_info.get('Threshold Breach', 'N/A')}
                        </div>
                    </div>
                    <div class="p-4">
                        <h3 class="font-semibold">Drift Plot</h3>
                        {drift_plot_html}
                        <h3 class="font-semibold">Data Distribution (Current vs. Reference)</h3>
                        {distribution_plot_html}
                    </div>
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
    return html_report_path