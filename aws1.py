import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import sqlite3
import pandas as pd
import os
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from functools import lru_cache
import time
import json
import math
import boto3
import tempfile
import os.path
from botocore import UNSIGNED
from botocore.client import Config

# Constants
AVAILABLE_METRICS = ['std_dev', 'rms', 'iqr', 'clean_max', 'clean_min', 'clean_range', 
                    'outlier_count', 'skewness', 'simpson', 'trapz', 'std_error']
SENSORS = ['s1', 's2', 's3', 's4', 's5', 's6']
BINS = np.arange(0, 18, 0.5)
CHANNELS = ['ch1', 'ch2', 'ch3']
COLORS = {'ch1': 'blue', 'ch2': 'red', 'ch3': 'green'}

# Define comparison periods
# Update the comparison periods
COMPARISON_PERIODS = {
    'Period A': ('2023-08-01', '2023-09-15'),
    'Period B': ('2024-06-01', '2024-07-15'),
    'Period C': ('2024-08-01', '2024-09-15'),
    'Period D': ('2024-09-16', '2024-10-31'),
    'Period E': ('2024-11-15', '2024-12-31')
}

# Available transformations
TRANSFORMATIONS = [
    {'label': 'None', 'value': 'none'},
    {'label': 'Log', 'value': 'log'},
    {'label': 'Z-Score', 'value': 'z_score'},
    {'label': 'Min-Max (0-1)', 'value': 'min_max'},
    {'label': 'Robust (IQR)', 'value': 'robust'},
    {'label': 'Decimal Scale', 'value': 'decimal'}
]

# S3 configuration
S3_BUCKET = "public-tarucca-db"
S3_KEY = "meta_data_may2025.db"

# Add this to your constants section at the top
LOG_EVENTS = {
    '2024-11-24': 'Iver Blade Service stops turbine for visual inspection of M02 turbine',
    '2024-11-13': 'Iver Blade Service repairs most urgent damage of M02 turbine',
    '2024-11-14': 'Iver Blade Service repairs most urgent damage of M02 turbine',
    '2024-11-21': 'Topwind stops M02 turbine due to ice curtailment, 8pm Topwind monitors M02 turbine due to ice ',
    '2024-11-22': 'Topwind monitors M02 turbine due to ice ',
    '2025-01-7': 'IVER spotted at M02 for maintenance ',
    # Add more log events as needed
}

# Function to get database file from S3
def get_db_file():
    """Download the database file from S3 to a temporary file"""
    try:
        # Create a temporary file that will be deleted when closed
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_file_path = temp_file.name
        temp_file.close()
        
        print(f"Downloading database from S3: {S3_BUCKET}/{S3_KEY}")
        
        # Create an S3 client with unsigned config for public bucket access
        s3_client = boto3.client(
            's3',
            region_name='eu-central-1',
            config=Config(signature_version=UNSIGNED)
        )
        
        # Download the file
        s3_client.download_file(S3_BUCKET, S3_KEY, temp_file_path)
        print(f"Database downloaded successfully to temporary file")
        return temp_file_path
        
    except Exception as e:
        print(f"Error downloading database from S3: {str(e)}")
        raise

# First, modify the caching decorator to include more parameters
@lru_cache(maxsize=32)  # Increased cache size
def load_data_cached(metric, start_date, end_date):
    """Load data with better caching strategy"""
    try:
        start_time = time.time()
        print(f"Starting data load for {metric} from {start_date} to {end_date}...")
        
        # Use global variable to store db_path
        global db_path
        if not hasattr(load_data_cached, 'db_path'):
            db_path = get_db_file()
            load_data_cached.db_path = db_path
        
        # Create connection and read data efficiently
        with sqlite3.connect(db_path) as conn:
            # Read data with date range filter
            df = pd.read_sql(
                'SELECT id, time FROM main_data WHERE time >= ? AND time <= ?',
                conn,
                params=(start_date, end_date),
                parse_dates=['time']
            )
            
            # Read RPM data only for the filtered IDs
            df_rpm = pd.read_sql(
                'SELECT id, ch1s1 FROM rpm WHERE id IN (SELECT id FROM main_data WHERE time >= ?)',
                conn,
                params=(start_date,)
            )
            
            # Read metric data only for the filtered IDs
            df1 = pd.read_sql(
                f'SELECT * FROM {metric} WHERE id IN (SELECT id FROM main_data WHERE time >= ?)',
                conn,
                params=(start_date,)
            )
            
            # Read corruption status data
            try:
                corruption_df = pd.read_sql(
                    'SELECT * FROM corruption_status',
                    conn
                )
                
                # For each channel-sensor combination in the metric data
                for col in df1.columns:
                    if col != 'id' and any(ch in col for ch in CHANNELS):
                        # If this column exists in corruption_status
                        if col in corruption_df.columns:
                            # Get IDs where this channel-sensor is corrupted
                            corrupted_ids = corruption_df[corruption_df[col] == 1]['id'].tolist()
                            
                            # Set the corresponding values in the metric data to NaN
                            df1.loc[df1['id'].isin(corrupted_ids), col] = np.nan
                            
                            print(f"Set {len(corrupted_ids)} corrupted values to NaN for {col}")
                
            except Exception as e:
                print(f"Warning: Could not apply corruption filtering: {str(e)}")
            
            # Process JSON columns and convert to float where possible
            for col in df1.columns:
                if col != 'id' and any(ch in col for ch in CHANNELS):
                    try:
                        # Try direct float conversion first
                        df1[col] = pd.to_numeric(df1[col], errors='coerce')
                    except ValueError:
                        try:
                            # If that fails, try to extract 'magnitude' from JSON
                            df1[col] = df1[col].apply(lambda x: json.loads(x)['magnitude'] 
                                                    if isinstance(x, str) and x.startswith('{') 
                                                    else x)
                            df1[col] = pd.to_numeric(df1[col], errors='coerce')
                        except:
                            print(f"Warning: Could not process column {col}")
                            df1[col] = np.nan
            
            # Ensure all relevant columns are numeric before further processing
            for col in df1.columns:
                if col != 'id' and any(ch in col for ch in CHANNELS):
                    df1[col] = pd.to_numeric(df1[col], errors='coerce')
            
            # Merge dataframes
            merged_df1 = pd.merge(
                df[['id', 'time']], 
                df1, 
                on='id', 
                how='inner'
            )
            merged_df2 = pd.merge(
                df[['id', 'time']], 
                df_rpm[['id', 'ch1s1']], 
                on='id', 
                how='inner'
            )
            
            # Convert RPM column to float
            merged_df2['ch1s1'] = pd.to_numeric(merged_df2['ch1s1'], errors='coerce')
            
            del df, df1, df_rpm
            
            print(f"Data load completed in {time.time() - start_time:.2f} seconds")
            print(f"Loaded {len(merged_df1)} rows after filtering")
            return merged_df1, merged_df2
            
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise
    
# Calculate default y-limits
def calculate_y_limits(df, channels, sensors):
    all_values = []
    for ch in channels:
        for s in sensors:
            col = f'{ch}{s}'
            if col in df.columns:
                all_values.extend(df[col].dropna().values)
    return np.percentile(all_values, [2.5, 97.5]) if all_values else (0, 1)

# Function to apply data transformations
def apply_transformation(df, transform_type, columns):
    """Apply the selected transformation to the specified columns"""
    transformed_df = df.copy()
    transform_description = "No Transformation"
    
    if transform_type != 'none':
        for col in columns:
            if col in transformed_df.columns:
                # Apply the selected transformation
                if transform_type == 'log':
                    # Add a small constant to avoid log(0)
                    min_val = transformed_df[col].min()
                    offset = 0 if min_val > 0 else abs(min_val) + 1
                    transformed_df[col] = np.log(transformed_df[col] + offset)
                    transform_description = "Log Transformed"
                
                elif transform_type == 'z_score':
                    mean = transformed_df[col].mean()
                    std = transformed_df[col].std()
                    if std > 0:  # Avoid division by zero
                        transformed_df[col] = (transformed_df[col] - mean) / std
                    transform_description = "Z-Score Normalized"
                
                elif transform_type == 'min_max':
                    min_val = transformed_df[col].min()
                    max_val = transformed_df[col].max()
                    range_val = max_val - min_val
                    if range_val > 0:  # Avoid division by zero
                        transformed_df[col] = (transformed_df[col] - min_val) / range_val
                    transform_description = "Min-Max Scaled (0-1)"
                
                elif transform_type == 'robust':
                    median = transformed_df[col].median()
                    q1 = transformed_df[col].quantile(0.25)
                    q3 = transformed_df[col].quantile(0.75)
                    iqr = q3 - q1
                    if iqr > 0:  # Avoid division by zero
                        transformed_df[col] = (transformed_df[col] - median) / iqr
                    transform_description = "Robust Scaled (IQR)"
                
                elif transform_type == 'decimal':
                    max_abs = max(abs(transformed_df[col].max()), abs(transformed_df[col].min()))
                    if max_abs > 0:  # Avoid division by zero
                        scale = 10 ** math.floor(math.log10(max_abs))
                        transformed_df[col] = transformed_df[col] / scale
                    transform_description = "Decimal Scaled"
    
    return transformed_df, transform_description
def apply_sigma_filter_and_smooth(df, column, sigma_threshold=2.0, window_size=7, poly_order=3):
    """
    Apply a sigma filter to remove outliers and then smooth the data with LOWESS or Savitzky-Golay
    
    Parameters:
    -----------
    df: DataFrame containing the data
    column: Column name to process
    sigma_threshold: Number of standard deviations to use for outlier detection
    window_size: Window size for smoothing
    poly_order: Polynomial order for Savitzky-Golay filter
    
    Returns:
    --------
    Tuple of (filtered_series, smoothed_series)
    """
    # Make a copy of the series
    series = df[column].copy()
    
    # Apply sigma filter to remove outliers
    mean = series.mean()
    std = series.std()
    lower_bound = mean - sigma_threshold * std
    upper_bound = mean + sigma_threshold * std
    
    # Create a mask for values within bounds
    mask = (series >= lower_bound) & (series <= upper_bound)
    filtered_series = series.copy()
    filtered_series[~mask] = np.nan  # Set outliers to NaN
    
    # Interpolate missing values
    filtered_series = filtered_series.interpolate(method='linear')
    
    # Apply Savitzky-Golay filter for smoothing
    try:
        from scipy.signal import savgol_filter
        smoothed_values = savgol_filter(filtered_series, window_size, poly_order)
        smoothed_series = pd.Series(smoothed_values, index=filtered_series.index)
    except:
        # Fallback to simple moving average if scipy is not available
        smoothed_series = filtered_series.rolling(window=window_size, center=True, min_periods=1).mean()
    
    return filtered_series, smoothed_series

def apply_baseline_adjustment(df, time_col='time'):
    """Apply baseline adjustment between consecutive periods"""
    adjusted_df = df.copy()
    adjustments = {}
    
    try:
        # Sort periods chronologically
        sorted_periods = sorted(COMPARISON_PERIODS.items(), 
                               key=lambda x: pd.to_datetime(x[1][0]))
        
        # Process each pair of consecutive periods
        for i in range(len(sorted_periods) - 1):
            current_period_name, current_period = sorted_periods[i]
            next_period_name, next_period = sorted_periods[i+1]
            
            # Convert string dates to datetime objects
            current_period_end = pd.to_datetime(current_period[1])
            next_period_start = pd.to_datetime(next_period[0])
            
            print(f"Adjusting between {current_period_name} (ending {current_period[1]}) and {next_period_name} (starting {next_period[0]})")
            
            # Define date ranges for comparison (5 days before/after boundary)
            current_end_range_start = current_period_end - pd.Timedelta(days=5)
            next_start_range_end = next_period_start + pd.Timedelta(days=5)
            
            # Calculate adjustment factors for each column
            for col in adjusted_df.columns:
                # Skip time, id columns, and only process columns that contain channel names
                if col not in [time_col, 'id', 'main_data_id'] and any(ch in col for ch in CHANNELS):
                    try:
                        # Ensure column is numeric
                        if not pd.api.types.is_numeric_dtype(adjusted_df[col]):
                            adjusted_df[col] = pd.to_numeric(adjusted_df[col], errors='coerce')
                        
                        # Create masks for the date ranges
                        current_end_mask = (adjusted_df[time_col] >= current_end_range_start) & (adjusted_df[time_col] <= current_period_end)
                        next_start_mask = (adjusted_df[time_col] >= next_period_start) & (adjusted_df[time_col] <= next_start_range_end)
                        
                        # Get data for the ranges
                        current_end_data = adjusted_df.loc[current_end_mask, col].dropna()
                        next_start_data = adjusted_df.loc[next_start_mask, col].dropna()
                        
                        if not current_end_data.empty and not next_start_data.empty:
                            # Calculate average values at the boundary
                            current_end_avg = current_end_data.mean()
                            next_start_avg = next_start_data.mean()
                            
                            # Calculate adjustment (difference between current end and next start)
                            adjustment = current_end_avg - next_start_avg
                            
                            # Only apply significant adjustments (more than 5% of the data range)
                            data_range = adjusted_df[col].max() - adjusted_df[col].min()
                            if data_range > 0 and abs(adjustment) > 0.05 * data_range:
                                adjustments[f"{current_period_name}-{next_period_name}_{col}"] = adjustment
                                
                                # Apply adjustment to all data after current period
                                after_current_mask = adjusted_df[time_col] > current_period_end
                                adjusted_df.loc[after_current_mask, col] = adjusted_df.loc[after_current_mask, col] + adjustment
                                
                                print(f"  Applied adjustment of {adjustment:.4f} to {col} between {current_period_name} and {next_period_name}")
                            else:
                                print(f"  Skipped small adjustment of {adjustment:.4f} for {col} (less than 5% of data range)")
                        else:
                            print(f"  Insufficient data for {col} at boundary between {current_period_name} and {next_period_name}")
                    except Exception as col_error:
                        print(f"Error processing column {col}: {str(col_error)}")
                        continue
    
    except Exception as e:
        print(f"Error in baseline adjustment: {str(e)}")
        return df, {}
        
    return adjusted_df, adjustments

# Initialize Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[
        'https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css'
    ],
    external_scripts=[
        'https://code.jquery.com/jquery-3.5.1.min.js',
        'https://cdn.jsdelivr.net/npm/popper.js@1.16.1/dist/umd/popper.min.js',
        'https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js'
    ],
    suppress_callback_exceptions=True,
    assets_folder='assets'
)
server = app.server  # This line is crucial for Render deployment

# Add meta tags for better mobile responsiveness
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Sensor Data Analysis Dashboard</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        {%favicon%}
        {%css%}
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Load initial data
try:
    print("Loading initial data from local database...")
    initial_df1, initial_df2 = load_data_cached('std_dev', '2024-06-01', '2024-11-30')
    y_min, y_max = calculate_y_limits(initial_df1, CHANNELS, SENSORS)
    # print(f"Initial data loaded: {len(initial_df1)} rows")
except Exception as e:
    print(f"Error loading initial data: {str(e)}")
    initial_df1 = pd.DataFrame(columns=['id', 'time'])
    initial_df2 = pd.DataFrame(columns=['id', 'time', 'ch1s1'])
    y_min, y_max = 0, 1

# App Layout
app.layout = html.Div([
    dcc.Store(id='loaded-data-store'),
    html.H1("Sensor Data Analysis Dashboard"),
    
    # Add dropdowns for user input
    html.Div([
        html.Div([
            html.Label('Select Metric'),
            dcc.Dropdown(
                id='metric-dropdown',
                options=[{'label': metric.replace('_', ' ').title(), 'value': metric} 
                        for metric in AVAILABLE_METRICS],
                value='std_dev',
                clearable=False
            ),
        ], style={'width': '30%', 'padding': '10px'}),
        
        html.Div([
            html.Label('Select Sensor'),
            dcc.Dropdown(
                id='sensor-dropdown',
                options=[{'label': s, 'value': s} for s in SENSORS],
                value='s1',
                clearable=False
            ),
        ], style={'width': '30%', 'padding': '10px'}),
        
        html.Div([
            html.Label('Select Channels'),
            dcc.Dropdown(
                id='channel-dropdown',
                options=[{'label': ch, 'value': ch} for ch in CHANNELS],
                value=['ch1'],
                multi=True,
                clearable=False
            ),
        ], style={'width': '30%', 'padding': '10px'}),
    ], style={'display': 'flex', 'justifyContent': 'space-between'}),
    
    html.Div([
        html.Div([
            html.Label('Select RPM Range'),
            dcc.RangeSlider(
                id='rpm-range-slider',
                min=0,
                max=18,
                step=0.5,
                marks={i: f'{i}' for i in range(0, 19, 3)},
                value=[0, 18],
                tooltip={"placement": "bottom", "always_visible": True}
            ),
        ], style={'width': '45%', 'padding': '10px'}),
        
        html.Div([
            html.Label('Date Range'),
            dcc.DatePickerRange(
                id='date-picker',
                min_date_allowed=datetime(2023, 1, 1),
                max_date_allowed=initial_df1['time'].max().date() if not initial_df1.empty else datetime.now().date(),
                start_date=datetime(2026, 12, 31).date(),
                end_date=initial_df1['time'].max().date() if not initial_df1.empty else datetime.now().date(),
                display_format='YYYY-MM-DD',
                first_day_of_week=1,  # Monday
                persistence=True,
                persistence_type='session'
            ),
        ], style={'width': '45%', 'padding': '10px'}),
    ], style={'display': 'flex', 'justifyContent': 'space-between'}),
    
    html.Div([
        html.Div([
            html.Label('Moving Average (days)'),
            dcc.Slider(
                id='ma-slider',
                min=1,
                max=30,
                value=7,
                marks={i: str(i) for i in [1, 7, 14, 21, 30]},
                step=1
            ),
        ], style={'width': '30%', 'padding': '10px'}),
        
        html.Div([
            html.Label('Data Transformation'),
            dcc.Dropdown(
                id='transform-dropdown',
                options=TRANSFORMATIONS,
                value='none',
                clearable=False
            ),
        ], style={'width': '30%', 'padding': '10px'}),
        
        html.Div([
            html.Label('Baseline Adjustment'),
            dcc.RadioItems(
                id='baseline-radio',
                options=[
                    {'label': 'Raw Data', 'value': 'raw'},
                    {'label': 'Baseline Adjusted', 'value': 'adjusted'}
                ],
                value='raw',
                labelStyle={'display': 'inline-block', 'margin-right': '10px'}
            ),
        ], style={'width': '30%', 'padding': '10px'}),
    ], style={'display': 'flex', 'justifyContent': 'space-between'}),
    
    html.Div([
        html.Label('Y-axis Range'),
        dcc.Input(id='y-min-input', type='number', value=y_min, placeholder='Min Y'),
        dcc.Input(id='y-max-input', type='number', value=y_max, placeholder='Max Y'),
        html.Button('Auto Scale', id='auto-scale-button', n_clicks=0),
    ], style={'padding': '10px'}),
    
    html.Div([
        html.Div([
            html.Label('Custom Period Selection (Max 2 periods)'),
            dcc.DatePickerRange(
                id='period-picker',
                min_date_allowed=datetime(2023, 1, 1),
                max_date_allowed=datetime(2026, 12, 31),
                start_date=None,
                end_date=None,
                display_format='YYYY-MM-DD'
            ),
            html.Button(
                'Add Period', 
                id='add-period-button', 
                n_clicks=0,
                style={'marginLeft': '10px'}
            ),
        ], style={'width': '45%', 'padding': '10px'}),
        
        html.Div([
            html.Label('Selected Periods for Comparison'),
            dcc.Dropdown(
                id='selected-periods-dropdown',
                options=[],
                value=[],
                multi=True,
                placeholder='Select 1-2 periods to compare'
            ),
            html.Button(
                'Clear Periods', 
                id='clear-periods-button', 
                n_clicks=0,
                style={'marginLeft': '10px'}
            ),
            html.Div(
                id='period-selection-warning',
                style={'color': 'red', 'marginTop': '5px', 'fontSize': '12px'}
            ),
        ], style={'width': '45%', 'padding': '10px'}),
    ], style={'display': 'flex', 'justifyContent': 'space-between'}),
    
    dcc.Loading(
        id="loading-spinner",
        type="circle",
        color="#119DFF",
        children=[
            html.Div(
                id="loading-output",
                style={
                    'textAlign': 'center',
                    'padding': '20px',
                    'fontSize': '18px'
                }
            ),
            dcc.Graph(id='sensor-graph')
        ],
        fullscreen=True
    ),
    
    # Add initial loading message
    html.Div(
        id='initial-loading',
        children='Loading data from local database...',
        style={
            'textAlign': 'center',
            'padding': '20px',
            'fontSize': '18px',
            'color': '#666'
        }
    ),
    
    # Add a new graph for period comparison
    html.Div([
        dcc.Graph(id='period-comparison-graph')
    ], style={'padding': '10px', 'marginTop': '20px'}),
    
    # Add a Store component to keep track of custom periods
    dcc.Store(id='custom-periods-store'),
])

# Modify the load_and_store_data callback
@app.callback(
    Output('loaded-data-store', 'data'),
    [Input('metric-dropdown', 'value'),
     Input('date-picker', 'start_date'),
     Input('date-picker', 'end_date')]
)
def load_and_store_data(selected_metric, start_date, end_date):
    """Load data and store it in the browser"""
    if not all([selected_metric, start_date, end_date]):
        raise PreventUpdate
        
    try:
        # Convert dates to datetime
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # Load data
        merged_df1, merged_df2 = load_data_cached(
            selected_metric,
            start_dt.strftime('%Y-%m-%d'),
            end_dt.strftime('%Y-%m-%d')
        )
        
        # Store the transformation state
        return {
            'df1': merged_df1.to_dict('records'),
            'df2': merged_df2.to_dict('records'),
            'columns1': merged_df1.columns.tolist(),
            'columns2': merged_df2.columns.tolist(),
            'last_transform': None,  # Reset transformation state
            'last_baseline': None    # Reset baseline state
        }
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

# Update the update_graph callback to handle transformations properly
@app.callback(
    [Output('sensor-graph', 'figure'),
     Output('loading-output', 'children')],
    [Input('loaded-data-store', 'data'),
     Input('metric-dropdown', 'value'),
     Input('sensor-dropdown', 'value'),
     Input('channel-dropdown', 'value'),
     Input('rpm-range-slider', 'value'),
     Input('date-picker', 'end_date'),
     Input('y-min-input', 'value'),
     Input('y-max-input', 'value'),
     Input('ma-slider', 'value'),
     Input('transform-dropdown', 'value'),
     Input('baseline-radio', 'value')]
)
def update_graph(stored_data, selected_metric, selected_sensor, selected_channels, rpm_range,
                end_date, y_min, y_max, ma_days, transform_type, baseline_type):
    """Update visualization using stored data"""
    if not all([stored_data, selected_metric, selected_sensor, selected_channels]):
        raise PreventUpdate

    try:
        # Reconstruct dataframes from stored data
        merged_df1 = pd.DataFrame.from_records(stored_data['df1'], columns=stored_data['columns1'])
        merged_df2 = pd.DataFrame.from_records(stored_data['df2'], columns=stored_data['columns2'])
        
        # Convert time column back to datetime
        merged_df1['time'] = pd.to_datetime(merged_df1['time'])
        merged_df2['time'] = pd.to_datetime(merged_df2['time'])
        
        # Create a deep copy for transformations
        working_df = merged_df1.copy(deep=True)
        
        # Process each channel independently
        for ch in (selected_channels if isinstance(selected_channels, list) else [selected_channels]):
            col_name = f'{ch}{selected_sensor}'
            if col_name in working_df.columns:
                # Apply baseline adjustment if selected
                if baseline_type == 'adjusted':
                    channel_df = working_df[[col_name, 'time']].copy()
                    channel_df, _ = apply_baseline_adjustment(channel_df)
                    working_df[col_name] = channel_df[col_name]
                
                # Apply transformation if selected
                if transform_type != 'none':
                    channel_df = working_df[[col_name]].copy()
                    transformed_df, _ = apply_transformation(
                        channel_df,
                        transform_type,
                        [col_name]
                    )
                    working_df[col_name] = transformed_df[col_name]

        # Create figure
        fig = go.Figure()
        
        # Add period masks first (before data traces)
        colors = ['rgba(173, 216, 230, 0.2)',  # light blue
                 'rgba(144, 238, 144, 0.2)',  # light green
                 'rgba(255, 182, 193, 0.2)',  # light pink
                 'rgba(255, 218, 185, 0.2)',  # peach
                 'rgba(221, 160, 221, 0.2)']  # plum
                 
        for (period_name, (start_date, end_date)), color in zip(COMPARISON_PERIODS.items(), colors):
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            # Add shaded area for period
            fig.add_shape(
                type="rect",
                x0=start_dt,
                x1=end_dt,
                y0=y_min if y_min is not None else 0,
                y1=y_max if y_max is not None else 1,
                fillcolor=color,
                opacity=0.5,
                layer="below",
                line_width=0,
            )
            
            # Add period label at the top
            fig.add_annotation(
                x=start_dt + (end_dt - start_dt)/2,
                y=1.05,
                text=period_name,
                showarrow=False,
                xref="x",
                yref="paper",
                font=dict(size=10),
                bgcolor="white",
                bordercolor=color.replace('0.2', '1.0'),  # Solid color for border
                borderwidth=1,
                borderpad=2
            )
            
            # Add vertical lines at period boundaries
            fig.add_shape(
                type="line",
                x0=start_dt,
                x1=start_dt,
                y0=0,
                y1=1,
                yref="paper",
                line=dict(color="gray", width=1, dash="dash"),
                opacity=0.5,
                layer="below"
            )
            fig.add_shape(
                type="line",
                x0=end_dt,
                x1=end_dt,
                y0=0,
                y1=1,
                yref="paper",
                line=dict(color="gray", width=1, dash="dash"),
                opacity=0.5,
                layer="below"
            )

        # Add log event markers
        for date, event in LOG_EVENTS.items():
            event_date = pd.to_datetime(date)
            
            # Add vertical dotted line for event
            fig.add_shape(
                type="line",
                x0=event_date,
                x1=event_date,
                y0=0,
                y1=1,
                yref="paper",
                line=dict(
                    color="rgba(255, 0, 0, 0.5)",
                    width=2,
                    dash="dot"
                ),
                layer="below"
            )
            
            # Add invisible hover target that spans the height of the graph
            fig.add_trace(go.Scatter(
                x=[event_date, event_date],
                y=[0, 1],
                mode='lines',
                line=dict(width=0),  # Invisible line
                hoverinfo='text',
                hovertext=f"⚡ {date}<br>{event}",
                showlegend=False,
                yaxis='y'
            ))
            
            # Add small star marker at the top
            fig.add_trace(go.Scatter(
                x=[event_date],
                y=[1.02],
                mode='markers',
                marker=dict(
                    symbol='star',
                    size=8,
                    color='red',
                ),
                hoverinfo='text',
                hovertext=f"⚡ {date}<br>{event}",
                showlegend=False
            ))

        # Add single legend entry for log events
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='lines',
            line=dict(
                color="rgba(255, 0, 0, 0.5)",
                width=2,
                dash="dot"
            ),
            name="Log Events ⚡",
            showlegend=True
        ))

        # Process each channel
        for ch in (selected_channels if isinstance(selected_channels, list) else [selected_channels]):
            col_name = f'{ch}{selected_sensor}'
            if col_name in working_df.columns:
                # Calculate daily means using processed data
                daily_data = working_df.set_index('time')[col_name].resample('D').mean()
                
                # Apply moving average
                ma_data = daily_data.rolling(window=ma_days, min_periods=1).mean()
                
                if not ma_data.empty:
                    # Add the raw moving average line
                    fig.add_trace(go.Scatter(
                        x=ma_data.index,
                        y=ma_data.values,
                        mode='lines+markers',
                        name=f'{ch} ({ma_days}-day MA)',
                        line=dict(color=COLORS[ch], width=1.5),
                        marker=dict(color=COLORS[ch], size=5),
                        opacity=0.7
                    ))
                    
                    # Apply sigma filter and smoothing
                    try:
                        temp_df = pd.DataFrame({'value': ma_data.values}, index=ma_data.index)
                        filtered_series, smoothed_series = apply_sigma_filter_and_smooth(
                            temp_df, 'value', 
                            sigma_threshold=2.0,
                            window_size=15,
                            poly_order=3
                        )
                        
                        # Add filtered and smoothed data traces
                        fig.add_trace(go.Scatter(
                            x=ma_data.index,
                            y=filtered_series.values,
                            mode='markers',
                            name=f'{ch} (Filtered)',
                            marker=dict(color=COLORS[ch], size=4, symbol='circle-open'),
                            opacity=0.5
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=ma_data.index,
                            y=smoothed_series.values,
                            mode='lines',
                            name=f'{ch} (Smoothed Trend)',
                            line=dict(color=COLORS[ch], width=3, dash='solid'),
                            opacity=1.0
                        ))
                    except Exception as e:
                        print(f"Error creating smoothed curve: {str(e)}")

        # Update layout with transformation and baseline information
        transform_text = f"Transform: {transform_type.replace('_', ' ').title()}" if transform_type != 'none' else "No Transform"
        baseline_text = "Baseline Adjusted" if baseline_type == 'adjusted' else "Raw Data"
        title_text = f'{selected_metric.replace("_", " ").title()} - Sensor {selected_sensor}'
        rpm_text = f"RPM Range: {rpm_range[0]}-{rpm_range[1]}"
            
        fig.update_layout(
            title=dict(
                text=title_text,
                y=0.98,  # Moved title even higher (was 0.95)
                x=0.5,
                xanchor='center',
                yanchor='top',
                font=dict(size=20)
            ),
            xaxis_title='Time',
            yaxis_title=f'{selected_metric.replace("_", " ").title()} Value',
            yaxis=dict(
                range=[y_min, y_max],
                gridcolor='rgba(128, 128, 128, 0.2)',  # Light gray grid
                zerolinecolor='rgba(128, 128, 128, 0.5)'  # Darker zero line
            ),
            xaxis=dict(
                gridcolor='rgba(128, 128, 128, 0.2)',
                showgrid=True,
                rangeslider=dict(
                    visible=False,
                    thickness=0.1,  # Make the rangeslider thinner (default is 0.15)
                    bgcolor='rgba(211, 211, 211, 0.5)'  # Lighter background
                ),
                rangeselector=dict(
                    buttons=list([
                        dict(count=7, label="1w", step="day", stepmode="backward"),
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=3, label="3m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(step="all", label="All")
                    ])
                )
            ),
            plot_bgcolor='white',
            height=600,
            legend=dict(
                orientation="v",  # Changed to vertical orientation
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02,  # Move legend outside to the right
                bgcolor="rgba(255, 255, 255, 0.8)",  # Semi-transparent white background
                bordercolor="rgba(128, 128, 128, 0.2)",
                borderwidth=1
            ),
            margin=dict(
                b=80,   # Reduced bottom margin (was 100)
                t=150,  # Increased top margin (was 120)
                l=50,
                r=150
            ),
        )
        
        # Add legend for periods
        for (period_name, _), color in zip(COMPARISON_PERIODS.items(), colors):
            fig.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(size=10, color=color.replace('0.2', '1.0')),
                name=period_name,
                showlegend=True
            ))

        # Move the processing information annotation up slightly
        fig.add_annotation(
            text=f"{transform_text} | {baseline_text} | {rpm_text} | Sigma-filtered (2σ) with Savitzky-Golay smoothing",
            xref="paper", yref="paper",
            x=0.5, y=-0.12,  # Moved up slightly (was -0.15)
            showarrow=False,
            font=dict(size=10),
            bgcolor="rgba(255, 240, 220, 0.8)",
            bordercolor="orange",
            borderwidth=1
        )
        
        return fig, "Data processed successfully"
        
    except Exception as e:
        print(f"Error updating visualization: {str(e)}")
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="Error",
            annotations=[dict(
                text=f"Error: {str(e)}",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5
            )]
        )
        return empty_fig, f"Error: {str(e)}"

@app.callback(
    [Output('y-min-input', 'value'),
     Output('y-max-input', 'value')],
    [Input('auto-scale-button', 'n_clicks'),
     Input('loaded-data-store', 'data')],
    [State('sensor-dropdown', 'value'),
     State('channel-dropdown', 'value'),
     State('transform-dropdown', 'value'),
     State('baseline-radio', 'value')]
)
def update_y_axis_range(n_clicks, stored_data, sensor, channels, transform_type, baseline_type):
    if n_clicks == 0 or not stored_data:
        raise PreventUpdate
    
    try:
        # Reconstruct dataframe from stored data
        df1 = pd.DataFrame.from_records(stored_data['df1'], columns=stored_data['columns1'])
        
        # Create working copy
        working_df = df1.copy()
        
        # Apply baseline adjustment if selected
        if baseline_type == 'adjusted':
            working_df, _ = apply_baseline_adjustment(working_df)
        
        # Apply transformation if selected
        if transform_type != 'none':
            # Get columns to transform
            columns_to_transform = [f'{ch}{sensor}' for ch in channels]
            working_df, _ = apply_transformation(working_df, transform_type, columns_to_transform)
        
        # Calculate y limits using transformed data
        y_min, y_max = calculate_y_limits(working_df, channels, [sensor])
        
        # Add some padding
        y_range = y_max - y_min
        y_min = y_min - 0.05 * y_range
        y_max = y_max + 0.05 * y_range
        
        print(f"Auto-scaled y-axis: {y_min:.2f} to {y_max:.2f} (transform: {transform_type}, baseline: {baseline_type})")
        return y_min, y_max
        
    except Exception as e:
        print(f"Error in auto-scaling: {str(e)}")
        return None, None

# Update the callback to manage custom periods
@app.callback(
    [Output('custom-periods-store', 'data'),
     Output('selected-periods-dropdown', 'options'),
     Output('selected-periods-dropdown', 'value'),
     Output('period-selection-warning', 'children')],
    [Input('add-period-button', 'n_clicks'),
     Input('clear-periods-button', 'n_clicks')],
    [State('period-picker', 'start_date'),
     State('period-picker', 'end_date'),
     State('custom-periods-store', 'data'),
     State('selected-periods-dropdown', 'value')]
)
def manage_custom_periods(add_clicks, clear_clicks, start_date, end_date, stored_periods, selected_values):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
        
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    warning_message = ""
    
    if triggered_id == 'clear-periods-button':
        return None, [], [], ""
        
    if triggered_id == 'add-period-button' and start_date and end_date:
        if not stored_periods:
            stored_periods = {}
            
        # Check if we already have 2 periods
        if len(stored_periods) >= 2:
            warning_message = "Maximum 2 periods allowed. Please clear existing periods first."
            options = [{'label': name, 'value': name} for name in stored_periods.keys()]
            return stored_periods, options, selected_values, warning_message
            
        # Add new period
        period_name = f"Period {len(stored_periods) + 1}: {start_date} to {end_date}"
        stored_periods[period_name] = [start_date, end_date]
        
        options = [{'label': name, 'value': name} for name in stored_periods.keys()]
        
        # Automatically select the new period if we have 2 or fewer
        if len(stored_periods) <= 2:
            selected_values = list(stored_periods.keys())
        
        return stored_periods, options, selected_values, warning_message
        
    raise PreventUpdate

# Update the period comparison callback to handle exactly two periods
@app.callback(
    Output('period-comparison-graph', 'figure'),
    [Input('selected-periods-dropdown', 'value'),
     Input('loaded-data-store', 'data'),
     Input('sensor-dropdown', 'value'),
     Input('channel-dropdown', 'value'),
     Input('transform-dropdown', 'value'),
     Input('baseline-radio', 'value'),
     Input('ma-slider', 'value')],
    [State('custom-periods-store', 'data')]
)
def update_period_comparison(selected_periods, stored_data, selected_sensor, selected_channels,
                           transform_type, baseline_type, ma_days, custom_periods):
    if not all([stored_data, custom_periods, selected_sensor, selected_channels]) or len(selected_periods) < 1:
        return go.Figure()
        
    try:
        # Create figure for period comparison
        fig = go.Figure()
        
        # Reconstruct dataframes from stored data
        merged_df1 = pd.DataFrame.from_records(stored_data['df1'], columns=stored_data['columns1'])
        merged_df1['time'] = pd.to_datetime(merged_df1['time'])
        
        # Process data for each selected period
        for i, period_name in enumerate(selected_periods[:2]):  # Limit to first two periods
            start_date, end_date = custom_periods[period_name]
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            # Filter data for this period
            period_data = merged_df1[
                (merged_df1['time'] >= start_dt) & 
                (merged_df1['time'] <= end_dt)
            ].copy()
            
            if period_data.empty:
                print(f"No data found for period {period_name}")
                continue
                
            # Process each channel
            for ch in (selected_channels if isinstance(selected_channels, list) else [selected_channels]):
                col_name = f'{ch}{selected_sensor}'
                if col_name in period_data.columns:
                    # Create a working copy for transformations
                    working_data = period_data.copy()
                    
                    # Apply baseline adjustment if selected
                    if baseline_type == 'adjusted':
                        working_data, _ = apply_baseline_adjustment(working_data)
                    
                    # Apply transformation if selected
                    if transform_type != 'none':
                        working_data, _ = apply_transformation(
                            working_data,
                            transform_type,
                            [col_name]
                        )
                    
                    # Calculate average value for the period
                    avg_value = working_data[col_name].mean()
                    print(f"Average value for {period_name} - {ch}: {avg_value:.3f}")
                    
                    # Normalize time to start from day 0
                    working_data['days'] = (working_data['time'] - working_data['time'].min()).dt.total_seconds() / (24 * 3600)
                    
                    # Calculate daily means
                    daily_data = working_data.groupby('days')[col_name].mean()
                    
                    # Apply moving average
                    ma_data = daily_data.rolling(window=ma_days, min_periods=1).mean()
                    
                    # Apply sigma filter and smoothing for trend
                    temp_df = pd.DataFrame({'value': ma_data.values}, index=ma_data.index)
                    filtered_series, smoothed_series = apply_sigma_filter_and_smooth(
                        temp_df, 'value',
                        sigma_threshold=2.0,
                        window_size=15,
                        poly_order=3
                    )
                    
                    # Add raw data trace with distinctive styling
                    line_style = 'solid' if i == 0 else 'dash'
                    period_num = i + 1
                    
                    # Add moving average trace
                    fig.add_trace(go.Scatter(
                        x=ma_data.index,
                        y=ma_data.values,
                        mode='lines',
                        name=f'Period {period_num}: {start_date} to {end_date} - {ch} (MA)',
                        line=dict(
                            color=COLORS[ch],
                            width=1,
                            dash=line_style
                        ),
                        opacity=0.3
                    ))
                    
                    # Add smoothed trend
                    fig.add_trace(go.Scatter(
                        x=ma_data.index,
                        y=smoothed_series.values,
                        mode='lines',
                        name=f'Period {period_num}: {ch} (Trend)',
                        line=dict(
                            color=COLORS[ch],
                            width=3,
                            dash=line_style
                        ),
                        opacity=1.0
                    ))
                    
                    # Add horizontal line for average
                    fig.add_shape(
                        type="line",
                        x0=ma_data.index.min(),
                        x1=ma_data.index.max(),
                        y0=avg_value,
                        y1=avg_value,
                        line=dict(
                            color=COLORS[ch],
                            width=1,
                            dash="dot"
                        ),
                        opacity=0.5
                    )
                    
                    # Add annotation for average value
                    fig.add_annotation(
                        x=ma_data.index.max(),
                        y=avg_value,
                        text=f"Avg: {avg_value:.3f}",
                        showarrow=False,
                        xanchor="left",
                        yanchor="bottom",
                        xshift=10,
                        font=dict(
                            color=COLORS[ch],
                            size=10
                        )
                    )
                    
                    print(f"Added traces for {period_name} - {ch} with {len(ma_data)} points")
        
        # Update layout
        fig.update_layout(
            title=dict(
                text='Period Comparison (Aligned from Day 0)',
                y=0.95
            ),
            xaxis_title='Days from Period Start',
            yaxis_title='Value',
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02,
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="rgba(128, 128, 128, 0.2)",
                borderwidth=1
            ),
            margin=dict(r=150, t=50),
            plot_bgcolor='white',
            height=400,
            hovermode='x unified'
        )
        
        return fig
        
    except Exception as e:
        print(f"Error updating period comparison: {str(e)}")
        return go.Figure()

if __name__ == "__main__":
    # Get port from environment variable or use default
    port = int(os.environ.get('PORT', 8057))
    
    # Get host from environment variable or use default
    host = os.environ.get('HOST', '0.0.0.0')
    
    # Run the server with specified host and port
    app.run_server(
        host=host,
        port=port,
        debug=False  # Set debug to False for production
    )