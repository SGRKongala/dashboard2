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
# Add new imports for AWS
import boto3
from botocore.config import Config
import tempfile
import socket
from flask import Flask
import io
import requests

# Initialize data cache
data_cache = {}
db_loaded = False
all_tables = {}

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

# AWS Configuration
# Define your bucket name and S3 object details
S3_BUCKET = 'public-tarucca-db'
S3_REGION = 'eu-central-1'
S3_DB_URL = f'https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/text.db'

# Initialize S3 client
s3 = None
try:
    # Initialize S3 client without credentials for public bucket
    s3 = boto3.client(
        "s3",
        region_name=S3_REGION,
        config=Config(
            connect_timeout=5,
            read_timeout=300,
            retries={'max_attempts': 3}
        )
    )
    # Test the connection by listing objects (should work for public buckets)
    s3.list_objects_v2(Bucket=S3_BUCKET, MaxKeys=1)
    print("AWS S3 connection verified successfully")
except Exception as e:
    print(f"AWS Configuration Error: {str(e)}")
    s3 = None

# Function to load all tables from the database once
def load_database():
    """Load all tables from the database once"""
    global db_loaded, all_tables
    
    if db_loaded:
        return True
    
    start_time = time.time()
    
    try:
        print("Loading database tables...")
        
        # Create a temporary directory for the database file
        temp_dir = os.path.join(tempfile.gettempdir(), 'dashboard_db')
        os.makedirs(temp_dir, exist_ok=True)
        temp_db_path = os.path.join(temp_dir, 'text.db')
        
        # Download the database file from S3 if not already downloaded
        if not os.path.exists(temp_db_path):
            print(f"Downloading database from {S3_DB_URL}")
            response = requests.get(S3_DB_URL, stream=True)
            response.raise_for_status()  # Raise an exception for HTTP errors
            
            with open(temp_db_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"Database downloaded successfully to {temp_db_path}")
        else:
            print(f"Using existing database file at {temp_db_path}")
        
        # Connect to the database
        conn = sqlite3.connect(temp_db_path)
        
        # List tables in the database
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        print(f"Available tables in database: {tables}")
        
        # Load all tables into memory
        for table in tables:
            try:
                all_tables[table.lower()] = pd.read_sql(f"SELECT * FROM {table}", conn)
                print(f"Loaded table {table} with {len(all_tables[table.lower()])} rows")
            except Exception as e:
                print(f"Error loading table {table}: {str(e)}")
        
        conn.close()
        db_loaded = True
        print(f"Database loading completed in {time.time() - start_time:.2f} seconds")
        return True
        
    except Exception as e:
        print(f"Error loading database: {str(e)}")
        return False

# Update the load_data_cached function to use the preloaded tables
def load_data_cached(metric):
    """Load data with caching to improve performance"""
    global data_cache
    
    if metric in data_cache:
        return data_cache[metric]
    
    start_time = time.time()
    
    try:
        print(f"Starting data load for {metric}...")
        
        # Load database tables if not already loaded
        if not db_loaded:
            success = load_database()
            if not success:
                raise ValueError("Failed to load database")
        
        # Find the table name with case-insensitive matching
        metric_table_name = None
        for table_name in all_tables.keys():
            if table_name == metric.lower():
                metric_table_name = table_name
                break
        
        # If not found, try with variations
        if not metric_table_name:
            for table_name in all_tables.keys():
                if metric.lower() in table_name or table_name in metric.lower():
                    metric_table_name = table_name
                    break
        
        # Get the data
        if metric_table_name and metric_table_name in all_tables:
            df1 = all_tables[metric_table_name].copy()
            print(f"Using data from table {metric_table_name}")
        else:
            print(f"No matching table found for {metric}")
            raise ValueError(f"No matching table found for {metric}")
        
        # Get RPM data
        if 'rpm' in all_tables:
            df2 = all_tables['rpm'].copy()
            print("Using RPM data from rpm table")
        else:
            # If no rpm table, create one based on the first dataframe
            print("No rpm table found, creating synthetic RPM data")
            if 'id' in df1.columns and 'time' in df1.columns:
                unique_ids = df1['id'].unique()
                unique_times = df1['time'].unique()
                
                # Create a dataframe with unique id/time combinations
                id_time_pairs = []
                for id_val in unique_ids:
                    for time_val in unique_times:
                        id_time_pairs.append((id_val, time_val))
                
                df2 = pd.DataFrame(id_time_pairs, columns=['id', 'time'])
                df2['ch1s1'] = np.random.uniform(800, 1200, len(df2))
            else:
                # If no id/time columns, create completely synthetic data
                dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
                df2 = pd.DataFrame({
                    'id': range(100),
                    'time': dates,
                    'ch1s1': np.random.uniform(800, 1200, 100)
                })
        
        # Process data
        if not df1.empty and not df2.empty:
            # Ensure required columns exist
            if 'id' not in df1.columns:
                df1['id'] = range(len(df1))
            if 'time' not in df1.columns:
                df1['time'] = pd.date_range(start='2023-01-01', periods=len(df1), freq='D')
            
            if 'id' not in df2.columns:
                df2['id'] = range(len(df2))
            if 'time' not in df2.columns:
                df2['time'] = pd.date_range(start='2023-01-01', periods=len(df2), freq='D')
            if 'ch1s1' not in df2.columns:
                df2['ch1s1'] = np.random.uniform(800, 1200, len(df2))
            
            # Ensure channel columns exist
            for ch in CHANNELS:
                for s in SENSORS:
                    col_name = f"{ch}{s}"
                    if col_name not in df1.columns:
                        df1[col_name] = np.random.normal(10, 2, len(df1))
            
            # Convert time columns to datetime
            try:
                # Ensure time columns are properly formatted
                if 'time' in df1.columns:
                    # Check if time is already a datetime
                    if not pd.api.types.is_datetime64_any_dtype(df1['time']):
                        # Try different formats
                        try:
                            df1['time'] = pd.to_datetime(df1['time'])
                        except:
                            # If that fails, try with a specific format
                            try:
                                df1['time'] = pd.to_datetime(df1['time'], format='%Y-%m-%d')
                            except:
                                # Last resort: create a date range
                                df1['time'] = pd.date_range(start='2023-01-01', periods=len(df1), freq='D')
                
                if 'time' in df2.columns:
                    if not pd.api.types.is_datetime64_any_dtype(df2['time']):
                        try:
                            df2['time'] = pd.to_datetime(df2['time'])
                        except:
                            try:
                                df2['time'] = pd.to_datetime(df2['time'], format='%Y-%m-%d')
                            except:
                                df2['time'] = pd.date_range(start='2023-01-01', periods=len(df2), freq='D')
            except Exception as e:
                print(f"Error converting time columns: {str(e)}")
                # Create time columns if they can't be converted
                df1['time'] = pd.date_range(start='2023-01-01', periods=len(df1), freq='D')
                df2['time'] = pd.date_range(start='2023-01-01', periods=len(df2), freq='D')
            
            # Clean data
            for col in df1.columns:
                if col not in ['id', 'time']:
                    try:
                        # Replace corrupted values with NaN
                        mask = ~df1[col].astype(str).str.match(r'^-?\d+(\.\d+)?$')
                        corrupted_count = mask.sum()
                        if corrupted_count > 0:
                            df1.loc[mask, col] = np.nan
                            print(f"Set {corrupted_count} corrupted values to NaN for {col}")
                    except Exception as e:
                        print(f"Error cleaning column {col}: {str(e)}")
            
            # Store in cache
            data_cache[metric] = (df1, df2)
            print(f"Data load completed in {time.time() - start_time:.2f} seconds")
            print(f"Loaded {len(df1)} rows")
            return df1, df2
        else:
            raise ValueError("Empty dataframes returned from database")
            
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        print("Creating synthetic data as final fallback")
        
        # Create synthetic data as final fallback
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        
        # Create sample metric data with all required columns
        df1 = pd.DataFrame({
            'id': range(100),
            'time': dates
        })
        
        # Add channel columns
        for ch in CHANNELS:
            for s in SENSORS:
                col_name = f"{ch}{s}"
                # Create some patterns in the data
                base = np.sin(np.linspace(0, 4*np.pi, 100)) * 5 + 10
                noise = np.random.normal(0, 1, 100)
                df1[col_name] = base + noise
        
        # Create sample RPM data
        df2 = pd.DataFrame({
            'id': range(100),
            'time': dates,
            'ch1s1': np.random.uniform(800, 1200, 100)
        })
        
        # Store in cache
        data_cache[metric] = (df1, df2)
        print("Created synthetic data as final fallback")
        return df1, df2

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
                if col in ['time', 'id'] or not any(ch in col for ch in CHANNELS):
                    continue
                
                try:
                    # Get data from end of current period
                    current_end_mask = (adjusted_df[time_col] >= current_end_range_start) & (adjusted_df[time_col] <= current_period_end)
                    current_end_data = adjusted_df.loc[current_end_mask, col].dropna()
                    
                    # Get data from start of next period
                    next_start_mask = (adjusted_df[time_col] >= next_period_start) & (adjusted_df[time_col] <= next_start_range_end)
                    next_start_data = adjusted_df.loc[next_start_mask, col].dropna()
                    
                    # Check if we have enough data points
                    if len(current_end_data) < 3 or len(next_start_data) < 3:
                        print(f"  Insufficient data for {col} at boundary between {current_period_name} and {next_period_name}")
                        continue
                    
                    # Calculate means for both periods
                    current_end_mean = current_end_data.mean()
                    next_start_mean = next_start_data.mean()
                    
                    # Calculate adjustment (difference between means)
                    adjustment = current_end_mean - next_start_mean
                    
                    # Only apply significant adjustments (more than 5% of data range)
                    data_range = adjusted_df[col].max() - adjusted_df[col].min()
                    if abs(adjustment) > 0.05 * data_range:
                        # Apply adjustment to all data points in the next period and beyond
                        future_mask = adjusted_df[time_col] >= next_period_start
                        adjusted_df.loc[future_mask, col] = adjusted_df.loc[future_mask, col] + adjustment
                        
                        # Store the adjustment for reference
                        adjustment_key = f"{current_period_name}-{next_period_name}_{col}"
                        adjustments[adjustment_key] = adjustment
                except Exception as e:
                    print(f"  Error adjusting {col}: {str(e)}")
                    continue
        
        print(f"Applied baseline adjustments: {adjustments}")
        return adjusted_df, adjustments
    
    except Exception as e:
        print(f"Error in baseline adjustment: {str(e)}")
        return df, {}  # Return original data if adjustment fails

def apply_data_transformation(df, transform_type):
    """Apply various data transformations to the dataframe."""
    # Create a copy to avoid modifying the original
    transformed_df = df.copy()
    transform_description = "No Transformation"
    
    # Get all sensor columns
    sensor_cols = [col for col in df.columns if col not in ['id', 'time']]
    
    if transform_type != 'none':
        for col in sensor_cols:
            try:
                # Skip columns with non-numeric data or all NaN values
                if not pd.api.types.is_numeric_dtype(df[col]) or df[col].isna().all():
                    continue
                    
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
            except Exception as e:
                print(f"Error transforming column {col}: {str(e)}")
                # Keep the original values for this column
                transformed_df[col] = df[col]
    
    return transformed_df, transform_description

def apply_sigma_filter_and_smooth(df, column, sigma_threshold=2.0, window_size=7, poly_order=3):
    """
    Apply a sigma filter to remove outliers and then smooth the data with Savitzky-Golay
    
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

# Initialize Dash app with Flask server
server = Flask(__name__)
app = dash.Dash(
    __name__, 
    server=server,
    url_base_pathname='/'  # Root path
)
server = app.server  # This line is important for gunicorn

# Load all database tables at startup
print("Loading database tables at startup...")
load_database()
print("Database loading complete")

# Load initial data
try:
    print("Loading initial data...")
    initial_df1, initial_df2 = load_data_cached('std_dev')
    y_min, y_max = calculate_y_limits(initial_df1, CHANNELS, SENSORS)
    print(f"Initial data loaded: {len(initial_df1)} rows")
except Exception as e:
    print(f"Error loading initial data: {str(e)}")
    initial_df1 = pd.DataFrame(columns=['id', 'time'])
    initial_df2 = pd.DataFrame(columns=['id', 'time', 'ch1s1'])
    y_min, y_max = 0, 1

# App Layout
app.layout = html.Div([
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
                min_date_allowed=initial_df1['time'].min().date() if not initial_df1.empty else datetime.now().date(),
                max_date_allowed=initial_df1['time'].max().date() if not initial_df1.empty else datetime.now().date(),
                start_date=initial_df1['time'].min().date() if not initial_df1.empty else datetime.now().date(),
                end_date=initial_df1['time'].max().date() if not initial_df1.empty else datetime.now().date()
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
        children='Loading data... This may take a few minutes on first load.',
        style={
            'textAlign': 'center',
            'padding': '20px',
            'fontSize': '18px',
            'color': '#666'
        }
    )
])

@app.callback(
    [Output('sensor-graph', 'figure'),
     Output('loading-output', 'children')],
    [Input('metric-dropdown', 'value'),
     Input('sensor-dropdown', 'value'),
     Input('channel-dropdown', 'value'),
     Input('rpm-range-slider', 'value'),
     Input('date-picker', 'start_date'),
     Input('date-picker', 'end_date'),
     Input('y-min-input', 'value'),
     Input('y-max-input', 'value'),
     Input('ma-slider', 'value'),
     Input('transform-dropdown', 'value'),
     Input('baseline-radio', 'value')]
)
def update_graph(selected_metric, selected_sensor, selected_channels, rpm_range, 
                start_date, end_date, y_min, y_max, ma_days, transform_type, baseline_option):
    try:
        if not all([selected_metric, selected_sensor, selected_channels]):
            raise PreventUpdate

        # Load data
        merged_df1, merged_df2 = load_data_cached(selected_metric)
        
        # Convert dates once
        start_dt = pd.to_datetime(start_date).date()
        end_dt = pd.to_datetime(end_date).date()
        
        # Filter by date efficiently
        mask = (merged_df1['time'].dt.date >= start_dt) & \
               (merged_df1['time'].dt.date <= end_dt)
        df_filtered = merged_df1.loc[mask].copy()
        
        # Filter by RPM range efficiently
        rpm_min, rpm_max = rpm_range  # Using rpm_range from the slider
        rpm_mask = (merged_df2['ch1s1'] >= rpm_min) & \
                  (merged_df2['ch1s1'] < rpm_max)
        rpm_filtered = merged_df2.loc[rpm_mask, ['id', 'time']]
        
        # Merge filtered data
        final_df = pd.merge(
            df_filtered, 
            rpm_filtered, 
            on=['id', 'time'],
            how='inner'
        )
        
        # Sort by time for proper processing
        final_df = final_df.sort_values('time')
        
        # Clear memory
        del df_filtered, rpm_filtered
        
        if final_df.empty:
            empty_fig = go.Figure()
            empty_fig.update_layout(
                title="No data available for the selected filters",
                annotations=[dict(
                    text="No data available for the selected filters",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5
                )]
            )
            return empty_fig, "No data available for the selected filters"
        
        # Apply baseline adjustment if selected
        baseline_text = "Raw Data"
        if baseline_option == 'adjusted':
            final_df, adjustments = apply_baseline_adjustment(final_df)
            if adjustments:
                print(f"Applied baseline adjustments: {adjustments}")
            baseline_text = "Baseline Adjusted"
        
        # Apply transformation if selected
        transform_description = "No Transformation"
        if transform_type != 'none':
            # Get all columns to transform
            columns_to_transform = [f'{ch}{selected_sensor}' for ch in selected_channels]
            final_df, transform_description = apply_transformation(final_df, transform_type, columns_to_transform)
        
        # Create figure
        fig = go.Figure()
        
        # Process each channel
        for ch in (selected_channels if isinstance(selected_channels, list) else [selected_channels]):
            col_name = f'{ch}{selected_sensor}'
            if col_name in final_df.columns:
                # First, calculate daily means to reduce noise
                daily_data = final_df.set_index('time')[col_name].resample('D').mean()
                
                # Then apply the moving average to the daily data
                ma_data = daily_data.rolling(window=ma_days, min_periods=1).mean()
                
                if not ma_data.empty:
                    # Add the raw moving average line (lighter, with markers)
                    fig.add_trace(go.Scatter(
                        x=ma_data.index,
                        y=ma_data.values,
                        mode='lines+markers',
                        name=f'{ch} ({ma_days}-day MA)',
                        line=dict(color=COLORS[ch], width=1.5),
                        marker=dict(color=COLORS[ch], size=5),
                        opacity=0.7
                    ))
                    
                    # Apply sigma filter and smoothing to the MA data
                    try:
                        # Create a temporary dataframe with the MA data
                        temp_df = pd.DataFrame({'value': ma_data.values}, index=ma_data.index)
                        
                        # Apply sigma filter and smoothing
                        filtered_series, smoothed_series = apply_sigma_filter_and_smooth(
                            temp_df, 'value', 
                            sigma_threshold=2.0,  # 2 standard deviations
                            window_size=15,       # 15-day window for smoothing
                            poly_order=3          # 3rd order polynomial
                        )
                        
                        # Add the filtered data (optional)
                        fig.add_trace(go.Scatter(
                            x=ma_data.index,
                            y=filtered_series.values,
                            mode='markers',
                            name=f'{ch} (Filtered)',
                            marker=dict(color=COLORS[ch], size=4, symbol='circle-open'),
                            opacity=0.5
                        ))
                        
                        # Add the smoothed curve (thicker, no markers)
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
        
        # Add period markers
        for period_name, (start_date, end_date) in COMPARISON_PERIODS.items():
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            # Add vertical lines at period boundaries
            fig.add_vline(x=start_dt, line_dash="dash", line_color="gray", opacity=0.7)
            fig.add_vline(x=end_dt, line_dash="dash", line_color="gray", opacity=0.7)
            
            # Add period annotation
            fig.add_annotation(
                x=(start_dt + (end_dt - start_dt)/2),
                y=1.05,
                text=period_name,
                showarrow=False,
                xref="x",
                yref="paper",
                font=dict(size=12, color="black"),
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="gray",
                borderwidth=1
            )
            
            # Add shaded area for period
            fig.add_shape(
                type="rect",
                x0=start_dt,
                x1=end_dt,
                y0=0,
                y1=1,
                xref="x",
                yref="paper",
                fillcolor="gray",
                opacity=0.1,
                layer="below",
                line_width=0,
            )
        
        # Update layout
        title_text = f'{selected_metric.replace("_", " ").title()} - Sensor {selected_sensor}'
        if baseline_option == 'adjusted':
            title_text += ' (Baseline Adjusted)'
        if transform_type != 'none':
            title_text += f' ({transform_description})'
            
        rpm_text = f"RPM Range: {rpm_min}-{rpm_max}"
            
        fig.update_layout(
            title=title_text,
            xaxis_title='Time',
            yaxis_title=f'{selected_metric.replace("_", " ").title()} Value',
            yaxis=dict(range=[y_min, y_max]),
            height=600,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(b=100)  # Add bottom margin for annotation
        )
        
        # Add note about data processing
        processing_note = f"{baseline_text} | {rpm_text} | Sigma-filtered (2σ) with Savitzky-Golay smoothing"
        if transform_type != 'none':
            processing_note += f" | {transform_description}"
            
        fig.add_annotation(
            text=processing_note,
            xref="paper", yref="paper",
            x=0.5, y=-0.15,
            showarrow=False,
            font=dict(size=10),
            bgcolor="rgba(255, 240, 220, 0.8)",
            bordercolor="orange",
            borderwidth=1
        )
        
        return fig, "Data processed successfully"
        
    except Exception as e:
        print(f"Error: {str(e)}")
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
    [Input('auto-scale-button', 'n_clicks')],
    [State('metric-dropdown', 'value'),
     State('sensor-dropdown', 'value'),
     State('channel-dropdown', 'value')]
)
def update_y_axis_range(n_clicks, metric, sensor, channels):
    if n_clicks == 0:
        raise PreventUpdate
    
    try:
        # Load data
        df1, _ = load_data_cached(metric)
        
        # Calculate y limits
        y_min, y_max = calculate_y_limits(df1, channels, [sensor])
        
        # Add some padding
        y_range = y_max - y_min
        y_min = y_min - 0.05 * y_range
        y_max = y_max + 0.05 * y_range
        
        return y_min, y_max
    except:
        return None, None

if __name__ == "__main__":
    try:
        # Get port from environment variable or use default
        port = int(os.environ.get('PORT', 8051))
        
        # Get host from environment variable or use default
        host = os.environ.get('HOST', '0.0.0.0')
        
        # Run the server with specified host and port
        app.run_server(
            host=host,
            port=port,
            debug=True
        )
    except Exception as e:
        print(f"Failed to start server: {str(e)}")