# Update imports
import boto3
import io
from flask import Flask
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
import socket
from botocore.config import Config

# AWS Configuration
# Use environment variables for AWS credentials
AWS_ACCESS_KEY = os.environ.get('AWS_ACCESS_KEY')
AWS_SECRET_KEY = os.environ.get('AWS_SECRET_KEY')

if not AWS_ACCESS_KEY or not AWS_SECRET_KEY:
    raise ValueError("AWS credentials not found in environment variables")

# Initialize S3 client with custom configuration
try:
    s3 = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name='ap-southeast-2',
        config=Config(
            connect_timeout=5,
            read_timeout=300,
            retries={'max_attempts': 3}
        )
    )
    # Test the connection
    s3.list_buckets()
    print("AWS credentials verified successfully")
except Exception as e:
    print(f"AWS Configuration Error: {str(e)}")
    raise

# Define your bucket name and local file details
BUCKET_NAME = "mytaruccadb1"
LOCAL_FILE_PATH = "DB/text.db"
S3_OBJECT_NAME = "meta_data.db"

# Initialize Flask and Dash
server = Flask(__name__)
app = dash.Dash(
    __name__, 
    server=server,
    url_base_pathname='/'  # Root path
)

# Constants
AVAILABLE_METRICS = ['std_dev', 'rms', 'iqr', 'clean_max', 'clean_min', 'clean_range', 
                    'outlier_count', 'skewness', 'simpson', 'trapz', 'std_error']
SENSORS = ['s1', 's2', 's3', 's4', 's5', 's6']
BINS = np.arange(0, 18, 0.5)
CHANNELS = ['ch1', 'ch2', 'ch3']
COLORS = {'ch1': 'blue', 'ch2': 'red', 'ch3': 'green'}

# Add caching decorator to load_data function
@lru_cache(maxsize=1)  # Reduce cache size
def load_data_cached(metric):
    temp_file = None
    try:
        start_time = time.time()
        print(f"Starting data load for {metric}...")
        
        # Get the object from S3
        response = s3.get_object(Bucket=BUCKET_NAME, Key=S3_OBJECT_NAME)
        
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, mode='wb')
        
        # Download with minimal memory usage
        chunk_size = 16 * 1024 * 1024  # 16MB chunks
        stream = response['Body']
        
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            temp_file.write(chunk)
            
        temp_file.close()
        
        # Create connection and read data efficiently
        with sqlite3.connect(temp_file.name) as conn:
            # Read only necessary columns
            df = pd.read_sql(
                'SELECT id, time FROM main_data',
                conn,
                parse_dates=['time']
            )
            
            # Read RPM data
            df_rpm = pd.read_sql(
                'SELECT id, ch1s1 FROM rpm',
                conn,
                dtype={'ch1s1': 'float32'}
            )
            
            # Read metric data with specific dtypes
            dtype_dict = {col: 'float32' for col in CHANNELS}
            df1 = pd.read_sql(
                f'SELECT * FROM {metric}',
                conn,
                dtype=dtype_dict
            )
            
            # Merge with minimal memory usage
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
            
            # Clear memory
            del df, df1, df_rpm
            
            print(f"Data load completed in {time.time() - start_time:.2f} seconds")
            return merged_df1, merged_df2
            
    except Exception as e:
        print(f"Error loading data from S3: {str(e)}")
        raise
    finally:
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)

# Calculate default y-limits
def calculate_y_limits(df, channels, sensors):
    all_values = []
    for ch in channels:
        for s in sensors:
            col = f'{ch}{s}'
            if col in df.columns:
                all_values.extend(df[col].dropna().values)
    return np.percentile(all_values, [2.5, 97.5])

# Load initial data and calculate y-limits
initial_df1, initial_df2 = load_data_cached('std_dev')
y_min, y_max = calculate_y_limits(initial_df1, CHANNELS, SENSORS)

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
            dcc.Dropdown(
                id='rpm-dropdown',
                options=[{'label': f'{bin:.1f}-{bin+0.5:.1f}', 'value': bin} 
                        for bin in BINS],
                value=BINS[0],
                clearable=False
            ),
        ], style={'width': '30%', 'padding': '10px'}),
        
        html.Div([
            html.Label('Date Range'),
            dcc.DatePickerRange(
                id='date-picker',
                min_date_allowed=initial_df1['time'].min().date(),
                max_date_allowed=initial_df1['time'].max().date(),
                start_date=initial_df1['time'].min().date(),
                end_date=initial_df1['time'].max().date()
            ),
        ], style={'width': '40%', 'padding': '10px'}),
        
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
    ], style={'display': 'flex', 'justifyContent': 'space-between'}),
    
    html.Div([
        html.Label('Y-axis Range'),
        dcc.Input(id='y-min-input', type='number', value=y_min, placeholder='Min Y'),
        dcc.Input(id='y-max-input', type='number', value=y_max, placeholder='Max Y'),
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
        children='Loading data from S3... This may take a few minutes on first load.',
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
     Input('rpm-dropdown', 'value'),
     Input('date-picker', 'start_date'),
     Input('date-picker', 'end_date'),
     Input('y-min-input', 'value'),
     Input('y-max-input', 'value'),
     Input('ma-slider', 'value')]
)
def update_graph(selected_metric, selected_sensor, selected_channels, rpm_bin, 
                start_date, end_date, y_min, y_max, ma_days):
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
        
        # Filter by RPM efficiently
        rpm_mask = (merged_df2['ch1s1'] >= rpm_bin) & \
                  (merged_df2['ch1s1'] < (rpm_bin + 0.5))
        rpm_filtered = merged_df2.loc[rpm_mask, ['id', 'time']]
        
        # Merge filtered data
        final_df = pd.merge(
            df_filtered, 
            rpm_filtered, 
            on=['id', 'time'],
            how='inner'
        )
        
        # Clear memory
        del df_filtered, rpm_filtered
        
        if final_df.empty:
            return {}, "No data available for the selected filters"
        
        # Create figure
        fig = go.Figure()
        
        # Process each channel
        for ch in (selected_channels if isinstance(selected_channels, list) else [selected_channels]):
            col_name = f'{ch}{selected_sensor}'
            if col_name in final_df.columns:
                # Calculate moving average efficiently
                ma_data = (final_df.set_index('time')[col_name]
                          .resample('D')
                          .mean()
                          .rolling(window=ma_days, min_periods=1)
                          .mean())
                
                if not ma_data.empty:
                    fig.add_trace(go.Scatter(
                        x=ma_data.index,
                        y=ma_data.values,
                        mode='lines+markers',
                        name=f'Channel {ch} ({ma_days}-day MA)',
                        line=dict(color=COLORS[ch], width=1.5),
                        marker=dict(color=COLORS[ch], size=5)
                    ))
        
        # Update layout
        fig.update_layout(
            title=f'{selected_metric.replace("_", " ").title()} - Sensor {selected_sensor}',
            xaxis_title='Time',
            yaxis_title='Value',
            yaxis=dict(range=[y_min, y_max]),
            height=600
        )
        
        return fig, "Data processed successfully"
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return {}, f"Error: {str(e)}"

@app.callback(
    Output("download-graph", "data"),
    Input("btn-download", "n_clicks"),
    [State('metric-dropdown', 'value'),
     State('sensor-dropdown', 'value'),
     State('rpm-dropdown', 'value'),
     State('ma-slider', 'value'),
     State('sensor-graph', 'figure')],
    prevent_initial_call=True
)
def download_graph(n_clicks, selected_metric, selected_sensor, rpm_bin, ma_days, figure):
    if n_clicks:
        filename = f'{selected_metric}_Sensor_{selected_sensor}_RPM_{rpm_bin}-{rpm_bin+0.5}_MA_{ma_days}days.png'
        img_bytes = go.Figure(figure).to_image(
            format='png',
            width=1920,
            height=1080,
            scale=2.0
        )
        return dcc.send_bytes(img_bytes, filename)

def find_free_port(start_port=8050, max_port=8070):
    """Find a free port in the given range."""
    for port in range(start_port, max_port + 1):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(('0.0.0.0', port))
            sock.close()
            return port
        except OSError:
            continue
    raise OSError("No free ports found in range")

if __name__ == "__main__":
    try:
        print("Starting Sensor Analysis Dashboard on port 8050")
        app.run_server(debug=False, host='0.0.0.0', port=8050)
    except Exception as e:
        print(f"Failed to start server: {str(e)}")