# Gunicorn configuration for extremely memory-constrained environments
import os
import sys

# Force garbage collection at startup
import gc
gc.collect()

# Worker settings - absolute minimum
workers = 1
threads = 1
worker_class = 'sync'  # Simplest worker type

# Timeouts
timeout = 120
keepalive = 5

# Memory management - very aggressive
max_requests = 5  # Restart workers frequently
max_requests_jitter = 2

# Logging
loglevel = 'info'
accesslog = '-'
errorlog = '-'

# Bind to the port Render expects
bind = f"0.0.0.0:{os.environ.get('PORT', '8050')}"

# Don't preload app to save memory
preload_app = False

# Set lower memory limits for Python
import resource
# 400MB soft limit, 450MB hard limit
resource.setrlimit(resource.RLIMIT_AS, (400 * 1024 * 1024, 450 * 1024 * 1024))

def on_starting(server):
    """Log when server starts and set memory limits"""
    print("Starting server with extremely memory-optimized settings")
    
    # Set pandas options to minimize memory
    try:
        import pandas as pd
        pd.options.mode.chained_assignment = None
        pd.options.mode.use_inf_as_na = True
    except:
        pass 