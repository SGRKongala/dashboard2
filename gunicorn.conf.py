# Gunicorn configuration for memory-constrained environments
import os
import gc
import sys

# Force garbage collection at startup
gc.collect()

# Worker settings
workers = 1  # Use only one worker
threads = 2  # Use 2 threads per worker
worker_class = 'sync'  # Use sync worker

# Timeouts
timeout = 120  # 2 minute timeout
keepalive = 5  # 5 seconds keepalive

# Memory management
max_requests = 5  # Restart workers after 5 requests
max_requests_jitter = 2  # Add jitter to prevent all workers restarting at once

# Logging
loglevel = 'info'
accesslog = '-'
errorlog = '-'

# Bind to the port Render expects
bind = f"0.0.0.0:{os.environ.get('PORT', '8050')}"

# Set environment variables to control memory usage
os.environ['MALLOC_ARENA_MAX'] = '2'
os.environ['PYTHONMALLOC'] = 'malloc'

def on_starting(server):
    """Log when server starts"""
    print("Starting server with memory-optimized settings")
    
    # Set environment variable to indicate we're running in Gunicorn
    os.environ['RUNNING_IN_GUNICORN'] = 'true'

def post_fork(server, worker):
    """After forking, clean up memory"""
    gc.collect()
    print("Worker forked, memory cleaned") 