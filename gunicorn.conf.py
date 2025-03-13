# Gunicorn configuration for memory-constrained environments
import os

# Worker settings
workers = 1  # Use only one worker
threads = 2  # Use 2 threads per worker
worker_class = 'gthread'  # Use threaded worker

# Timeouts
timeout = 120  # 2 minute timeout
keepalive = 5  # 5 seconds keepalive

# Memory management
max_requests = 10  # Restart workers after 10 requests
max_requests_jitter = 3  # Add jitter to prevent all workers restarting at once

# Logging
loglevel = 'info'
accesslog = '-'
errorlog = '-'

# Bind to the port Render expects
bind = f"0.0.0.0:{os.environ.get('PORT', '8050')}"

# Preload app to save memory
preload_app = True

def on_starting(server):
    """Log when server starts"""
    import logging
    logging.info("Starting server with memory-optimized settings") 