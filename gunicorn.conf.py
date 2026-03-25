"""
gunicorn.conf.py
----------------
Gunicorn configuration file.
"""

import multiprocessing

# Server socket
bind = "0.0.0.0:5000"
workers = 2

# Timeout
timeout = 120


# Worker hook — runs inside every worker immediately after it forks. Guarantees the model is loaded in every worker before any request arrives
def post_fork(server, worker):
    from api.utils import load_pipeline, load_threshold

    load_pipeline()
    load_threshold()
    server.log.info("Model loaded in worker %s", worker.pid)
