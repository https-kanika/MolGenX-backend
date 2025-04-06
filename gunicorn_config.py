import multiprocessing
import os

bind = "0.0.0.0:" + os.environ.get("PORT", "8080")
workers = multiprocessing.cpu_count() * 2 + 1
threads = 4
worker_class = 'gevent'
timeout = 300
keepalive = 5
max_requests = 1000
max_requests_jitter = 50