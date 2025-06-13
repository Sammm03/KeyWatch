# monitor_resources.py

"""
Feature: System Resource Monitor

Description:
This script monitors system's CPU and memory usage in real time.
It logs this information every 5 seconds to a file called `resource_monitor.log`.
Log Location:
    logs/resource_monitor.log
"""

import os                     # For handling file paths and directory creation
import time                   # For sleep delays and time tracking
from datetime import datetime # To get current timestamp for logging
import psutil                 # For accessing system-level CPU and memory usage

def monitor_resources(log_file='logs/resource_monitor.log', interval=5):
    """
    Monitors CPU and memory usage and logs it every 'interval' seconds.

    Args:
        log_file (str): The file path to write the log data.
        interval (int): How often to log resource usage (in seconds).
    """
   # Validate interval
    if interval < 1:
        print(f"⚠️ Interval cannot be less than 1 second. Using 1 second instead.")
        interval = 1
         
    # Ensure logs folder exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    print(f"📊 Resource monitor started. \nLogging to: {log_file} every {interval} seconds.")
    print("Press Ctrl+C to stop monitoring.")

    try:
        with open(log_file, 'a') as f:
            while True:
                cpu = psutil.cpu_percent(interval=1)  # CPU usage in percentage
                memory = psutil.virtual_memory().percent  # Memory usage in percentage
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"{timestamp} | CPU: {cpu}% | Memory: {memory}%\n")
                f.flush()  # Ensure logs are saved immediately
                time.sleep(interval - 1)  # Adjust sleep after 1 sec used by cpu_percent
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    monitor_resources()