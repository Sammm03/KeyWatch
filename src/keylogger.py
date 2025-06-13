# ----------------------------
# KEYLOGGER IMPLEMENTATION
# ----------------------------
"""
KEYLOGGER FEATURES:
1. Real-time keystroke logging with precise timestamps
2. Dwell time measurement (duration key is held down)
3. Flight time calculation (time between key releases)
4. CSV data storage with automatic file creation
5. Error handling for robust operation
6. Secure shutdown via ESC key or Ctrl+C
7. Local storage only - no network transmission
8. Detailed logging for debugging
"""

# System operations
import os          # Directory/file path handling
import sys         # System-level functions (exit, argv)
import signal      # Signal handling (Ctrl+C)
import termios     # Terminal I/O control (input buffering)

# Data handling
import csv         # CSV file read/write operations

# Logging/Time
import logging     # Error/event logging
from datetime import datetime  # Timestamp generation

# Keyboard input
from pynput.keyboard import Listener, Key  # Key press/release detection

# ======================
# CONFIGURATION SETTINGS
# ======================
LOG_DIR = os.path.join(os.path.dirname(__file__), '..', 'logs') # Output directory
CSV_PATH = os.path.join(LOG_DIR, 'keystrokes.csv') # Main data storage

# =================
# GLOBAL VARIABLES
# =================
key_press_time = {}       # Track press time for each key {Key: datetime}
last_release_time = None  # Track time of last key release

# ======================
# LOGGING CONFIGURATION
# ======================
# Initialize logging to track keylogger operations
logging.basicConfig(
    filename=os.path.join(LOG_DIR, 'keylogger.log'),
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# ======================
# FILE SYSTEM SETUP
# ======================
def setup_log_file():
    """Initialize CSV file with headers if it doesn't exist"""
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        if not os.path.isfile(CSV_PATH):
            with open(CSV_PATH, "w", newline='') as file:
                csv.writer(file).writerow([
                    "timestamp", 
                    "key_pressed", 
                    "dwell_time",   # Time key held down (seconds)
                    "flight_time"   # Time between key releases (seconds)
                ])
            logging.info("Created new keystroke log file")
    except Exception as e:
        logging.critical(f"File setup failed: {str(e)}")
        sys.exit(1)

# ======================
# KEYSTROKE PROCESSING
# ======================
def get_key_value(key):
    """Extract readable value from key press event"""
    try:
        return key.char if hasattr(key, 'char') else key.name
    except AttributeError:
        return str(key)

def log_keystroke(key, dwell_time, flight_time):
    """Record keystroke data to CSV with error handling"""
    try:
        with open(CSV_PATH, "a", newline='', buffering=1) as log_file:
            writer = csv.writer(log_file)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            key_value = get_key_value(key)
            
            # Store numerical values with 4 decimal precision
            writer.writerow([
                timestamp,
                key_value,
                round(dwell_time, 4) if dwell_time else 0.0,
                round(flight_time, 4) if flight_time else 0.0
            ])
            log_file.flush()  # Ensure data is written immediately
            
            logging.info(f"Logged: {key_value} (D: {dwell_time:.4f}, F: {flight_time:.4f})")
    except Exception as e:
        logging.error(f"Logging failed: {str(e)}")

# ======================
# KEY EVENT HANDLERS
# ======================
def on_press(key):
    """Track key press timing"""
    global key_press_time
    key_press_time[key] = datetime.now()

def on_release(key):
    """Calculate timing metrics and log keystroke"""
    global last_release_time, key_press_time
    
    current_time = datetime.now()
    dwell_time = 0.0
    flight_time = 0.0

    try:
        # Calculate dwell time (key press duration)
        if key in key_press_time:
            dwell_time = (current_time - key_press_time[key]).total_seconds()
            
        # Calculate flight time (time since last release)
        if last_release_time:
            flight_time = (current_time - last_release_time).total_seconds()
            
        # Log the keystroke
        log_keystroke(key, dwell_time, flight_time)
        
    except Exception as e:
        logging.error(f"Key processing error: {str(e)}")
        
    finally:
        # Update tracking variables
        last_release_time = current_time
        if key in key_press_time:
            del key_press_time[key]

    # Exit mechanism using ESC key
    if key == Key.esc:
        return False

# ======================
# SHUTDOWN HANDLING
# ======================
def handle_exit(sig, frame):
    """Ensure clean shutdown on interrupt signals"""
    logging.info('Received shutdown signal')
    termios.tcflush(sys.stdin, termios.TCIOFLUSH)
    sys.exit(0)

# ======================
# MAIN EXECUTION
# ======================
if __name__ == '__main__':
    # Set up signal handler for CTRL+C
    signal.signal(signal.SIGINT, handle_exit)
    
    # Initialize logging system
    setup_log_file()
    
    try:
        logging.info("Starting keylogger...")
        print("Keylogger running (Press ESC or Ctrl+C to exit)...")
        
        # Start keyboard listener
        with Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()
            
    except Exception as e:
        logging.critical(f"Fatal error: {str(e)}", exc_info=True)
    finally:
        logging.info("Keylogger stopped")
        print("\nKeylogger stopped. Logs saved to:", CSV_PATH)