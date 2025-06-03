# utilities/logging_utils.py
"""
Utility functions for setting up and managing logging.
Based on Project 2's logging_utils.
"""
import logging
import sys
import os

# This global list will be populated by setup_logger with active file handlers
_active_file_handlers = []
_configured_loggers = {} # Keep track of loggers already configured to avoid duplicate handlers

def setup_logger(name: str, log_file: str, level_str: str = None, console_output: bool = True):
    """
    Sets up a logger with specified file and console output.
    Avoids adding duplicate handlers to the same logger instance.

    Args:
        name (str): The name of the logger.
        log_file (str): The path to the log file.
        level_str (str, optional): Logging level as a string (e.g., "INFO", "DEBUG").
                                   Defaults to what might be in config.LOG_LEVEL or INFO.
        console_output (bool): Whether to output logs to the console.

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Determine log level
    if level_str is None:
        try:
            import config # Try to import config to get LOG_LEVEL
            level_str = config.LOG_LEVEL
        except (ImportError, AttributeError):
            level_str = 'INFO' # Fallback if config or config.LOG_LEVEL is not available
            
    level = getattr(logging, level_str.upper(), logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [%(module)s:%(funcName)s:%(lineno)d] - %(message)s')

    logger = logging.getLogger(name)
    
    # If logger is already configured by this function, just return it
    # This prevents re-adding handlers if setup_logger is called multiple times for the same name.
    if name in _configured_loggers and logger.hasHandlers():
        # Update level if different, but don't re-add handlers
        if logger.level != level:
            logger.setLevel(level)
            # print(f"Logger '{name}' already configured. Updated level to {level_str}.") # Debug
        # else:
            # print(f"Logger '{name}' already configured. Returning existing instance.") # Debug
        return logger

    logger.setLevel(level)
    logger.propagate = False # Prevent logs from propagating to the root logger if we add handlers

    # Ensure log directory exists
    log_file_dir = os.path.dirname(log_file)
    if not os.path.exists(log_file_dir):
        try:
            os.makedirs(log_file_dir, exist_ok=True)
        except OSError as e:
            # Fallback to current working directory if log_file_dir creation fails
            sys.stderr.write(f"Error creating log directory {log_file_dir}: {e}\n")
            log_file_basename = os.path.basename(log_file)
            log_file = os.path.join(os.getcwd(), log_file_basename)
            sys.stderr.write(f"Logging to fallback file: {log_file}\n")

    # File handler
    # Check if a file handler for this specific file already exists on this logger
    has_this_file_handler = any(isinstance(h, logging.FileHandler) and h.baseFilename == os.path.abspath(log_file) for h in logger.handlers)
    if not has_this_file_handler:
        try:
            file_handler_instance = logging.FileHandler(log_file, mode='a') # Append mode
            file_handler_instance.setFormatter(formatter)
            logger.addHandler(file_handler_instance)
            if file_handler_instance not in _active_file_handlers: # Avoid duplicates in global list
                _active_file_handlers.append(file_handler_instance)
        except Exception as e:
            sys.stderr.write(f"Error setting up file handler for {log_file} on logger '{name}': {e}\n")

    # Console handler
    if console_output:
        has_console_handler = any(isinstance(h, logging.StreamHandler) and h.stream in [sys.stdout, sys.stderr] for h in logger.handlers)
        if not has_console_handler:
            console_handler = logging.StreamHandler(sys.stdout) # Default to stdout
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            
    _configured_loggers[name] = logger # Mark this logger name as configured by this function
    # print(f"Logger '{name}' configured. Handlers: {logger.handlers}") # Debug
    return logger

def close_all_file_handlers():
    """
    Closes all tracked file handlers and clears the list.
    This should be called before attempting to delete log files.
    """
    global _active_file_handlers
    global _configured_loggers # Also reset configured loggers list

    # print(f"Attempting to close {len(_active_file_handlers)} active file handlers.") # Debug
    
    # Iterate over all known loggers and remove our file handlers from them
    all_logger_names = list(logging.Logger.manager.loggerDict.keys()) + [logging.root.name] # Include root
    for logger_name in all_logger_names:
        logger_instance = logging.getLogger(logger_name)
        for handler in list(logger_instance.handlers): # Iterate over a copy
            if handler in _active_file_handlers:
                try:
                    handler.close()
                    logger_instance.removeHandler(handler)
                    # print(f"Closed and removed handler {handler} from logger '{logger_name}'.") # Debug
                except Exception as e:
                    sys.stderr.write(f"Error closing/removing handler {handler} from logger '{logger_name}': {e}\n")

    # Clear the global tracking list
    _active_file_handlers = []
    _configured_loggers = {} # Reset this so loggers can be reconfigured if app runs again in same process
    # print(f"All tracked file handlers processed. Active list cleared.") # Debug