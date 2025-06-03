# utilities/file_utils.py
"""
Utility functions for file and directory operations,
including CSV handling and generic object serialization with pickle.
Model-specific saving (joblib/json) is handled by ModelManager.
Based on Project 2's file_utils.
"""
import os
import pandas as pd
import pickle
import shutil # For clear_directory_contents, though that's in main.py

# Assuming config and logging_utils are accessible for logger setup
# For standalone, a basic logger will be used.
try:
    import config
    from . import logging_utils # Relative import if in the same package
except ImportError:
    # Fallback for direct execution or if imports are tricky
    print("Warning: Could not perform standard config/logging_utils import in file_utils. Using basic logger.")
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    # Mock config for ensure_dir if needed for standalone testing
    class MockConfig:
        LOG_FILE_APP = "file_utils_temp.log" # Example
    if 'config' not in locals() and 'config' not in globals(): config = MockConfig()
    # If logging_utils itself failed to import, logger might not be fully set up by it.
    # This is a utility, so critical logging might not be its primary role.
else:
    logger = logging_utils.setup_logger(__name__, config.LOG_FILE_APP)


def ensure_dir(directory_path: str):
    """
    Ensures that a directory exists. If it doesn't, it creates it.
    Logs information or errors.
    """
    if not os.path.exists(directory_path):
        try:
            os.makedirs(directory_path, exist_ok=True)
            logger.info(f"Directory created: {directory_path}")
        except OSError as e:
            logger.error(f"Error creating directory {directory_path}: {e}", exc_info=True)
            raise # Re-raise the exception as directory creation is often critical
    else:
        if not os.path.isdir(directory_path):
            msg = f"Path {directory_path} exists but is not a directory."
            logger.error(msg)
            raise NotADirectoryError(msg)
        # else:
            # logger.debug(f"Directory already exists: {directory_path}")


def save_dataframe_to_csv(df: pd.DataFrame, file_path: str, index: bool = False, **kwargs):
    """
    Saves a Pandas DataFrame to a CSV file.
    Ensures directory exists. Logs success or failure.
    """
    try:
        ensure_dir(os.path.dirname(file_path))
        df.to_csv(file_path, index=index, **kwargs)
        logger.info(f"DataFrame successfully saved to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving DataFrame to CSV {file_path}: {e}", exc_info=True)
        return False

def load_dataframe_from_csv(file_path: str, **kwargs) -> pd.DataFrame | None:
    """
    Loads a Pandas DataFrame from a CSV file.
    Logs success or failure. Returns None on error.
    """
    if not os.path.exists(file_path):
        logger.warning(f"CSV file not found: {file_path}")
        return None
    try:
        df = pd.read_csv(file_path, **kwargs)
        logger.info(f"DataFrame successfully loaded from {file_path}. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading DataFrame from CSV {file_path}: {e}", exc_info=True)
        return None

def save_object_pickle(obj, file_path: str):
    """
    Saves a Python object to a file using pickle.
    Ensures directory exists. Logs success or failure.
    """
    try:
        ensure_dir(os.path.dirname(file_path))
        with open(file_path, 'wb') as f:
            pickle.dump(obj, f)
        logger.info(f"Object successfully saved using pickle to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving object with pickle to {file_path}: {e}", exc_info=True)
        return False

def load_object_pickle(file_path: str):
    """
    Loads a Python object from a file using pickle.
    Logs success or failure. Returns None on error or if file not found.
    """
    if not os.path.exists(file_path):
        logger.warning(f"Pickle file not found: {file_path}")
        return None
    try:
        with open(file_path, 'rb') as f:
            obj = pickle.load(f)
        logger.info(f"Object successfully loaded using pickle from {file_path}")
        return obj
    except Exception as e:
        logger.error(f"Error loading object with pickle from {file_path}: {e}", exc_info=True)
        return None