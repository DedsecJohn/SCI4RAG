"""
Unified logging system for SCI4RAG
Provides consistent logging across all modules with file rotation and colored output
Supports global, user-level, and dataset-level logging
"""
import sys
from pathlib import Path
from typing import Optional
from loguru import logger
from src.core.paths import logs_dir, logs_sci4rag, logs_error, ensure_dir


class Logger:
    """Centralized logger configuration for SCI4RAG"""
    
    def __init__(self, log_dir: str = "logs", log_level: str = "INFO"):
        """
        Initialize global logger with file and console handlers
        
        Args:
            log_dir: Directory to store log files
            log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.log_dir = ensure_dir(logs_dir())
        self.log_level = log_level
        
        # Remove default handler
        logger.remove()
        
        # Add console handler with colors
        logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level=log_level,
            colorize=True
        )
        
        # Add global file handler for all logs
        logger.add(
            logs_sci4rag(),
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level="DEBUG",
            rotation="10 MB",
            retention="30 days",
            compression="zip",
            encoding="utf-8"
        )
        
        # Add global file handler for errors only
        logger.add(
            logs_error(),
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level="ERROR",
            rotation="5 MB",
            retention="60 days",
            compression="zip",
            encoding="utf-8"
        )
        
        logger.info(f"Global logger initialized with level: {log_level}")
    
    def get_logger(self):
        """Get the configured logger instance"""
        return logger


class UserLogger:
    """User-specific logger that writes to user's log directory"""
    
    def __init__(self, username: str, dataset_name: Optional[str] = None, log_level: str = "INFO"):
        """
        Initialize user-specific logger
        
        Args:
            username: Username for the log directory
            dataset_name: Optional dataset name for dataset-specific logs
            log_level: Minimum log level
        """
        self.username = username
        self.dataset_name = dataset_name
        self.log_level = log_level
        
        # Determine log directory using paths module
        self.log_dir = ensure_dir(logs_dir(username, dataset_name))
        self.log_prefix = f"{username}/{dataset_name}" if dataset_name else username
        
        # Create a new logger instance with unique ID
        self.logger_id = f"user_{username}_{dataset_name or 'global'}"
        
        # Add user-specific file handler
        logger.add(
            self.log_dir / "activity.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
            level="INFO",
            rotation="5 MB",
            retention="90 days",
            compression="zip",
            encoding="utf-8",
            filter=lambda record: record["extra"].get("user_logger_id") == self.logger_id
        )
        
        # Add user-specific error log
        logger.add(
            self.log_dir / "error.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level="ERROR",
            rotation="2 MB",
            retention="90 days",
            compression="zip",
            encoding="utf-8",
            filter=lambda record: record["extra"].get("user_logger_id") == self.logger_id
        )
        
        logger.bind(user_logger_id=self.logger_id).info(
            f"User logger initialized for: {self.log_prefix}"
        )
    
    def get_logger(self):
        """Get the user-specific logger with bound context"""
        return logger.bind(user_logger_id=self.logger_id, user=self.username, dataset=self.dataset_name)


# Global logger instance
_global_logger = None
_user_loggers = {}


def get_logger():
    """
    Get the global logger instance
    
    Returns:
        loguru.Logger: Configured global logger instance
    
    Example:
        >>> from src.core.logger import get_logger
        >>> logger = get_logger()
        >>> logger.info("System started")
    """
    global _global_logger
    if _global_logger is None:
        _global_logger = Logger()
    return _global_logger.get_logger()


def get_user_logger(username: str, dataset_name: Optional[str] = None):
    """
    Get a user-specific logger instance
    
    Args:
        username: Username for the logger
        dataset_name: Optional dataset name for dataset-specific logging
    
    Returns:
        loguru.Logger: User-specific logger instance
    
    Example:
        >>> from src.core.logger import get_user_logger
        >>> logger = get_user_logger("admin", "schwarz")
        >>> logger.info("Processing document")
        >>> # Logs to: users/admin/schwarz/logs/activity.log
    """
    global _user_loggers
    
    key = f"{username}_{dataset_name or 'global'}"
    if key not in _user_loggers:
        _user_loggers[key] = UserLogger(username, dataset_name)
    
    return _user_loggers[key].get_logger()


def set_log_level(level: str):
    """
    Change the global log level
    
    Args:
        level: New log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    global _global_logger
    _global_logger = Logger(log_level=level)
    logger.info(f"Log level changed to: {level}")


# Convenience functions for global logger
def debug(message: str, **kwargs):
    """Log debug message to global logger"""
    get_logger().debug(message, **kwargs)


def info(message: str, **kwargs):
    """Log info message to global logger"""
    get_logger().info(message, **kwargs)


def warning(message: str, **kwargs):
    """Log warning message to global logger"""
    get_logger().warning(message, **kwargs)


def error(message: str, **kwargs):
    """Log error message to global logger"""
    get_logger().error(message, **kwargs)


def critical(message: str, **kwargs):
    """Log critical message to global logger"""
    get_logger().critical(message, **kwargs)


def success(message: str, **kwargs):
    """Log success message to global logger"""
    get_logger().success(message, **kwargs)


if __name__ == "__main__":
    print("=== Testing Global Logger ===")
    global_logger = get_logger()
    global_logger.info("Global system message")
    global_logger.warning("Global warning")
    
    print("\n=== Testing User Logger ===")
    user_logger = get_user_logger("admin")
    user_logger.info("User admin logged in")
    user_logger.success("User operation completed")
    
    print("\n=== Testing Dataset Logger ===")
    dataset_logger = get_user_logger("admin", "schwarz")
    dataset_logger.info("Processing document: paper1.pdf")
    dataset_logger.success("Document parsed successfully")
    dataset_logger.warning("Missing DOI information")
    
    print("\n=== Testing Another User ===")
    user2_logger = get_user_logger("researcher", "test")
    user2_logger.info("Starting experiment")
    user2_logger.error("Experiment failed")
    
    print("\n=== Testing Exception Logging ===")
    try:
        1 / 0
    except Exception:
        dataset_logger.exception("Error during processing")
    
    print("\n=== Log Files Created ===")
    print(f"Global logs: {Path('logs').absolute()}")
    print(f"User logs: {Path('users/admin/logs').absolute()}")
    print(f"Dataset logs: {Path('users/admin/schwarz/logs').absolute()}")
    print(f"User2 logs: {Path('users/researcher/test/logs').absolute()}")
