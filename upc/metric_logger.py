import logging
import json
import time
import functools
from datetime import datetime
from logging.handlers import RotatingFileHandler
import os
from typing import Optional, Any, Dict
from upc.config import FunctionConfig


class MetricLogger:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MetricLogger, cls).__new__(cls)
            cls._instance._initialize_logger()
        return cls._instance

    def _initialize_logger(self):
        """Initialize the metric logger with custom formatting"""
        self.logger = logging.getLogger('metrics')
        self.logger.setLevel(logging.INFO)

        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)

        # Create rotating file handler
        handler = RotatingFileHandler(
            FunctionConfig.METRIC_LOG_FILE,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )

        # Set formatter based on config
        if FunctionConfig.METRIC_LOG_FORMAT == 'json':
            formatter = logging.Formatter('%(message)s')
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )

        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def log_metric(self,
                   function_name: str,
                   execution_time: float,
                   status: str = "success",
                   request_id: Optional[str] = None,
                   additional_data: Optional[Dict[str, Any]] = None):
        """
        Log metric information for a function execution
        """
        function_config = FunctionConfig.FUNCTION_CONFIGS.get(function_name, {})
        expected_duration = function_config.get('expected_duration', 0)

        metric_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "function": function_name,
            "execution_time": round(execution_time, 3),
            "expected_duration": expected_duration,
            "status": status,
            "request_id": request_id or "unknown",
        }

        if additional_data:
            metric_data.update(additional_data)

        # Check if execution time exceeds threshold
        if (expected_duration > 0 and
                execution_time > expected_duration * FunctionConfig.METRIC_ALERT_THRESHOLD):
            metric_data["alert"] = "Execution time exceeded threshold"
            log_level = logging.WARNING
        else:
            log_level = logging.INFO

        if FunctionConfig.METRIC_LOG_FORMAT == 'json':
            self.logger.log(log_level, json.dumps(metric_data))
        else:
            self.logger.log(
                log_level,
                f"Function: {function_name}, Time: {execution_time:.3f}s, Status: {status}"
            )


def track_execution_time(request_id_arg: Optional[str] = None):
    """
    Decorator to track function execution time and log metrics

    Args:
        request_id_arg: Name of the argument containing request ID (e.g., 'conversation_id')
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"
            request_id = None

            try:
                # Try to get request_id from arguments if specified
                if request_id_arg:
                    if request_id_arg in kwargs:
                        request_id = kwargs[request_id_arg]
                    elif args and hasattr(args[0], request_id_arg):
                        request_id = getattr(args[0], request_id_arg)

                result = func(*args, **kwargs)
                return result

            except Exception as e:
                status = "error"
                raise

            finally:
                execution_time = time.time() - start_time
                metric_logger = MetricLogger()

                additional_data = {
                    "module": func.__module__,
                    "function_name": func.__name__
                }

                metric_logger.log_metric(
                    function_name=func.__name__,
                    execution_time=execution_time,
                    status=status,
                    request_id=request_id,
                    additional_data=additional_data
                )

        return wrapper

    return decorator