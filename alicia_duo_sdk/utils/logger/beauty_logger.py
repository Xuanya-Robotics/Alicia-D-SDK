#  Copyright (C) 2024, Junjia Liu
#
#  This file is part of Rofunc.
#
#  Rofunc is licensed under the GNU General Public License v3.0.
#  You may use, distribute, and modify this code under the terms of the GPL-3.0.
#
#  Additional Terms for Commercial Use:
#  Commercial use requires sharing 50% of net profits with the copyright holder.
#  Financial reports and regular payments must be provided as agreed in writing.
#  Non-compliance results in revocation of commercial rights.
#
#  For more details, see <https://www.gnu.org/licenses/>.
#  Contact: skylark0924@gmail.com


import os


# 定义日志级别常量
class LogLevel:
    """日志级别定义"""
    DEBUG = 0
    INFO = 1
    MODULE = 2
    WARNING = 3
    ERROR = 4
    SUCCESS = 5


class BeautyLogger:
    """
    Lightweight logger for Alicia-D-SDK package.
    """

    def __init__(self, log_dir: str, log_name: str = 'rofunc.log', verbose: bool = True, min_level: int = LogLevel.INFO):
        """
        Lightweight logger for Alicia-D-SDK package.

        Example::

            >>> from rofunc.utils.logger import BeautyLogger, LogLevel
            >>> logger = BeautyLogger(log_dir=".", log_name="rofunc.log", verbose=True, min_level=LogLevel.WARNING)

        :param log_dir: the path for saving the log file
        :param log_name: the name of the log file
        :param verbose: whether to print the log to the console
        :param min_level: minimum log level to print (default: LogLevel.INFO)
        """
        self.log_dir = log_dir
        self.log_name = log_name
        self.log_path = os.path.join(self.log_dir, self.log_name)
        self.verbose = verbose
        self.min_level = min_level

        os.makedirs(self.log_dir, exist_ok=True)
        
    def _write_log(self, content, type):
        with open(self.log_path, "a") as f:
            f.write(" Alicia-D-SDK:{}] {}\n".format(type.upper(), content))

    def _should_print(self, level: int) -> bool:
        """
        Check if the log should be printed based on current minimum level.
        
        :param level: the log level to check
        :return: True if should print, False otherwise
        """
        return self.verbose and level >= self.min_level

    def set_min_level(self, level: int):
        """
        Set the minimum log level for printing.
        
        :param level: minimum log level from LogLevel class
        """
        if level < LogLevel.DEBUG or level > LogLevel.SUCCESS:
            raise ValueError("Invalid log level. Must be between LogLevel.DEBUG and LogLevel.SUCCESS")
        self.min_level = level

    def warning(self, content, local_verbose=True):
        """
        Print the warning message.

        Example::

            >>> logger.warning("This is a warning message.")

        :param content: the content of the warning message
        :param local_verbose: whether to print the warning message to the console
        :return:
        """
        if self._should_print(LogLevel.WARNING) and local_verbose:
            beauty_print(content, type="warning")
        self._write_log(content, type="warning")

    def module(self, content, local_verbose=True):
        """
        Print the module message.

        Example::

            >>> logger.module("This is a module message.")

        :param content: the content of the module message
        :param local_verbose: whether to print the module message to the console
        :return:
        """
        if self._should_print(LogLevel.MODULE) and local_verbose:
            beauty_print(content, type="module")
        self._write_log(content, type="module")

    def info(self, content, local_verbose=True):
        """
        Print the info message.

        Example::

            >>> logger.info("This is a info message.")

        :param content: the content of the info message
        :param local_verbose: whether to print the info message to the console
        :return:
        """
        if self._should_print(LogLevel.INFO) and local_verbose:
            beauty_print(content, type="info")
        self._write_log(content, type="info")

    def debug(self, content, local_verbose=True):
        """
        Print the debug message.
        
        :param content: the content of the debug message
        :param local_verbose: whether to print the debug message to the console
        :return:
        """
        if self._should_print(LogLevel.DEBUG) and local_verbose:
            beauty_print(content, type="debug")
        self._write_log(content, type="debug")

    def error(self, content, local_verbose=True):
        """
        Print the error message.
        
        :param content: the content of the error message
        :param local_verbose: whether to print the error message to the console
        :return:
        """
        if self._should_print(LogLevel.ERROR) and local_verbose:
            beauty_print(content, type="error")
        self._write_log(content, type="error")

    def success(self, content, local_verbose=True):
        """
        Print the success message.
        
        :param content: the content of the success message
        :param local_verbose: whether to print the success message to the console
        :return:
        """
        if self._should_print(LogLevel.SUCCESS) and local_verbose:
            beauty_print(content, type="success")
        self._write_log(content, type="success")


def beauty_print(content, type: str = None):
    """
    Print the content with different colors.

    Example::

        >>> from alicia_duo_sdk.utils.logger import beauty_print
        >>> beauty_print("This is a warning message.", type="warning")

    :param content: the content to be printed
    :param type: support "warning", "module", "info", "error", "debug", "success"
    :return:
    """
    if type is None:
        type = "info"
    if type == "warning":
        print("\033[1;37m [Alicia-D-SDK:WARNING] {}\033[0m".format(content))  # For warning (gray)
    elif type == "module":
        print("\033[1;33m [Alicia-D-SDK:MODULE] {}\033[0m".format(content))  # For a new module (light yellow)
    elif type == "info":
        print("\033[1;35m [Alicia-D-SDK:INFO] {}\033[0m".format(content))  # For info (light purple)
    elif type == "debug":
        print("\033[1;34m [Alicia-D-SDK:DEBUG] {}\033[0m".format(content))  # For debug (light blue)
    elif type == "error":
        print("\033[1;31m [Alicia-D-SDK:ERROR] {}\033[0m".format(content))  # For error (red)
    elif type == "success":
        print("\033[1;32m [Alicia-D-SDK:SUCCESS] {}\033[0m".format(content))  # For success (green)
    else:
        raise ValueError("Invalid level")
