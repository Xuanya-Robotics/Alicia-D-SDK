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


class BeautyLogger:
    """
    Lightweight logger for Alicia-D-SDK package.
    """

    def __init__(self, log_dir: str, log_name: str = 'rofunc.log', verbose: bool = True):
        """
        Lightweight logger for Alicia-D-SDK package.

        Example::

            >>> from rofunc.utils.logger import BeautyLogger
            >>> logger = BeautyLogger(log_dir=".", log_name="rofunc.log", verbose=True)

        :param log_dir: the path for saving the log file
        :param log_name: the name of the log file
        :param verbose: whether to print the log to the console
        """
        self.log_dir = log_dir
        self.log_name = log_name
        self.log_path = os.path.join(self.log_dir, self.log_name)
        self.verbose = verbose

        os.makedirs(self.log_dir, exist_ok=True)
        
    def _write_log(self, content, type):
        with open(self.log_path, "a") as f:
            f.write(" Alicia-D-SDK:{}] {}\n".format(type.upper(), content))

    def warning(self, content, local_verbose=True):
        """
        Print the warning message.

        Example::

            >>> logger.warning("This is a warning message.")

        :param content: the content of the warning message
        :param local_verbose: whether to print the warning message to the console
        :return:
        """
        if self.verbose and local_verbose:
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
        if self.verbose and local_verbose:
            beauty_print(content, type="module")
        self._write_log(content, type="module")

    def info(self, content, local_verbose=True):
        """
        Print the module message.

        Example::

            >>> logger.info("This is a info message.")

        :param content: the content of the info message
        :param local_verbose: whether to print the info message to the console
        :return:
        """
        if self.verbose and local_verbose:
            beauty_print(content, type="info")
        self._write_log(content, type="info")


def beauty_print(content, type: str = None):
    """
    Print the content with different colors.

    Example::

        >>> from alicia_d_sdk.utils.logger import beauty_print
        >>> beauty_print("This is a warning message.", type="warning")

    :param content: the content to be printed
    :param type: support "warning", "module", "info", "error"
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
    elif type == "error":
        print("\033[1;31m [Alicia-D-SDK:ERROR] {}\033[0m".format(content))  # For error (red)
    elif type == "success":
        print("\033[1;32m [Alicia-D-SDK:SUCCESS] {}\033[0m".format(content))  # For success (green)
    else:
        raise ValueError("Invalid level")
