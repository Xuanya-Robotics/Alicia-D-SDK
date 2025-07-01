import os
import pathlib


def create_dir(path, local_verbose=False):
    """
    Create the directory if it does not exist, can create the parent directories as well.

    :param path: the path of the directory
    :param local_verbose: if True, print the message
    :return:
    """
    if not pathlib.Path(path).exists():
        if local_verbose:
            print('{} not exist, created.'.format(path))
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def list_absl_path(dir, recursive=False, prefix=None, suffix=None):
    """
    Get the absolute path of each file in the directory.

    :param dir: directory path
    :param recursive: if True, list the files in the subdirectories as well
    :param prefix: if not None, only list the files with the appointed prefix
    :param suffix: if not None, only list the files with the appointed suffix
    :return: list
    """
    if recursive:
        return [os.path.join(root, file) for root, dirs, files in os.walk(dir) for file in files if
                (suffix is None or file.endswith(suffix)) and (prefix is None or file.startswith(prefix))]
    else:
        return [os.path.join(dir, file) for file in os.listdir(dir) if
                (suffix is None or file.endswith(suffix)) and (prefix is None or file.startswith(prefix))]


def beauty_print(content, type=None):
    """
    Print the content with different colors.

    :param content: the content to be printed
    :param type: support "warning", "module", "info", "error"
    :return:
    """
    if type is None:
        type = "info"
    if type == "warning":
        print("\033[1;37m[Rofunc:WARNING] {}\033[0m".format(content))  # For warning (gray)
    elif type == "module":
        print("\033[1;33m[Rofunc:MODULE] {}\033[0m".format(content))  # For a new module (light yellow)
    elif type == "info":
        print("\033[1;35m[Rofunc:INFO] {}\033[0m".format(content))  # For info (light purple)
    elif type == "error":
        print("\033[1;31m[Rofunc:ERROR] {}\033[0m".format(content))  # For error (red)
    else:
        raise ValueError("Invalid level")
