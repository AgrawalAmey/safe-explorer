import inspect
import os


def get_project_root_dir():
    return f"{get_current_file_path()}/../../"

def get_current_file_path():
    caller_file_path = os.path.abspath(inspect.getfile(inspect.currentframe().f_back))
    return os.path.dirname(caller_file_path)

def get_files_in_path(path):
    return [f for f in os.listdir(path) \
            if os.path.isfile(os.path.join(path, f))]