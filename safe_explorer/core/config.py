import os
import sys
import yaml

from safe_explorer.utils.namespacify import Namespacify
from safe_explorer.utils.path import get_project_root_dir


class Config:
    _config_file_path = f"{get_project_root_dir()}/config/defaults.yml"
    _config = Namespacify(yaml.load(open(_config_file_path), Loader=yaml.FullLoader))

    @classmethod
    def get_conf(cls):
        return cls._config