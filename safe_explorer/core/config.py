import argparse
import copy
import os
import sys
import yaml

from safe_explorer.utils.namespacify import Namespacify
from safe_explorer.utils.path import get_project_root_dir


class Config:
    _config = None

    @classmethod
    def _get_argument_groups(cls, arg_config):
        """Reads config.yml file and creates groups of arguments by flattening hierarchy"""
        argument_groups = []
        for group in arg_config:
            if "properties" in group.keys():
                sub_groups = cls._get_argument_groups(group["properties"])
                for sub_group in sub_groups:
                    if not sub_group.get("_is_root", False):
                        sub_group["name"] = f"{group['name']}_{sub_group['name']}".strip()
                        sub_group["help"] = f"{group.get('help', '')} {sub_group.get('help', '')}".strip()

                    for argument in sub_group.get("arguments", []):
                        argument["name"] = f"{sub_group['name']}_{argument['name']}".strip()
                        del argument["_is_root"]

                if all([x.get("_is_root", False) for x in sub_groups]):
                    argument_groups.append({"name": group["name"],
                                            "help": group.get("help", ""),
                                            "arguments": sub_groups})
                else:
                    argument_groups += sub_groups
            else:
                group["_is_root"] = True
                argument_groups.append(copy.deepcopy(group))

        return argument_groups

    @classmethod
    def _create_parser(cls, name, _help, argument_groups):
        """Creates argument parser from argument groups"""
        parser = argparse.ArgumentParser(prog=name,
                                         description=_help,
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        for argument_group in argument_groups:
            group = parser.add_argument_group(argument_group["name"], argument_group.get('help', ''))
            for argument in argument_group.get("arguments", []):
                default_value = argument.get("default", None)
                options = {
                    "help": argument.get('help', ' '),
                    "default": default_value
                }
                
                if type(default_value) == bool:
                    options["action"] = 'store_true'
                else:
                    options["type"] = type(default_value) if default_value else None
                    if type(default_value) == list:
                        options["nargs"] = '+'

                group.add_argument(f"--{argument['name']}", **options)
        
        return parser
    
    @classmethod
    def _split_namespace(cls, parent_name, arg_config, parsed_dict):
        """Reestablish the hierarchy from parsed arguments"""
        group_namespaces = {}
        for group in arg_config:
            name = group["name"]
            if "properties" in group.keys():
                matching_arguments = {k[(len(name) + 1):]: v for k, v \
                                    in parsed_dict.items() if k.startswith(name)}
                group_namespaces[name] = cls._split_namespace(name, group["properties"], matching_arguments)
            else:
                group_namespaces[name] = parsed_dict[name]

        return Namespacify(parent_name, group_namespaces)

    @classmethod
    def _load_config(cls, args=None):
        """Reads the default config and parses the commandline args to generate config"""
        config_file_path = f"{get_project_root_dir()}/config/defaults.yml"
        config = yaml.load(open(config_file_path), Loader=yaml.FullLoader)
        argument_groups = cls._get_argument_groups(config["arguments"])
        parser = cls._create_parser(config["name"], config.get("help", ''), argument_groups)
        parsed_config = cls._split_namespace(config["name"], config["arguments"], parser.parse_args(args).__dict__)

        cls._config = parsed_config

    @classmethod
    def get(cls):
        if not cls._config:
            cls._load_config()

        return cls._config