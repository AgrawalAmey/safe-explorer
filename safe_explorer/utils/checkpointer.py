import inspect
import os
import pickle

from safe_explorer.core.config import Config
from safe_explorer.utils.hash import get_hash
from safe_explorer.utils.inspect import get_class_that_defined_method
from safe_explorer.utils.path import get_project_root_dir, get_files_in_path


class Checkpointer(object):
    _relative_checkpoint_dir = Config.get_conf().basics.checkpoint_dir
    _checkpoint_dir = f"{get_project_root_dir()}/{_relative_checkpoint_dir}"

    @staticmethod
    def _get_checkpoint_file_name(checkpoint_name, signature):
        return f"{checkpoint_name}_{signature}.p"

    @staticmethod
    def _get_data_signature(*args, **kwargs):
        return get_hash((args, kwargs))

    @staticmethod
    def _get_code_signature(func):
        class_of_func = get_class_that_defined_method(func)
        if class_of_func:
            return hash(inspect.getsource(class_of_func))
        else:
            return hash(inspect.getsource(func))

    @staticmethod
    def _get_signature(data_signature, code_signature):
        return hash((code_signature, data_signature))

    @classmethod
    def _has_checkpoint(cls, checkpoint_name, signature):
        return  cls._get_checkpoint_file_name(checkpoint_name, signature) \
                    in get_files_in_path(cls._checkpoint_dir)
    
    @classmethod
    def _load_checkpoint(cls, checkpoint_name, signature):
        return pickle.load(
                open(cls._get_checkpoint_file_name(checkpoint_name, signature), 'rb'))

    @classmethod
    def _store_checkpoint(cls, data, checkpoint_name, signature):
        pickle.dump(data,
                    open(cls._get_checkpoint_file_name(checkpoint_name, signature), 'wb'))

    @classmethod
    def run_with_checkpointing(cls, func):
        def function_wrapper(*args, **kwargs):
            checkpoint_name = func.__qualname__
            signature = cls._get_signature(cls._get_data_signature(args, kwargs),
                                           cls._get_code_signature(func))
            if cls._has_checkpoint(checkpoint_name, signature):
                return cls._load_checkpoint(checkpoint_name, signature)
            else:
                result = func(args, kwargs)
                cls._store_checkpoint(result, checkpoint_name, signature)
                return result
        return function_wrapper