import json
import os

CONFIGURATION_DATA_DIR = os.path.join(os.path.dirname(__file__), 'configuration_data/')

def get_config_file(file_name: str) -> str:
    return os.path.join(
        CONFIGURATION_DATA_DIR,
        file_name,
    )

class Configuration(object):
    '''
    A configuration manager to handle json configuration files with a default configuration
    '''
    def __init__(self, file_name: str):
        file_name = get_config_file(file_name)
        if not os.path.exists(file_name):
            raise Exception(f'File not found: {file_name}')
        with open(file_name, 'r') as f:
            self.configuration = json.load(f)

        default_config_file_name = self.configuration['default_schema']
        if default_config_file_name is not None:
            default_config_file_name = get_config_file(default_config_file_name)
            default_config_obj = Configuration(default_config_file_name)
            self.inherit(default_config_obj)
    
    def inherit(self, default_configuration_obj: object = None):
        '''
        Given a configuration object, inhereit the missing fields
        into this configuration
        '''
        default_configuration = default_configuration_obj.configuration
        if default_configuration is not None:
            for key in default_configuration:
                if key not in self.configuration:
                    self.configuration[key] = default_configuration[key]
    
    def get_key(self, key: str):
        '''
        retreive values using slash notation

        i.e. a/b

        will return

        configuration: {
            a: {
                b: VALUE
            }
        }
        '''
        key_tokens = key.split('/')
        res = self.configuration
        for key_tok in key_tokens:
            if len(key_tok) > 0:
                res = res[key_tok]
        return res
    