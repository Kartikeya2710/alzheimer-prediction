import os
import logging
from logging import Formatter
from logging.handlers import RotatingFileHandler
import yaml
from utils.dirs import create_dirs
from utils.dictionary import ConfigDict

def setup_logging(log_dir):
    log_file_format = "[%(levelname)s] - %(asctime)s - %(name)s - : %(message)s in %(pathname)s:%(lineno)d"
    log_console_format = "[%(levelname)s]: %(message)s"

    # Main logger
    main_logger = logging.getLogger()
    main_logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(Formatter(log_console_format))

    exp_file_handler = RotatingFileHandler('{}exp_debug.log'.format(log_dir), maxBytes=10**6, backupCount=5)
    exp_file_handler.setLevel(logging.DEBUG)
    exp_file_handler.setFormatter(Formatter(log_file_format))

    exp_errors_file_handler = RotatingFileHandler('{}exp_error.log'.format(log_dir), maxBytes=10**6, backupCount=5)
    exp_errors_file_handler.setLevel(logging.WARNING)
    exp_errors_file_handler.setFormatter(Formatter(log_file_format))

    main_logger.addHandler(console_handler)
    main_logger.addHandler(exp_file_handler)
    main_logger.addHandler(exp_errors_file_handler)

def get_config_from_yaml(yaml_file) -> ConfigDict:
    """
    Get the config from a yaml file
    :param yaml_file: the path of the config file
    :return: config(dictionary)
    """

    # parse the configurations from the config yaml file provided
    with open(yaml_file, "r") as stream:
        try:
            config = yaml.safe_load(stream)
            converted_dict = ConfigDict(config)
            return converted_dict
        
        except yaml.YAMLError as exc:
            print(exc)

def process_configs(yaml_files: list):
    config = ConfigDict({})
    
    for yaml_file in yaml_files:
        config.update(get_config_from_yaml(yaml_file))
    
    config.checkpoint_dir = os.path.join(
        'experiments', config.model_name + config.dataset_name, 'checkpoints/'
    )
    config.log_dir = os.path.join(
        'experiments', config.model_name + config.dataset_name, 'logs/'
    )
    config.summary_dir = os.path.join(
        'experiments', config.model_name + config.dataset_name, 'summary/'
    )

    create_dirs([config.checkpoint_dir, config.log_dir, config.summary_dir])

    print(f"Checkpoints to be saved at {config.checkpoint_dir}")
    print(f"Logs to be saved at {config.log_dir}")
    print(f"TensorBoard summary to be saved at {config.summary_dir}")

    setup_logging(config.log_dir)

    return config