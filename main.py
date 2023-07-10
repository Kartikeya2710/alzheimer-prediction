import argparse
from utils.config import *
from agents import *

def main():
    arg_parser = argparse.ArgumentParser(description="",
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    arg_parser.add_argument(
        '--model-config',
        default=None,
        help='The model configuration file in yaml format',
        required=True)
    
    arg_parser.add_argument(
        '--dataset-config',
        default=None,
        help='The dataset configuration file in yaml format',
        required=True
    )

    args = arg_parser.parse_args()

    # parse the config json file
    config = process_configs([args.model_config, args.dataset_config])
    print(config.checkpoint_dir)
    

    agent_class = globals()[config.agent]
    agent = agent_class(config)

    agent.run()


if __name__ == '__main__':
    main()