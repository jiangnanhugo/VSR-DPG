import os

import commentjson as json

from cvdso.utils import safe_merge_dicts


def get_base_config():
    # Load base config
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "config_common.json"), encoding='utf-8') as f:
        base_config = json.load(f)

    # Load task specific config
    task_config_file = "config_regression.json"
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), task_config_file), encoding='utf-8') as f:
        task_config = json.load(f)


    return safe_merge_dicts(base_config, task_config)


def load_config(config=None):
    # Load user config
    if isinstance(config, str):
        with open(config, encoding='utf-8') as f:
            user_config = json.load(f)
    elif isinstance(config, dict):
        user_config = config
    else:
        assert config is None, "Config must be None, str, or dict."
        user_config = {}

    # Load task-specific base config
    base_config = get_base_config()

    # Return combined configs
    return safe_merge_dicts(base_config, user_config)
