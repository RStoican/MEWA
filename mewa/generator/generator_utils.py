import datetime
import os
import os.path as osp

import dateutil.tz
import yaml

# Change this
LOCAL_LOG_DIR = 'output'


# The fixed task parameters are the task parameters given in the task description
def get_task_params(task_yaml):
    yaml_content = ''
    with open(task_yaml, 'r') as f:
        for line in f:
            if not line.strip().startswith('!!python/object'):
                if '!!python/object' in line:
                    line = line.split(' !!python/object')[0] + '\n'
                yaml_content += line

    return yaml.load(yaml_content, Loader=yaml.Loader)


def create_log_dir(exp_prefix, exp_id=None, seed=0, base_log_dir=None):
    """
    Creates and returns a unique log directory.

    :param exp_prefix: name of log directory
    :param exp_id: name of experiment category (e.g. the env)
    :return:
    """
    if base_log_dir is None:
        base_log_dir = LOCAL_LOG_DIR
    exp_name = exp_id
    if exp_name is None:
        exp_name = create_simple_exp_name()
    log_dir = osp.join(base_log_dir, exp_prefix.replace("_", "-"), exp_name)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def create_simple_exp_name():
    """
    Create a unique experiment name with a timestamp
    """
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    return timestamp


def dict_to_safe_json(d):
    """
    Convert each value in the dictionary into a JSON'able primitive.
    :param d:
    :return:
    """
    new_d = {}
    for key, item in d.items():
        if safe_json(item):
            new_d[key] = item
        else:
            if isinstance(item, dict):
                new_d[key] = dict_to_safe_json(item)
            else:
                new_d[key] = str(item)
    return new_d


def safe_json(data):
    if data is None:
        return True
    elif isinstance(data, (bool, int, float)):
        return True
    elif isinstance(data, (tuple, list)):
        return all(safe_json(x) for x in data)
    elif isinstance(data, dict):
        return all(isinstance(k, str) and safe_json(v) for k, v in data.items())
    return False
