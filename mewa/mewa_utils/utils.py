import datetime
import logging
import os

import dateutil.tz
import numpy as np
import yaml

ACTION_LABELS = ["orange", "blue", "green", "red"]


def load_task(task_yaml_path):
    yaml_content = ''
    with open(task_yaml_path, 'r') as f:
        for line in f:
            if not line.strip().startswith('!!python/object'):
                if '!!python/object' in line:
                    line = line.split(' !!python/object')[0] + '\n'
                yaml_content += line
    return yaml.load(yaml_content, Loader=yaml.Loader)


# Load a fixed set of curated tasks (to use during meta-testing). The task description and human behaviours are given
# in a yaml file
def load_curated_tasks(env, curated_task_file):
    curated_tasks_config = load_task(curated_task_file)

    if 'tasks' not in curated_tasks_config.keys():
        raise ValueError('Expected the task file to be a set of curated tasks')

    tasks = []
    for curated_wide_task in curated_tasks_config['tasks']:
        task_path = list(curated_wide_task.keys())[0]
        task_description = load_task(task_yaml_path=task_path)

        narrow_tasks = env.sample_tasks(task_path, wide_count=1, narrow_count=len(curated_wide_task[task_path]))
        for index in range(len(narrow_tasks)):
            human_mean = curated_wide_task[task_path][index]
            human_sds = [human_gauss[1] for human_gauss in narrow_tasks[index]['worker_personality']]

            narrow_tasks[index]['description'] = task_path
            narrow_tasks[index]['worker_personality'] = [(human_mean[i], human_sds[i]) for i in range(len(human_sds))]
            narrow_tasks[index]['task'] = task_description

        tasks += narrow_tasks
    return tasks


def color_to_one_hot(color):
    one_hot = 4 * [0]
    one_hot[ACTION_LABELS.index(color)] = 1
    return np.array(one_hot)


def create_split_dict(args, train_task_count=None, test_task_count=None):
    if 'split' not in args.keys() or args['split'] is None or len(args['split']) <= 0:
        return None
    split = args['split']
    split_train = args['meta_train']['split_train']
    split_eval = args['meta_test']['split_test']
    if train_task_count is None:
        train_task_count = args['meta_train']['env']['wide_tasks'] * args['meta_train']['env']['narrow_tasks']
    if test_task_count is None:
        test_task_count = args['meta_test']['env']['wide_tasks'] * args['meta_test']['env']['narrow_tasks']

    return {
        'split': split,
        'train_count': train_task_count,
        'split_train': split_train,
        'eval_count': test_task_count,
        'split_eval': split_eval
    }


def create_simple_exp_name():
    """
    Create a unique experiment name with a timestamp
    """
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    return timestamp


LOG_DIR = None


def create_logger(log_name, log_file, use_format=False):
    if LOG_DIR is not None:
        log_file = os.path.join(str(LOG_DIR), log_file)
    else:
        raise ValueError('The global variable LOG_DIR should be set before creating a logger. Got: {}'.format(LOG_DIR))

    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)

    ch = logging.FileHandler(log_file, mode='w')
    ch.setLevel(logging.DEBUG)

    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s' if use_format else ''
    formatter = logging.Formatter(log_format)
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    return logger
