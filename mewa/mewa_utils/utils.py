import argparse
import logging
import os

import numpy as np

ACTION_LABELS = ["orange", "blue", "green", "red"]


def color_to_one_hot(color):
    one_hot = 4 * [0]
    one_hot[ACTION_LABELS.index(color)] = 1
    return np.array(one_hot)


def create_split_dict(args, train_task_count=None, test_task_count=None):
    if isinstance(args, argparse.Namespace):
        if train_task_count is None or test_task_count is None:
            raise ValueError('train_task_count or test_task_count are not given')
        if 'split' not in args or args.split is None or len(args.split) <= 0:
            return None
        split = args.split
        split_train = args.split_train
        split_eval = args.split_eval

    elif isinstance(args, dict):
        if 'split' not in args.keys() or args['split'] is None or len(args['split']) <= 0:
            return None
        split = args['split']
        split_train = args['split_train']
        split_eval = args['split_eval']
        if train_task_count is None:
            if 'n_train_tasks' not in args.keys():
                raise ValueError('train_task_count not given')
            train_task_count = args['n_train_tasks']
        if test_task_count is None:
            if 'n_eval_tasks' not in args.keys():
                raise ValueError('n_eval_tasks not given')
            test_task_count = args['n_eval_tasks']

    else:
        raise ValueError('Unknown arguments type: {}'.format(type(args)))

    return {
        'split': split,
        'train_count': train_task_count,
        'split_train': split_train,
        'eval_count': test_task_count,
        'split_eval': split_eval
    }


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
