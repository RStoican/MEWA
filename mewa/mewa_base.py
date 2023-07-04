import abc
from abc import ABC
from datetime import datetime

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

from mewa.mewa_utils.one_hot_encoding import OneHotEncoding
from mewa.mewa_utils.utils import create_logger


class MEWA(gym.Env, ABC):
    def __init__(self,
                 task_path,
                 wide_tasks,
                 narrow_tasks,
                 complex_worker,
                 seed,
                 split_dict,
                 tasks,
                 verbose,
                 log_name,

                 input_shape=None,
                 observation_space=None,):
        super(MEWA, self).__init__()
        self.complex_worker = complex_worker
        self.split_dict = split_dict

        # Set up the logger
        self.verbose = verbose
        self.logger = None
        if log_name is not None:
            now = datetime.now()
            log_file = log_name + '_' + now.strftime('%Y_%m_%d_%H_%M_%S_%f')
            self.logger = create_logger('env', log_file)

        # Create a seed and set up the RNG
        seed = seed if seed is not None and seed >= 0 else np.random.randint(0, 65536)
        self._print(f'Creating environment with seed {seed}', log=True)
        self._np_random = None
        MEWA.reset(self, seed=seed)

        # Create the state and action spaces
        assert input_shape is not None or observation_space is not None
        self.action_space = OneHotEncoding(4)
        self.observation_space = observation_space
        if self.observation_space is 0:
            self.observation_space = spaces.Box(low=0, high=1, shape=input_shape, dtype=np.double)

        # For each task, the reward function will be given in the task description
        self._reward_function = None

        # Keep track of the sub-goals completed
        self._progress = 0

        # Randomly generate the required number of tasks. If a list of tasks has been given, use that instead
        self.tasks = self.sample_tasks(task_path, wide_tasks, narrow_tasks) if tasks is None else tasks

    @abc.abstractmethod
    def sample_tasks(self, task_path, wide_count, narrow_count):
        pass

    @abc.abstractmethod
    def reset_task(self, task_id):
        pass

    # Change the tasks to a list of tasks with uniform workers (starting from the strongest to the weakest worker)
    @abc.abstractmethod
    def create_uniform_workers(self, task_path, worker_count):
        pass

    def get_all_task_idx(self):
        return list(range(len(self.tasks)))

    def set_tasks(self, tasks):
        self.tasks = tasks

    # Reset the current task to the initial state
    def reset(self, seed=None, return_info=False, options=None):
        # # Initialize the RNG if the seed is manually passed
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)
        if seed is None and self._np_random is None:
            raise EnvironmentError("A seed (int) should be passed to the environment the first time reset() is called")

        _ = None if not return_info else (None, None)
        return _

    def count_risks(self):
        return {}

    def get_progress(self):
        return self._progress

    def _get_reward(self, done):
        if done:
            return self._reward_function[-1]
        return self._reward_function[self._progress]

    # Decode a potentially one-hot-encoded action into a single integer
    def _decode_action(self, action):
        POTENTIAL_ERROR = f'MEWA expects actions to be one-hot-encoded vectors ' \
                          f'or integers representing indexes. Got: {action}'

        from collections.abc import Iterable
        if isinstance(action, Iterable):
            # Decode the action
            zeros = np.count_nonzero(action == 0)
            ones = np.count_nonzero(action == 1)
            if zeros != len(action) - 1 or ones != 1:
                raise TypeError(POTENTIAL_ERROR)
            action_index = np.argmax(action)
        else:
            try:
                action_index = int(action)
            except TypeError as e:
                raise TypeError(POTENTIAL_ERROR) from e
        return action_index

    def _print(self, message, log=False, verbose=1):
        if log and self.logger is not None:
            self.logger.info(message)
        if self.verbose >= verbose:
            print(message)
