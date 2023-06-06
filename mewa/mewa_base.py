import abc
from abc import ABC

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

from mewa.mewa_utils.one_hot_encoding import OneHotEncoding


class MEWA(gym.Env, ABC):
    # The possible rewards. When the game ends, the agent gets the FINAL_REWARD. Whenever an action leads to the
    # supervisor making progress by completing sub-goals, the agent gets the PROGRESS_REWARD. Otherwise, the agent gets
    # the STEP_REWARD. Using discounted rewards should motivate the agent to get the progress rewards as early as
    # possible
    STEP_REWARD = -0.01
    PROGRESS_REWARD = 0.1
    FINAL_REWARD = 1

    def __init__(self,
                 input_shape,
                 task_path,
                 wide_tasks,
                 narrow_tasks,
                 seed,

                 split_dict=None,
                 tasks=None,
                 observation_space=None,
                 verbose=0):
        super(MEWA, self).__init__()
        self.input_shape = input_shape
        self.split_dict = split_dict
        self.verbose = verbose

        self._np_random = None
        MEWA.reset(self, seed=seed)

        # The reward vector is given in the config file of the task
        self._config_reward = None

        # Keep track of the sub-goals completed
        self._progress = 0

        self.action_space = OneHotEncoding(4)
        self.observation_space = observation_space
        if self.observation_space is 0:
            self.observation_space = spaces.Box(low=0, high=1, shape=input_shape, dtype=np.double)

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
        return range(len(self.tasks))

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
            return self._config_reward[-1]
        return self._config_reward[self._progress]

    # Convert the one-hot-encoded action into a single integer
    def _decode_action(self, action_vector):
        action_vector = action_vector.view(int(np.prod(self.action_space.shape)))
        action_vector = action_vector.to('cpu').detach().numpy()
        zeros = np.count_nonzero(action_vector == 0)
        ones = np.count_nonzero(action_vector == 1)
        if zeros != len(action_vector)-1 or ones != 1:
            raise ValueError("Expected the action to be a one-hot-encoded vector. Got: {}".format(action_vector))
        return np.argmax(action_vector)
