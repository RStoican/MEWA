import copy
import os.path
from abc import ABC
from datetime import datetime
from os import listdir
from os.path import isfile, join

import numpy as np
import torch
from gym import spaces

from mewa.mewa_base import MEWA
from mewa.mewa_utils.utils import create_logger, ACTION_LABELS, color_to_one_hot, load_task


def format_action(action):
    if not isinstance(action, torch.Tensor):
        action = torch.Tensor(action)
    dim = len(action.shape)
    if action.shape[dim - 2] == 1:
        action = action.squeeze(dim - 2)
    return action.cpu()


class MEWASymbolic(MEWA, ABC):
    TASK_DIR = '/workspace/tasks/'

    def __init__(self,
                 task_path,
                 wide_tasks,
                 narrow_tasks,
                 complex_worker,
                 seed,

                 test_worker='default',
                 split_dict=None,
                 tasks=None,
                 max_episode_steps=100,
                 verbose=0,
                 log_name=None):
        self.verbose = verbose
        self.complex_worker = complex_worker
        self.test_worker = test_worker
        self.max_episode_steps = max_episode_steps

        # Used for normalising rewards in wide task distributions
        self._reward_normaliser = None

        if log_name is not None:
            now = datetime.now()
            log_file = log_name + '_' + now.strftime('%Y_%m_%d_%H_%M_%S_%f')
            self.logger = create_logger('env', log_file)
        else:
            self.logger = None

        observation_space = spaces.Box(low=0, high=20, shape=(18,), dtype=np.double)
        super(MEWASymbolic, self).__init__(
            input_shape=None,
            task_path=task_path,
            wide_tasks=wide_tasks,
            narrow_tasks=narrow_tasks,
            seed=seed,
            split_dict=split_dict,
            tasks=tasks,
            observation_space=observation_space,
            verbose=verbose)

        # The current state of the env
        self._obs = None

        # The current task
        self._task = None

        self._split_index = 0
        self._step_count = 0

    def reset_task(self, task_id):
        self._print(f'reset_task({task_id})({self.tasks[task_id]["description"]})'
                    f'({self.tasks[task_id]["worker_personality"]})', log=True)
        self._task = self.tasks[task_id]
        self._config_reward = self._task['task']['worker_task']['rewards']
        self._progress = 0
        self._step_count = 0

    def reset(self, seed=None, return_info=False, options=None):
        self._print('reset()', log=True, verbose=2)
        super(MEWASymbolic, self).reset(seed, return_info, options)
        self._obs = self._create_initial_obs()
        self._progress = 0
        self._step_count = 0
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self._take_step(action)

        self._step_count += 1
        if self._step_count >= self.max_episode_steps:
            done = True

        # For wide task distributions, normalise the reward
        if self._reward_normaliser is not None:
            task = self._task['description']
            try:
                normaliser = self._reward_normaliser[task]['max_return'] - self._reward_normaliser[task]['min_return']
            except KeyError:
                min_return, max_return = self._get_min_max_return_normaliser(self._task)
                normaliser = max_return - min_return
            reward /= normaliser
            if done:
                try:
                    reward -= self._reward_normaliser[task]['min_return'] / normaliser
                except KeyError:
                    min_return, max_return = self._get_min_max_return_normaliser(self._task)
                    reward -= min_return / normaliser

        return obs, reward, done, info

    def get_task(self):
        return self._task

    def sample_tasks(self, task_path, wide_count, narrow_count):
        self._split_index = 0

        # Sample the task descriptions
        if os.path.isdir(task_path):
            all_descriptions = [task_file for task_file in listdir(task_path) if isfile(join(task_path, task_file))]
            if wide_count == -1:
                descriptions = all_descriptions
            else:
                # wide_count = min(wide_count, len(all_descriptions))
                descriptions = self._np_random.choice(all_descriptions, size=wide_count, replace=False)
        else:
            if wide_count > 1:
                raise ValueError(f'Trying to build a wide distribution of {wide_count} tasks, but only a single yaml '
                                 f'file was given. Give a directory instead')
            descriptions = ['']

        # Create narrow_count tasks for each task description
        tasks = []
        for description in descriptions:
            # Build the task object. This will be used to generate the worker personality
            description_path = os.path.join(task_path, description) if description != '' else task_path
            task = self._load_task(description_path)

            if self.complex_worker is not None:
                task['worker_task']['complex_worker'] = self.complex_worker

            for _ in range(narrow_count):
                # If we don't split the distribution into multiple regions,
                # then just sample a human from the entire distribution
                if self.split_dict is None or len(self.split_dict) == 0:
                    worker_personality = self._sample_human_simple(task['worker_task'])
                else:
                    # Else, randomly choose one of the sampling methods provided
                    if self._np_random.random() < 0.5:
                        worker_personality = self._sample_human_default(task['worker_task'])
                    else:
                        worker_personality = self._sample_human_balanced(task['worker_task'], use_inc_dist=True)

                tasks.append({"description": description_path, "worker_personality": worker_personality, "task": task})

            # If we are using a wide task distribution, we need a reward normaliser
            self._update_reward_normaliser(tasks[-1])
        return tasks

    # Change the tasks to a list of tasks with uniform workers (starting from the strongest to the weakest worker)
    def create_uniform_workers(self, task_path, worker_count):
        pass

    def _take_step(self, action):
        action = format_action(action)

        color_index = np.where(action == 1)[0][0]
        action_label = ACTION_LABELS[color_index]

        if action_label in self._task['task']['block_colors']:
            next_obs, done = self._do_state_transition(color_index)
            self._print(f'   Turn {self._step_count}: {ACTION_LABELS[color_index]}', log=True, verbose=2)
        else:
            self._print(f'   Turn {self._step_count}: {action_label} (NOT IN TASK)', log=True, verbose=2)
            next_obs = self._get_obs()
            done = False
        return next_obs, self._get_reward(done), done, {}

    def _get_obs(self):
        return copy.deepcopy(self._obs)

    def _update_reward_normaliser(self, task):
        min_return, max_return = self._get_min_max_return_normaliser(task)
        self._reward_normaliser[task['description']] = {
            'min_return': min_return,
            'max_return': max_return
        }

    def _get_min_max_return_normaliser(self, task):
        if self._reward_normaliser is None:
            self._reward_normaliser = {}

        worker_task = task['task']['worker_task']
        rewards = worker_task['rewards']

        # Compute the min and max returns of this task
        min_return = self.max_episode_steps * rewards[0]
        max_return = []
        for index in range(len(rewards)):
            min_steps = worker_task['blocks_per_struct']
            if index == 0:
                min_steps -= 1
            max_return.append(min_steps * rewards[index])
        max_return = np.sum(max_return)
        return min_return, max_return

    def _do_state_transition(self, color_index):
        valid_action = self._do_agent_action(color_index)
        if not valid_action:
            return self._obs, False

        complete_structure_mistake = self._do_struct_action(color_index)

        if complete_structure_mistake is not None:
            self._do_goal_action(color_index, complete_structure_mistake)

        return self._get_obs(), self._is_game_done()

    def _do_agent_action(self, color_index):
        if self._obs[color_index] == 0:
            return False
        self._obs[color_index] -= 1
        return True

    def _do_struct_action(self, color_index):
        struct_index = 4 + 2 * color_index
        self._obs[struct_index] += 1

        # Check if a mistake happened
        if self._is_mistake(struct_index):
            self._print("   Making mistake when working on the {} structure..."
                        .format(ACTION_LABELS[color_index]), log=True, verbose=2)
            self._obs[struct_index + 1] = 1
        else:
            if self._obs[struct_index] == 1:
                self._print("   Starting the {} structure...".format(ACTION_LABELS[color_index]), verbose=2)
            else:
                self._print("   Adding block to the {} structure...".format(ACTION_LABELS[color_index]), verbose=2)

        # If the current action completes the structure, update the state
        complete_structure_mistake = None
        if self._obs[struct_index] == self._task['task']['worker_task']['blocks_per_struct']:
            # Keep track of whether the complete structure has a mistake in it or not
            complete_structure_mistake = (self._obs[struct_index + 1] == 1)

            # Remove the mistake blocks from the play area and return them to the RL agent
            self._obs[struct_index] = 0
            self._obs[struct_index + 1] = 0
            if complete_structure_mistake:
                self._obs[color_index] += self._task['task']['worker_task']['blocks_per_struct']
        return complete_structure_mistake

    def _do_goal_action(self, color_index, complete_structure_mistake):
        if not complete_structure_mistake:
            # Add the complete structure to the goal and build the next stage of the goal, if available
            self._obs[12 + color_index] = 1
            self._build_goal()

    def _is_mistake(self, struct_index):
        # A mistake can happen when a complex worker adds the second block to a structure
        if self.complex_worker and self._obs[struct_index] == 2:
            # The mistake type depends on how many types of colours (i.e. structures) the worker has
            mistake_type = (self._obs[4:12:2] > 0).sum() - 1

            # FIXME: Add the option for Gaussian sampling
            # Get the mistake probability and make sure 0 <= mistake_probability <= 1 holds
            mistake_probability = self._task['worker_personality'][mistake_type][0]
            mistake_probability = min(max(mistake_probability, 0), 1)

            self._print("   Probability of mistake type {}: {}%"
                        .format(mistake_type, round(100 * mistake_probability, 2)), log=True, verbose=2)

            return self._np_random.random() < mistake_probability
        return False

    def _is_game_done(self):
        structures_done = self._obs[12:16].sum() == self._task['task']['supervisor_task']['struct_count']
        goal_done = self._progress == self._task['task']['supervisor_task']['subgoals_count']
        return structures_done and goal_done

    def _build_goal(self):
        task = self._task['task']
        if self._progress >= task['supervisor_task']['subgoals_count']:
            return

        # The requirements are given in the task and depend on the supervisor's current progress
        requirement = task['supervisor_task']['subgoals_requirements'][self._progress][0]
        requirement = color_to_one_hot(requirement)

        req_color_index = np.where(requirement == 1)[0][0]
        req_achieved = self._obs[req_color_index + 12] == 1
        if req_achieved:
            # Use some of the blocks to build the next stage of the goal
            subgoal = task['supervisor_task']['subgoals'][self._progress]
            self._obs[-2] -= len(subgoal)
            self._obs[-1] += len(subgoal)
            self._progress += 1

            # The supervisor has made some progress, so check whether the next requirement hasn't already been achieved
            self._build_goal()

    def _create_initial_obs(self):
        task = self._task['task']

        # The initial state of the RL agent's blocks
        #    obs = [#orange, #blue, #green, #red]
        total_actions = len(ACTION_LABELS)
        agent_obs = total_actions * [0]  # orange, blue, green, red
        for index, color in enumerate(task['block_colors']):
            obs_index = ACTION_LABELS.index(color)
            agent_obs[obs_index] = task['blocks'][index]

        # The initial obs of incomplete structures is an array of 0s, with one element for each colour and one for each
        # mistake
        #    obs = 4 * [#blocks/color, color_mistake]
        struct_obs = 2 * len(ACTION_LABELS) * [0]

        # The initial goal obs will contain a 0 for each possible sub-structure, the initial number of blocks required
        # to complete each sub-goal and a 0 for how many of those blocks have been used
        #    obs = 4 * [complete_substr/color] + [#supervisor blocks, #used supervisor blocks]
        goal_obs = len(ACTION_LABELS) * [0] + [task['supervisor_blocks'], 0]

        return np.concatenate([agent_obs, struct_obs, goal_obs]).astype(np.double)

    def _load_task(self, description_path):
        self._print(f"Loading task {description_path}...", log=True)
        return load_task(description_path)

    """
    The default sampling method. Sample the human "personality" from different regions of the distribution, depending on
    whether this task is used for training or testing
    """
    def _sample_human_default(self, worker_task):
        # Sample the interval that will be used for this human, depending on whether this task will be used for
        # training or testing
        if self._split_index < self.split_dict["train_count"]:
            mean_interval_index, std_interval_index = self._create_valid_distribution(self.split_dict["split_train"])
        else:
            mean_interval_index, std_interval_index = self._create_valid_distribution(self.split_dict["split_eval"])
        self._split_index += 1

        self._print("Creating worker personality:", log=True)
        self._print((mean_interval_index, std_interval_index), log=True)
        personality = []
        for index in range(len(worker_task['mistake_gaussians'])):
            mistake_gauss = worker_task['mistake_gaussians'][index]

            # Split the distribution for this mistake into several equal parts
            mean_intervals = self._split_interval(mistake_gauss[0], self.split_dict["split"][0])
            std_intervals = self._split_interval(mistake_gauss[1], self.split_dict["split"][1])

            # Sample the mean and std
            mean = self._np_random.uniform(mean_intervals[mean_interval_index][0],
                                           mean_intervals[mean_interval_index][1])
            std = self._np_random.uniform(std_intervals[std_interval_index][0],
                                          std_intervals[std_interval_index][1])

            personality.append((mean, std))
            self._print(f"   Mistake Type {index}: {(mean, std)}", log=True)
        return personality

    # Choose the region of the distribution from which to sample the current human behaviour
    def _create_valid_distribution(self, available_intervals):
        mean_index = self._np_random.integers(0, len(available_intervals[0]))
        mean_interval_index = available_intervals[0][mean_index]

        std_index = self._np_random.integers(0, len(available_intervals[1]))
        std_interval_index = available_intervals[1][std_index]

        return mean_interval_index, std_interval_index

    """
    When the testing tasks are sampled from different regions of the distribution than training tasks, it is very 
    likely that the testing distribution is not split into 2 equal distributions (i.e. because the training region 
    is not exactly in the centre of the whole distribution). To ensure we have equal samples from all the regions of 
    the distribution, first sample one of the two regions with equal probability, the sample the task from it
    
    Also, if use_inc_dist, then ensure that adding a new block ALWAYS reduces the probability of making a mistake. 
    If not use_inc_dist, then we can have a case where e.g. the mistake probability of type 0 is sampled from the left 
    side of the distribution and the one for type 1 is sampled from the right side, i.e. adding an additional block 
    greatly INCREASES the mistake probability
    
    The disadvantage of this sampling method is that good humans will generally be overall good, while bad humans will 
    be generally overall bad. We don't recommend using only this method for sampling tasks. See the sample_tasks() 
    method for a combination of multiple sampling approaches
    """
    def _sample_human_balanced(self, worker_task, use_inc_dist):
        self._print("Creating worker personality:", log=True)

        personality = []
        previous_mistake_prob = np.inf
        for index in range(len(worker_task['mistake_gaussians'])):
            # Get the mean and std of the average worker
            mistake_gauss = worker_task['mistake_gaussians'][index]
            avg_worker_mean, avg_worker_std = mistake_gauss[0]
            std_gaussian = mistake_gauss[1]

            # Create the regions of the distribution used for sampling training or testing tasks
            if self._split_index < self.split_dict["train_count"]:
                interval_limits = (
                    (avg_worker_mean - 2 * avg_worker_std, avg_worker_mean - 0.5 * avg_worker_std),
                    (avg_worker_mean + 0.5 * avg_worker_std, avg_worker_mean + 2 * avg_worker_std)
                )
            else:
                interval_limits = (
                    (0, avg_worker_mean - 2 * avg_worker_std),
                    (avg_worker_mean + 2 * avg_worker_std, 1)
                )

            # Sample an interval, then sample the worker personality mean for this mistake type from the interval
            interval = self._create_valid_interval(interval_limits, previous_mistake_prob, use_inc_dist)
            mean = self._np_random.uniform(interval[0], interval[1])
            std = self._np_random.normal(loc=std_gaussian[0], scale=std_gaussian[1])

            self._print(f"   Mistake Type {index}: {(mean, std)}", log=True)
            personality.append((mean, std))

            previous_mistake_prob = mean

        self._split_index += 1
        return personality

    # Create a valid interval from which to sample the current human behaviour
    def _create_valid_interval(self, interval_limits, previous_mistake_prob, use_inc_dist):
        # Depending on the previous mistake probability, create 1 or 2 intervals
        if use_inc_dist:
            # FIXME Use with increased_split
            if previous_mistake_prob > interval_limits[1][0]:
                intervals = (
                    interval_limits[0],
                    (interval_limits[1][0], min(interval_limits[1][1], previous_mistake_prob))
                )
            else:
                intervals = (
                    (interval_limits[0][0], min(interval_limits[0][1], previous_mistake_prob)),
                    None
                )
        else:
            # FIXME Use with no_order_constraints
            intervals = interval_limits

        sampled_interval = intervals[self._np_random.integers(0, 2)]
        if sampled_interval is None:
            sampled_interval = intervals[0]
        if sampled_interval[1] - sampled_interval[0] < 0:
            sampled_interval = (sampled_interval[1], sampled_interval[0])
        return sampled_interval

    # Create a random personality, to be used by a worker. For each possible mistake, generate the mean and std for a
    # specific worker. Mean and std are generated using the Gaussian distributions in the task description
    def _sample_human_simple(self, worker_task):
        self._print("Creating worker personality:", log=True)
        if not self.complex_worker:
            self._print("   Not a complex worker; personality not needed", log=True)
            return None

        # For each possible mistake, generate the mean and std for this specific worker. Mean and std are generated
        # using the Gaussian distributions in the task description
        mistake_gaussians = worker_task['mistake_gaussians']
        personality = []
        for index in range(len(mistake_gaussians)):
            mistake_gaussian = mistake_gaussians[index]

            # The gaussian for sampling the worker's mean and std
            mean_gaussian = mistake_gaussian[0]
            std_gaussian = mistake_gaussian[1]

            mean = self._np_random.normal(loc=mean_gaussian[0], scale=mean_gaussian[1])
            std = self._np_random.normal(loc=std_gaussian[0], scale=std_gaussian[1])

            personality.append((mean, std))
            self._print(f"   Mistake Type {index}: {(mean, std)}", log=True)
        return personality

    def _split_interval(self, gauss, split):
        # Simulate an interval of samples drawn from a Gaussian
        mean, std = gauss
        split_points = np.array([mean - 3 * std, mean - 2 * std, mean - 0.5 * std,
                                 mean + 0.5 * std, mean + 2 * std, mean + 3 * std])
        # split_points = np.linspace(mean - 3 * std, mean + 3 * std, split + 1)
        if split != 5 and split != 1:
            raise ValueError(f'Split should be 5 for mean and 1 for std. Got: {split}')

        # Split the interval into #split intervals of equal length
        intervals = np.zeros((split, 2))
        intervals[:, 0] = split_points[:split]
        intervals[:, 1] = np.roll(split_points, -1)[:split]
        return intervals

    def _print(self, message, log=False, verbose=1):
        if log and self.logger is not None:
            self.logger.info(message)
        if self.verbose >= verbose:
            print(message)


# Make sure the parameters of the given task are valid
def check_task(task):
    # Check agent parameters
    if len(task['blocks']) != len(task['block_colors']):
        raise ValueError("There should be as many types of blocks as there are colors. "
                         "Expected: {}; Got: {}".format(len(task['block_colors']), len(task['blocks'])))
    max_blocks = 5
    for index in range(len(task['blocks'])):
        if task['blocks'][index] < 0 or task['blocks'][index] > max_blocks:
            raise ValueError("Each color must have at least 0 and at most {} blocks. Color {} had: {}"
                             .format(max_blocks, task['block_colors'][index], task['blocks'][index]))
    max_blocks = 8
    if task['supervisor_blocks'] < 0 or task['supervisor_blocks'] > max_blocks:
        raise ValueError("The supervisor must have at least 0 and at most {} blocks. Got: {}"
                         .format(max_blocks, task['supervisor_blocks']))

    # Check structure parameters
    struct_task = task['worker_task']
    if len(struct_task['blocks_position']) != struct_task['blocks_per_struct']:
        raise ValueError("Every block should have a position relative to the first one. Expected: {}; Got: {}"
                         .format(struct_task['blocks_per_struct'], len(struct_task['blocks_position'])))
    if len(struct_task['mistake_positions']) != struct_task['blocks_per_struct']:
        raise ValueError("Every block should have a mistake position relative to the first one. "
                         "Expected: {}; Got: {}"
                         .format(struct_task['blocks_per_struct'], len(struct_task['mistake_positions'])))

    # Check goal parameters
    goal_task = task['supervisor_task']
    if len(goal_task['subgoals_requirements']) != goal_task['subgoals_count']:
        raise ValueError("Every step of the supervisor's goal should have a requirement. Expected: {}; Got: {}"
                         .format(goal_task['subgoals_count'], len(goal_task['subgoals_requirements'])))
    if len(goal_task['subgoals']) != goal_task['subgoals_count']:
        raise ValueError("Wrong number of sub-goals given. "
                         "Expected: {}; Got: {}".format(goal_task['subgoals_count'], len(goal_task['subgoals'])))
    if len(goal_task['worker_struct']) != goal_task['struct_count']:
        raise ValueError("Wrong number of worker structures given. "
                         "Expected: {}; Got: {}".format(goal_task['struct_count'], len(goal_task['worker_struct'])))
