# FIXME: Cite paper
"""
Given a task description (with some arbitrary transition and reward functions), compute the optimal
transition and reward functions that ensure the task is adaptive, as explained in <CITE_PAPER>
"""

import copy
import json
import os

import numpy as np
import torch
from torch import optim
from torch.nn import functional as F
from tqdm import trange

from mewa.generator.network import Mlp
from mewa.generator.task_evaluator import TaskEvaluator
from mewa.generator.generator_utils import dict_to_safe_json, create_log_dir


LOG_DIR = str(os.path.join('logs', 'task_generator'))


class TaskOptimiser:
    DELTA = 0
    EPSILON = 0
    LAMBDA = -1e-30

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    INPUT = torch.Tensor([1]).to(DEVICE)

    LOG_PATH = None

    def __init__(self,
                 task_path,
                 delta,
                 epsilon,
                 epochs,
                 search,
                 fixed_task_parameters,
                 no_save,
                 verbose):
        self.task_path = task_path
        self.delta = delta
        self.epsilon = epsilon
        self.epochs = epochs
        self.search = search
        self.fixed_task_parameters = fixed_task_parameters
        self.no_save = no_save
        self.verbose = verbose

    def train(self):
        RANGE_FN = trange if self.verbose == 1 else range
        if self.fixed_task_parameters == {}:
            raise ValueError('The fixed task parameters are not set')

        self.DELTA = self.delta * torch.ones((self.fixed_task_parameters['deadlines_count'] - 1,)).to(self.DEVICE)
        self.EPSILON = self.epsilon

        f, optimizer = self._create_network()

        LOG_DICT = {'ep': [], 'w': [], 'r': [], 'loss': []}
        log = copy.deepcopy(LOG_DICT)
        optimal = copy.deepcopy(LOG_DICT)

        min_loss = np.inf
        for ep in RANGE_FN(self.epochs):
            output = f(self.INPUT)
            w, r = torch.split(output, self.fixed_task_parameters['deadlines_count'])
            loss = self._compute_loss(w, r)
            log_results(log, [ep, w, r, loss])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if loss < min_loss and valid_results(w, r):
                min_loss = loss
                optimal = copy.deepcopy(LOG_DICT)
                log_results(optimal, [ep, w, r, loss])

        if len(optimal['r']) > 0:
            # Scale the reward by 10^x, such that r[0]*10^x is the smallest number >= -1
            # Scaling all the rewards by the same weight will not change the optimisation problem,
            # but will help the RL agent learn with SAC
            scale = 1 / np.abs(optimal['r'][0][0])
            scale = np.power(10, int(np.log10(scale)))
            optimal['r'][0] = [scale * reward for reward in optimal['r'][0]]

            return log, optimal
        return log, None

    # Find the optimal task parameters by doing a hyperparameter search during training
    def search_train(self):
        best_log, best_optimal, best_optimal_task_yaml = None, None, None
        best_delta, best_epsilon = None, None

        original_delta, original_epsilon = self.delta, self.epsilon

        min_cv = np.inf
        delta, epsilon, search = self.delta, self.epsilon, self.search
        delta_linspace = np.linspace(delta / 2, delta * 2, num=search)
        for index in trange(len(delta_linspace)):
            new_delta = delta_linspace[index]
            for new_epsilon in np.linspace(epsilon / 2, epsilon * 1.5, num=search):
                self.delta = new_delta
                self.epsilon = new_epsilon

                log, optimal = self.train()
                if optimal is not None:
                    optimal_task_yaml_str = self.create_task_yaml(optimal['w'][0], optimal['r'][0])

                    temp_dir = 'temp_tasks'
                    temp_task_path = os.path.join(temp_dir, 'temp.yaml')
                    try:
                        if os.path.isdir(temp_dir):
                            raise FileExistsError(f'The temp dir {temp_dir} already exists')
                        os.makedirs(temp_dir)
                        with open(temp_task_path, 'w') as f:
                            f.write(optimal_task_yaml_str)

                        task_evaluator = TaskEvaluator(
                            task_path=temp_task_path,
                            repeat=500,
                            avg_worker=0
                        )
                        _, _, avg_return, std_return, cv, _ = task_evaluator.evaluate_task(verbose=0)
                    finally:
                        os.remove(temp_task_path)
                        os.rmdir(temp_dir)

                    task_evaluator.print_results(
                        all_returns=None,
                        policy_params=None,
                        overall_avg_return=avg_return,
                        overall_std_return=std_return,
                        coefficient_of_variation=cv,
                        task_path=None,
                        save_results=False,
                        full_print=False
                    )

                    if cv < min_cv:
                        min_cv = cv
                        best_log, best_optimal, best_optimal_task_yaml = log, optimal, optimal_task_yaml_str
                        best_delta, best_epsilon = new_delta, new_epsilon
                else:
                    print(f'No optimal task found for delta={new_delta}; epsilon={new_epsilon}')
            self.delta, self.epsilon = original_delta, original_epsilon
        return best_delta, best_epsilon, best_log, best_optimal, best_optimal_task_yaml

    # Update the given task yaml with the optimal worker and reward
    def create_task_yaml(self, worker, reward):
        # FIXME Find a better way of doing this using the yaml loader
        yaml_content = ''
        mistake_field, reward_field = False, False
        i = 0
        task_yaml = self.task_path
        with open(task_yaml, 'r') as f:
            for line in f:
                # If the file already has a reward attribute, ignore it
                if line.startswith('  rewards: '):
                    pass
                elif reward_field:
                    yaml_content += line
                    yaml_content += '  rewards: [ '
                    for r in reward:
                        yaml_content += '{}, '.format(r)
                    yaml_content += '0 ]\n'
                    reward_field = False
                elif mistake_field and '- -' in line:
                    yaml_content += line.split('[')[0] + '['
                    yaml_content += f'{worker[i]},'
                    yaml_content += line.split(',')[-1]
                    i += 1
                    if i == len(worker):
                        mistake_field = False
                        reward_field = True
                else:
                    yaml_content += line

                if not mistake_field:
                    mistake_field = 'mistake_gaussians:' in line
        return yaml_content

    def save_results(self, log, optimal, optimal_task_yaml):
        if not self.no_save:
            if not os.path.isdir(LOG_DIR):
                os.makedirs(LOG_DIR, exist_ok=False)

            if not os.path.isdir(self.task_path):
                task_dir_name = self.task_path.split(os.sep)[-2]
            else:
                task_dir_name = self.task_path.split(os.sep)[-1]
            task_log_dir = os.path.join(LOG_DIR, task_dir_name)

            if self.LOG_PATH is None:
                self.LOG_PATH = create_log_dir(exp_prefix='', base_log_dir=task_log_dir)

            self._save_args()
            self._save_log(log, f'{task_dir_name}_log')
            self._save_log(optimal, f'{task_dir_name}_optimal_task')
            self._save_task(optimal_task_yaml)

    def _compute_loss(self, worker, rewards):
        d = []
        for w in worker:
            d.append((1 + (self.fixed_task_parameters['blocks_per_struct'] - 1) * w) / (1 - w))

        deadlines_count = self.fixed_task_parameters['deadlines_count']

        r = torch.empty((deadlines_count, deadlines_count, deadlines_count))
        r[:] = torch.nan
        for j in range(deadlines_count):
            for x in range(deadlines_count):
                for y in range(x + 1):
                    if x <= y + deadlines_count - 2:
                        deadline_return = (d[x] + y) * rewards[j]
                        for k in range(x - y + j + 1, min(x + j + 1, deadlines_count)):
                            deadline_return += -rewards[k]
                        r[j, x, y] = deadline_return

        objective1 = 0
        for j in range(deadlines_count):
            costs = r[j][~r[j].isnan()]
            for index in range(len(costs) - 1):
                objective1 += torch.pow(costs[index] - costs[index + 1], 2)

        q = torch.empty((deadlines_count, deadlines_count, deadlines_count))
        q[:] = torch.nan
        objective2 = 0
        for j in range(1, deadlines_count):
            for y in range(j):
                for x in range(j - y, deadlines_count):
                    q[j, x, y] = d[x] * rewards[y]
                    objective2 += torch.pow(r[j, x, x] - q[j, x, y], 2)

        relu = F.relu
        constraints = \
            torch.sum(relu(-1 * worker)) + \
            torch.sum(relu(worker - 1)) + \
            torch.sum(relu(rewards - self.LAMBDA)) + \
            torch.sum(relu(self.DELTA - worker[:-1] + worker[1:])) + \
            torch.sum(relu(rewards[:-1] - rewards[1:] - self.LAMBDA)) + \
            relu((self.EPSILON - worker[deadlines_count - 1]))

        loss = objective1 + objective2 + constraints
        return loss

    def _create_network(self):
        hidden_size = 512
        f = Mlp(
            input_size=1,
            hidden_sizes=5 * [hidden_size],
            output_size=2 * self.fixed_task_parameters['deadlines_count'],
            hidden_activation=F.relu
        )
        optimizer = optim.Adam(f.parameters(), lr=1e-3)
        return f.to(self.DEVICE), optimizer

    def _save_args(self):
        with open(os.path.join(str(self.LOG_PATH), 'args.log'), 'w') as f:
            f.write(json.dumps({
                'args': dict_to_safe_json(self.__dict__),
                # 'fixed_task_params': self.fixed_task_parameters
            }, indent=2))

    def _save_task(self, yaml_content):
        with open(self.task_path, 'w') as f:
            f.write(yaml_content)

    def _save_log(self, log, file):
        key = list(log.keys())[0]
        total_episodes = len(log[key])

        with open(os.path.join(str(self.LOG_PATH), file), 'w') as f:
            for index in range(total_episodes):
                for key in log.keys():
                    f.write('{}: {}; '.format(key, log[key][index]))
                f.write('\n')


def valid_results(w, r):
    w_valid_probs = torch.all(0 <= w) and torch.all(w <= 1)
    w_descending_order = torch.equal(w, torch.sort(w, descending=True)[0])
    r_negative = torch.all(r < 0)
    r_ascending_order = torch.equal(r, torch.sort(r)[0])

    valid = w_valid_probs and w_descending_order and r_negative and r_ascending_order
    return valid.item() if isinstance(valid, torch.Tensor) else valid


def log_results(log, results):
    for index in range(len(log.keys())):
        key = list(log.keys())[index]
        result = results[index]
        if isinstance(result, torch.Tensor):
            result = result.tolist()
        log[key].append(result)
