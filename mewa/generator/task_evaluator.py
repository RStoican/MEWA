"""
Given an optimised task, evaluate different types of (hard-coded) policies
"""

import os
import random
from pathlib import Path

import gym
import numpy as np
from tqdm import trange

from mewa.generator.generator_utils import create_log_dir
from mewa.generator.test_policies import TestPolicy
from mewa.mewa_utils.utils import load_task


class TaskEvaluator:
    AVG_WORKER = []

    def __init__(self,
                 task_path,
                 repeat,
                 avg_worker=0):
        self.task_path = task_path
        self.repeat = repeat
        self.avg_worker = avg_worker

    def evaluate_task(self, task_params=None, worker=None, normalise_rewards=False, verbose=1):
        RANGE_FN = trange if verbose >= 1 else range

        if task_params is None:
            task_params = load_task(self.task_path)

        # Create the environment with a single task, then set the task to a specific worker
        # (i.e. usually the average worker)
        if worker is None:
            worker = self._create_worker(task_params)
        env = self._create_env(normalise_rewards=normalise_rewards)
        for index in range(len(env.tasks[0]['worker_personality'])):
            env.tasks[0]['worker_personality'][index] = (worker[index], env.tasks[0]['worker_personality'][index][1])

        # Set the env to the only existing task
        env.reset_task(0)

        all_returns, policy_params = [], []
        total_colours = len(task_params['block_colors'])
        for extra_safe_blocks in [False, True]:
            for risk_type_1 in RANGE_FN(total_colours):
                # A policy which takes no risks of type 1 (i.e. risk_type_1 == 0) is not affected by whether extra safe
                # blocks are used (i.e. extra_safe_blocks == False is the same as extra_safe_blocks == True)
                if not (risk_type_1 == 0 and extra_safe_blocks):
                    for risk_type_2 in range(risk_type_1 + 1):
                        policy = TestPolicy(risk_type_1, risk_type_2, extra_safe_blocks, task_params)

                        # Repeat the test for this policy several time, then use the average return
                        returns = []
                        for _ in range(self.repeat):
                            episode_return = run_policy(policy, env)
                            returns.append(episode_return)

                        avg_return = np.array(returns).mean()
                        all_returns.append(avg_return)
                        policy_params.append({'risk_type_1': risk_type_1, 'risk_type_2': risk_type_2,
                                              'extra_safe_blocks': extra_safe_blocks})

        all_returns = np.array(all_returns)
        overall_avg_return = all_returns.mean()
        overall_std_return = all_returns.std()
        coefficient_of_variation = overall_std_return / abs(overall_avg_return)
        return all_returns, policy_params, overall_avg_return, overall_std_return, coefficient_of_variation, worker

    def print_results(self, all_returns, policy_params, overall_avg_return, overall_std_return,
                      coefficient_of_variation, task_path, save_results=True, worker=None, full_print=True):
        overall_results = f'\nreturn avg: {round(overall_avg_return, 3)}    ' \
                          f'return std: {round(overall_std_return, 3)}    ' \
                          f'coefficient of variation: {100 * round(coefficient_of_variation, 4)}%\n'
        if worker is not None:
            overall_results += f'worker: {worker}\n'
            avg_worker_MAE = [np.abs(worker[index] - self.AVG_WORKER[index]) for index in range(len(worker))]
            avg_worker_MAE = np.mean(avg_worker_MAE)
            overall_results += f'avg worker MAE: {avg_worker_MAE}\n'
        print(overall_results)

        out_file = None
        if save_results:
            tests_out_dir = '/workspace/pearl/env_tests2'
            task_name = Path(task_path).stem
            out_dir = os.path.join(tests_out_dir, task_name)
            out_dir = create_log_dir(exp_prefix='', base_log_dir=out_dir)
            out_file = os.path.join(out_dir, 'test')

            with open(out_file, 'a') as f:
                f.write(overall_results)
                f.write('\n')

        if full_print:
            for index in range(len(all_returns)):
                diff_from_mean = round(all_returns[index] - overall_avg_return, 3)
                diff_from_mean_str = str(diff_from_mean).ljust(6, "0") if diff_from_mean < 0 else str(
                    diff_from_mean).rjust(6)
                return_str = str(round(all_returns[index], 3)).rjust(7).ljust(7, "0")
                print(f'return: {return_str}    '
                      f'diff from mean: {diff_from_mean_str}    '
                      f'risk_1: {policy_params[index]["risk_type_1"]}    '
                      f'risk_2: {policy_params[index]["risk_type_2"]}    '
                      f'extra_safe_blocks: {policy_params[index]["extra_safe_blocks"]}')

                if save_results:
                    with open(out_file, 'a') as f:
                        f.write(f'policy {index}\n')
                        f.write(f'   return: {return_str}\n')
                        f.write('\n')

    def _create_env(self, narrow_tasks=1, split_dict=None, normalise_rewards=False, verbose=0):
        variant = {
            'task_path': self.task_path,
            'wide_tasks': 0 if normalise_rewards else 1,
            'narrow_tasks': narrow_tasks,
            'complex_worker': True,
            'split_dict': split_dict,
            'seed': np.random.randint(0, 65536),
            'verbose': verbose
        }
        return gym.make('MEWASymbolic-v0', **variant)

    def _create_worker(self, task_params):
        self.AVG_WORKER = create_avg_worker(task_params)
        if self.avg_worker == 0:
            return self.AVG_WORKER
        elif self.avg_worker < 0:
            return [random.uniform(0.3, 0.6) * mistake_prob for mistake_prob in self.AVG_WORKER]
        else:
            return [random.uniform(1.3, 1.6) * mistake_prob for mistake_prob in self.AVG_WORKER]


def run_policy(policy, env):
    ret = 0
    path_len = 0

    s = env.reset()
    policy.reset()

    while path_len < 100:
        a = policy.get_action(s, env)
        s, r, d, env_info = env.step(a)

        ret += r
        path_len += 1
        if d:
            break

    return ret


def create_avg_worker(task_params):
    worker = task_params['worker_task']['mistake_gaussians']
    return [w[0][0] for w in worker]
