import os.path

import click
import gym
import numpy as np
import yaml

from mewa.mewa_utils.utils import load_curated_tasks, create_split_dict, load_task, create_simple_exp_name


class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def sample_action(self, o):
        return self.action_space.sample()


def train_placeholder(env, agent):
    pass


# If only the human behaviour varies across MDPs (i.e. narrow distribution), the task path is a single file.
# Else, the task path is a directory of task files
def get_task_path(task_path, config):
    if config['env']['wide_tasks'] == 1:
        return os.path.join(task_path, 'optimal.yaml')
    return os.path.join(task_path, 'tasks')


def create_test_env(task_path, test_config, test_split_dict):
    if test_config['curated_tasks']:
        curated_path = os.path.join(task_path, 'curated/curated_tasks.yaml')
        curated_tasks_config = load_task(curated_path)
        if len(curated_tasks_config['total_tasks']) == 1:
            task_path = os.path.join(task_path, 'optimal.yaml')
        else:
            task_path = os.path.join(task_path, 'tasks')

        # Generate
        env = gym.make('MEWASymbolic-v0', task_path=task_path, split_dict=test_split_dict, **test_config['env'])
        curated_tasks = load_curated_tasks(env, curated_task_file=curated_path)
        env.set_tasks(curated_tasks)
        return env
    else:
        task_path = get_task_path(task_path, test_config)
        return gym.make('MEWASymbolic-v0', task_path=task_path, split_dict=test_split_dict, **test_config['env'])


def meta_train(task_path, config, agent, seed, train_dir):
    train_config = config['meta_train']
    task_path = get_task_path(task_path, train_config)

    train_split_dict = create_split_dict(config, test_task_count=0)
    env = gym.make('MEWASymbolic-v0',
                   task_path=task_path,
                   split_dict=train_split_dict,
                   seed=seed,
                   **train_config['env'])

    task_indices = env.get_all_task_idx()
    for _ in range(train_config['iterations']):
        for _ in range(train_config['task_samples']):
            # Sample a task
            idx = np.random.randint(len(task_indices))
            env.reset_task(idx)

            # Train on the current task
            train_placeholder(env, agent)

    # Save the model
    out_file = os.path.join(OUT_DIR, train_dir, 'models', f'{seed}_model.pth')
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, 'w') as f:
        f.write("Hello world!")


def meta_test(task_path, config, agent):
    test_config = config['meta_test']
    test_split_dict = create_split_dict(config, train_task_count=0)

    env = create_test_env(task_path, test_config, test_split_dict)

    tasks = []
    returns = []
    for _ in range(test_config['iterations']):
        returns.append([])
        for task_id in range(len(env.tasks)):
            tasks.append(env.tasks[task_id])
            returns[-1].append([])

            for _ in range(test_config['adapt_episodes']):
                env.reset_task(task_id)
                o = env.reset()
                next_o = None
                d = False
                ret = 0

                while not d:
                    a = agent.sample_action(o)
                    next_o, r, d, env_info = env.step(a)
                    o = next_o
                    ret += r
                returns[-1][-1].append(ret)
    returns = np.array(returns)   # [num_batches, wide_tasks*narrow_tasks, adapt_episodes]
    return returns, tasks


def save_results(results, train_dir, test_dir):
    out_file = os.path.join(OUT_DIR, train_dir, 'tests', test_dir, 'test.npz')
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, 'wb') as f:
        np.savez(f, **results)


OUT_DIR = os.path.join('results', 'random', )


@click.command()
@click.argument('task', type=str)
@click.argument('config', type=str)
def main(task, config):
    with open(config, 'r') as f:
        config_args = yaml.load(f, Loader=yaml.FullLoader)

    exp_dir = create_simple_exp_name()
    results = {'tasks': [], 'returns': []}

    train_seeds = np.random.choice(65536, replace=False, size=config_args['meta_test']['repeat'])
    for seed in train_seeds:
        seed = int(seed)
        env_placeholder = gym.make('MEWASymbolic-v0', task_path=task, seed=seed)
        agent = RandomAgent(env_placeholder.action_space)

        meta_train(task, config_args, agent, seed, exp_dir)
        returns, tasks = meta_test(task, config_args, agent)
        results['tasks'].append(tasks)
        results['returns'].append(returns)

    results['returns'] = np.array(results['returns'])
    test_dir = create_simple_exp_name()
    save_results(results, exp_dir, test_dir)
    print(np.mean(results['returns'].reshape(-1, config_args['meta_test']['adapt_episodes']), axis=0))


if __name__ == "__main__":
    main()
