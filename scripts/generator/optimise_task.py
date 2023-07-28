import os

import click
from tqdm import trange

import mewa.mewa_utils.utils as utl
from mewa.generator.task_optimiser import TaskOptimiser


def optimise(args):
    task_dict = utl.load_task(args['task_path'])
    args['fixed_task_parameters'] = {
        'blocks_per_struct': task_dict['worker_task']['blocks_per_struct'],
        'deadlines_count': len(task_dict['supervisor_task']['subgoals_requirements'])
    }

    task_optimiser = TaskOptimiser(**args)
    if args['search'] <= 1:
        log, optimal = task_optimiser.train()
        if optimal is None:
            raise ValueError('Could not find any valid task parameters. Try increasing the number of epochs')
        optimal_task_yaml_str = task_optimiser.create_task_yaml(optimal['w'][0], optimal['r'][0])
    else:
        best_delta, best_epsilon, best_log, best_optimal, best_optimal_task_yaml = task_optimiser.search_train()
        log, optimal, optimal_task_yaml_str = best_log, best_optimal, best_optimal_task_yaml

    task_optimiser.save_results(log, optimal, optimal_task_yaml_str)


@click.command()
@click.argument('task', type=str)
@click.option('--delta', type=float, default=0.15, help='the min difference between w1 and w2')
@click.option('--epsilon', type=float, default=0.15, help='the min value of w2')
@click.option('--epochs', type=int, default=10000)
@click.option('--search', type=int, default=1,
              help='search for the optimal task across different values for delta and epsilon')
@click.option('--no-save', is_flag=True, default=False, help='whether to save the results to a directory')
@click.option('--verbose', type=int, default=1)
def main(task, delta, epsilon, epochs, search, no_save, verbose):
    """ TASK: a task yaml file for a narrow distribution; OR a directory of task yaml files for a wide distribution """

    args = {
        'task_path': task,
        'delta': delta,
        'epsilon': epsilon,
        'epochs': epochs,
        'search': search,
        'no_save': no_save,
        'verbose': verbose
    }
    if not os.path.isdir(task):
        optimise(args)
    else:
        tasks = os.listdir(task)
        for index in trange(len(tasks)):
            task_yaml = tasks[index]
            _, file_extension = os.path.splitext(task_yaml)
            if file_extension != '.yaml':
                raise ValueError(f'Expected each file inside {task} to be YAML. Got: {file_extension}')
            args['task_path'] = os.path.join(task, task_yaml)
            optimise(args)


if __name__ == "__main__":
    main()
