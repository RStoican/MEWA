import click

from mewa.generator.task_generator import SymbolicTaskGenerator


# def main2():
#     GENERATOR_TYPES = ["default", "small"]
#
#     task_count = 100
#     generator_type = GENERATOR_TYPES[0]
#     target_dir = None
#     if len(sys.argv) > 4:
#         raise TypeError("Expected at most 3 arguments: \n"
#                         "   - (optional) The number of tasks to generate. Default: 100"
#                         "   - (optional) The type of generator to use"
#                         "   - (optional) The target directory to save all tasks")
#     if len(sys.argv) > 1:
#         try:
#             task_count = int(sys.argv[1])
#         except ValueError as e:
#             sys.exit("Expected the number of tasks to be an int. Got: {}".format(sys.argv[1]))
#     if len(sys.argv) > 2:
#         generator_type = sys.argv[2]
#         if generator_type not in GENERATOR_TYPES:
#             raise ValueError("Expected the generator type to be one of {}. Got: {}"
#                              .format(GENERATOR_TYPES, generator_type))
#     if len(sys.argv) > 3:
#         target_dir = sys.argv[3]
#
#     task_dir = rospkg.RosPack().get_path("sawyer_gazebo_blocks_puzzle") + "/tasks/"
#     if generator_type == "default":
#         task_generator = TaskGenerator(task_dir, target_dir)
#     else:
#         task_generator = SmallTaskGenerator(task_dir, target_dir)
#     task_generator.generate_tasks(task_count)


@click.command()
@click.argument('config', type=str)
@click.argument('out', type=str)
@click.option('--count', type=int, default=1,
              help='the number of tasks to generate. If > 1, then generate a wide distribution')
def main(out, config, count):
    """
    Create one or multiple tasks, with parameters given by the CONFIG yaml file.
    Save the task(s) in the OUT directory
    """
    if count < 1:
        raise ValueError(f'The number of tasks must be a positive number. Got {count}')

    task_generator = SymbolicTaskGenerator(config)
    task_generator.generate_tasks(count)


if __name__ == '__main__':
    main()
