import copy
import os
import random
import sys

import rospkg
import yaml
from sawyer_gazebo_blocks_puzzle.task import Task, WorkerTask, SupervisorTask


# A method for generating a list of integers, where each integer represents the length of a list. Each element has a
# min value of 1. The length of the list is given by list_len. The sum of all elements is at most max_sum if
# exact_sum is False, otherwise it is exactly max_sum
def generate_list_of_lens(list_len, max_sum, exact_sum=False):
    lens = []
    elements_available = max_sum
    for index in range(list_len):
        # How many elements still don't have a value (besides the current one)
        elements_unassigned = list_len - index - 1

        # The max element that can be sampled at this step, such that any future element is guaranteed to have a
        # value of at least 1
        current_max_sample = elements_available - elements_unassigned

        # If the sum is exact and this is the last element, then it gets the remaining value
        if exact_sum and elements_unassigned == 0:
            sample = current_max_sample
        else:
            # No requirement should contain all structures
            if not exact_sum and current_max_sample == max_sum:
                current_max_sample -= 1
            sample = random.randrange(1, current_max_sample + 1)

        lens.append(sample)
        elements_available -= sample
    return lens


# Create a list of lists of elements. Each list will have the length given by lens. Each list will randomly sample
# len . If sample is True, then each element will be randomly sampled. Otherwise, the elements are added in order
def generate_list(elements, lens, sample=True):
    final_list = []
    elements = copy.deepcopy(elements)
    for current_len in lens:
        if sample:
            element_samples = random.sample(elements, current_len)
        else:
            element_samples = elements[:current_len]
        final_list.append(element_samples)

        # The elements used for this list are no longer available
        [elements.remove(element) for element in element_samples]
    return final_list


def delete_existing_tasks(task_dir):
    tasks = [file_name for file_name in os.listdir(task_dir) if "task_" in file_name]
    for task in tasks:
        os.remove(task_dir + task)


class TaskGenerator(object):
    DEFAULT_TARGET_DIR = "random"
    AVAILABLE_COLORS = ["orange", "blue", "green", "red"]

    # The distance between 2 blocks
    APPEND_STEP_SIZE = 0.055
    STACK_STEP_SIZE = APPEND_STEP_SIZE - 0.015

    # The max number of blocks that can be stacked on top of each other
    MAX_STRUCT_BLOCK_HEIGHT = 3

    MIN_COLOR_SAMPLE = 3
    MAX_COLOR_SAMPLE = len(AVAILABLE_COLORS)

    MIN_BLOCK_PER_TYPE = 3
    MAX_BLOCK_PER_TYPE = 5

    MIN_SUPERVISOR_BLOCKS = 4
    MAX_SUPERVISOR_BLOCKS = 8

    MIN_BLOCKS_PER_STRUCT = 3
    MAX_BLOCKS_PER_STRUCT = 4

    MIN_SUPERVISOR_SUBGOAL_COUNT = 1

    def __init__(self, task_dir, target_dir):
        self.root_task_dir = task_dir
        self.target_dir = target_dir if target_dir is not None else self.DEFAULT_TARGET_DIR + "/"
        self.task_dir = self.root_task_dir + self.target_dir

        if not os.path.exists(self.task_dir):
            os.makedirs(self.task_dir)
        else:
            # Delete all existing tasks in this directory
            delete_existing_tasks(self.task_dir)

    # Returns the potential position of a new block, placed next to a randomly sampled block from the given list.
    # The direction in which the new block is appended is also random
    def sample_adjacent_position(self, block_list):
        # Sample a block to append the current block to
        sample_block = random.sample(block_list, 1)[0]

        # Sample the direction on the axis
        direction = random.sample([-self.APPEND_STEP_SIZE, self.APPEND_STEP_SIZE], 1)[0]

        # Sample the axis and get the potential position of the new block in one step
        return random.sample([(sample_block[0] + direction, sample_block[1], 0),
                              (sample_block[0], sample_block[1] + direction, 0)], 1)[0]

    def sample_block_colors(self):
        sample_size = random.randrange(self.MIN_COLOR_SAMPLE, self.MAX_COLOR_SAMPLE + 1)
        return random.sample(self.AVAILABLE_COLORS, sample_size)

    # Get the number of blocks for each colour. Each colour will have the same number of blocks
    def sample_block_count(self, no_of_colors):
        return no_of_colors * [random.randrange(self.MIN_BLOCK_PER_TYPE, self.MAX_BLOCK_PER_TYPE + 1)]

    def sample_supervisor_blocks(self):
        return random.randrange(self.MIN_SUPERVISOR_BLOCKS, self.MAX_SUPERVISOR_BLOCKS + 1)

    # The number of blocks the worker needs to complete a structure
    def sample_blocks_per_struct(self, blocks_per_color):
        # A structure cannot have more blocks than the total available per colour
        actual_max_blocks_per_struct = min(self.MAX_BLOCKS_PER_STRUCT, blocks_per_color)
        return random.randrange(self.MIN_BLOCKS_PER_STRUCT, actual_max_blocks_per_struct + 1)

    def sample_block_positions(self, blocks_per_struct):
        # The position of the first block
        block_positions = [(0, 0, 0)]

        # Keep track of the base of the structure. To be used when appending blocks
        base = [(0, 0, 0)]

        step_types = ["append", "stack"]
        for _ in range(blocks_per_struct - 1):
            # Each block can be either on top of another block (only 1 per level) or adjacent to a base block
            step = random.sample(step_types, 1)[0]

            # TODO: Allow stacking on top of multiple blocks (e.g. for pyramids)
            if step == "stack":
                # If stacking on a block would make the structure too tall, try to resample several times. If the block
                # has still not been added, then change the method to "append"
                resample_times = 5
                resample = 0

                valid_direction = False
                while not valid_direction:
                    # Sample a block and try to stack the new one on top of it
                    sample_block = random.sample(block_positions, 1)[0]
                    potential_position = (sample_block[0], sample_block[1], sample_block[2] + self.STACK_STEP_SIZE)

                    # If a structure is too tall, no more blocks can be stacked on it
                    too_tall = round(potential_position[2] / self.STACK_STEP_SIZE) >= self.MAX_STRUCT_BLOCK_HEIGHT
                    if too_tall:
                        resample += 1
                    if resample == resample_times:
                        step = "append"
                        break

                    if not too_tall and potential_position not in block_positions:
                        block_positions.append(potential_position)
                        valid_direction = True

            if step == "append":
                valid_direction = False
                while not valid_direction:
                    # Sample a potential position for the new block
                    potential_position = self.sample_adjacent_position(base)

                    # If there is already a block at this position, then it is not valid. If it is valid, add it to the
                    # structure (and to the base)
                    if potential_position not in base:
                        block_positions.append(potential_position)
                        base.append(potential_position)
                        valid_direction = True
        return block_positions

    def sample_complex_worker(self):
        return True

    # The number of mistake types is the same as the number of actions (i.e. number of different colours)
    def sample_mistake_gaussians(self, action_space_size):
        # The Gaussians for a complete action space (4 colours)
        complete_gaussians = [[[0.65, 0.12], [0.09, 0.02]],
                              [[0.45, 0.12], [0.09, 0.02]],
                              [[0.2, 0.1], [0.09, 0.02]],
                              [[0, 0.08], [0.09, 0.02]]]

        # For reduced action spaces, only the last action_space_size Gaussians are used. This ensures that having many
        # different colours still gives a low mistake probability, not a relatively high one
        first_gaussian_index = len(complete_gaussians) - action_space_size
        return complete_gaussians[first_gaussian_index:]

    # Sample the positions of the blocks when a mistake has been made. Make sure these positions are not identical to
    # the correct ones
    def sample_mistake_positions(self, blocks_per_struct, correct_positions):
        potential_positions = self.sample_block_positions(blocks_per_struct)
        if potential_positions == correct_positions:
            return self.sample_mistake_positions(blocks_per_struct, correct_positions)
        return potential_positions

    # How many structures the worker needs to give to the supervisor to complete the task. At the moment, this is the
    # same as the number of different colours (so 1 structure per colour)
    def sample_struct_count(self, action_space_size):
        return action_space_size

    # How many steps the supervisor has to take to finish its subgoal. Each step will have a condition (i.e. one or
    # more structures received from the worker). The max number of steps is the number of structures the supervisor
    # will receive
    def sample_subgoal_count(self, struct_count):
        # FIXME: For now, only use one sub-structure per goal. So, subgoal_count = num_of_colours_in_task
        # return random.randrange(self.MIN_SUPERVISOR_SUBGOAL_COUNT, struct_count + 1)
        return struct_count

    # Each subgoal will have a requirement. Each requirement means receiving one or more structures from the worker.
    # The number of structures per requirement is random, but smaller than the total number of structures (having a
    # single requirement with all structures is similar to having no requirement). It is possible that some structures
    # will not be used at all as requirements
    def sample_subgoal_requirements(self, struct_count, requirement_count, colors):
        # Create a list of the number of structures per requirement. Each element is between 1 (a requirement is needed)
        # and the remaining number of unused structures - the number of undefined elements (so that each future
        # requirement will have at least 1 structure)
        req_lens = generate_list_of_lens(list_len=requirement_count, max_sum=struct_count)

        # The first requirements are more likely to have a larger length (due to a larger pool of still available
        # structures). Randomise the order of the list's elements
        random.shuffle(req_lens)

        # Finally, generate a list of lists of colors, using the sampled lengths
        return generate_list(colors, req_lens)

    # Sample the position of each block required to build each of the supervisor's subgoal
    def sample_subgoals(self, supervisor_block_count, subgoal_count):
        # Start by generating a random structure that uses all the supervisor's blocks
        supervisor_structure = self.sample_block_positions(supervisor_block_count)

        # Split the structure into subgoal_count parts
        subgoal_lens = generate_list_of_lens(list_len=subgoal_count, max_sum=supervisor_block_count, exact_sum=True)
        return generate_list(supervisor_structure, subgoal_lens, sample=False)

    def sample_worker_struct(self, struct_count, subgoals, worker_positions):
        if len(subgoals) == 0:
            raise ValueError("Expected the list of subgoals to be non-empty")

        # Get the complete supervisor structure
        supervisor_struct = []
        for subgoal in subgoals:
            supervisor_struct += subgoal

        # For each worker structure, select a random block from the supervisor structure and a direction. If the current
        # worker structure can be appended there, add it to the list. If not, sample another supervisor block
        full_struct = supervisor_struct
        worker_struct = []
        for index in range(struct_count):
            valid_position = False
            while not valid_position:
                # A potential position for the base of this worker structure
                potential_position = self.sample_adjacent_position(full_struct)

                # How this worker structure will be added to the supervisor structure depends on their shapes
                if potential_position not in full_struct:
                    # If this supervisor block is an edge, make sure the entire worker structure can be added without
                    # hitting another supervisor or worker block
                    full_struct_temp = copy.deepcopy(full_struct)
                    for worker_block in worker_positions:
                        # The position of this worker block in the final structure will be its original position plus
                        # the potential position
                        new_position = tuple(
                            [worker_block[axis] + potential_position[axis] for axis in range(len(worker_block))]
                        )

                        # If this new position is invalid, then go back and sample a new block
                        if new_position in full_struct_temp:
                            valid_position = False
                            break

                        # Otherwise, add this worker block to the full structure
                        valid_position = True
                        full_struct_temp.append(new_position)

                    # If the entire worker structure was valid, then add it to the actual list of blocks
                    if valid_position:
                        full_struct = full_struct_temp
                        worker_struct.append(potential_position)
        return worker_struct

    def create_task(self):
        goal = ""
        block_colors = self.sample_block_colors()
        blocks = self.sample_block_count(len(block_colors))
        supervisor_block_color = "purple"
        supervisor_blocks = self.sample_supervisor_blocks()
        worker_task = self.create_worker_task(action_space_size=len(blocks), blocks_per_color=blocks[0])
        supervisor_task = self.create_supervisor_task(
            action_space_size=len(blocks), colors=block_colors, supervisor_block_count=supervisor_blocks,
            worker_positions=worker_task.blocks_position
        )

        return Task(goal=goal, block_colors=block_colors, blocks=blocks, supervisor_block_color=supervisor_block_color,
                    supervisor_blocks=supervisor_blocks, worker_task=worker_task, supervisor_task=supervisor_task)

    def create_worker_task(self, action_space_size, blocks_per_color):
        struct_type = ""
        blocks_per_struct = self.sample_blocks_per_struct(blocks_per_color)
        blocks_position = self.sample_block_positions(blocks_per_struct)
        complex_worker = self.sample_complex_worker()
        mistake_type = ""
        mistake_gaussians = self.sample_mistake_gaussians(action_space_size)
        mistake_positions = self.sample_mistake_positions(blocks_per_struct, blocks_position)

        return WorkerTask(struct_type=struct_type, blocks_per_struct=blocks_per_struct, blocks_position=blocks_position,
                          complex_worker=complex_worker, mistake_type=mistake_type, mistake_gaussians=mistake_gaussians,
                          mistake_positions=mistake_positions)

    def create_supervisor_task(self, action_space_size, colors, supervisor_block_count, worker_positions):
        struct_count = self.sample_struct_count(action_space_size)
        subgoals_count = self.sample_subgoal_count(struct_count)
        subgoals_requirements = self.sample_subgoal_requirements(struct_count, subgoals_count, colors)
        subgoals = self.sample_subgoals(supervisor_block_count, subgoals_count)
        worker_struct = self.sample_worker_struct(struct_count, subgoals, worker_positions)

        return SupervisorTask(struct_count=struct_count, subgoals_count=subgoals_count,
                              subgoals_requirements=subgoals_requirements, subgoals=subgoals,
                              worker_struct=worker_struct)

    def save_task_yaml(self, task, index):
        stream = file(self.task_dir + "/task_" + str(index) + ".yaml", "w")
        yaml.dump(task, stream)
        stream.close()

    def generate_tasks(self, no_of_tasks):
        print("Generating {} random tasks".format(no_of_tasks))
        for index in range(no_of_tasks):
            print("Creating task {}".format(index))
            task = self.create_task()
            self.save_task_yaml(task, index)
        print("Done")


class SmallTaskGenerator(TaskGenerator):
    DEFAULT_TARGET_DIR = "random_small"

    MIN_COLOR_SAMPLE = 2
    MAX_COLOR_SAMPLE = 3

    MIN_BLOCK_PER_TYPE = 2
    MAX_BLOCK_PER_TYPE = 4

    MIN_SUPERVISOR_BLOCKS = 4
    MAX_SUPERVISOR_BLOCKS = 6

    MIN_BLOCKS_PER_STRUCT = 2
    MAX_BLOCKS_PER_STRUCT = 3

    MIN_SUPERVISOR_SUBGOAL_COUNT = 1

    def __init__(self, task_dir, target_dir):
        super(SmallTaskGenerator, self).__init__(task_dir, target_dir)


def main():
    GENERATOR_TYPES = ["default", "small"]

    task_count = 100
    generator_type = GENERATOR_TYPES[0]
    target_dir = None
    if len(sys.argv) > 4:
        raise TypeError("Expected at most 3 arguments: \n"
                        "   - (optional) The number of tasks to generate. Default: 100"
                        "   - (optional) The type of generator to use"
                        "   - (optional) The target directory to save all tasks")
    if len(sys.argv) > 1:
        try:
            task_count = int(sys.argv[1])
        except ValueError as e:
            sys.exit("Expected the number of tasks to be an int. Got: {}".format(sys.argv[1]))
    if len(sys.argv) > 2:
        generator_type = sys.argv[2]
        if generator_type not in GENERATOR_TYPES:
            raise ValueError("Expected the generator type to be one of {}. Got: {}"
                             .format(GENERATOR_TYPES, generator_type))
    if len(sys.argv) > 3:
        target_dir = sys.argv[3]

    task_dir = rospkg.RosPack().get_path("sawyer_gazebo_blocks_puzzle") + "/tasks/"
    if generator_type == "default":
        task_generator = TaskGenerator(task_dir, target_dir)
    else:
        task_generator = SmallTaskGenerator(task_dir, target_dir)
    task_generator.generate_tasks(task_count)


if __name__ == '__main__':
    main()
