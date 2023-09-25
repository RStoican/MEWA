import os

from mewa.mewa_symbolic import MEWASymbolic


class MEWASymbolicSingle(MEWASymbolic):
    DEFAULT_PRIVILEGE = 3

    def __init__(self,
                 task_path,
                 wide_tasks,
                 narrow_tasks,
                 complex_worker,
                 single_task_worker,

                 seed=None,
                 split_dict=None,
                 tasks=None,
                 max_episode_steps=100,

                 verbose=0,
                 log_path=None):
        # FIXME Assert that the single task worker is a list of max_worker_probs_len for this task
        # assert ...

        if os.path.isdir(task_path):
            raise ValueError('For the single-task MEWA, the task_path should point to a single task '
                             '(i.e. a task file, not a task directory)')

        # FIXME DELETE
        print(f'============================================= CREATING NEW ENV OBJECT SEED {seed} =============================================')

        self.single_task_worker = single_task_worker
        super().__init__(task_path,
                         wide_tasks=1,
                         narrow_tasks=1,
                         complex_worker=True,
                         seed=seed,
                         split_dict=None,
                         tasks=None,
                         max_episode_steps=max_episode_steps,
                         verbose=verbose,
                         log_path=log_path)

    def sample_tasks(self, task_path, wide_count, narrow_count):
        tasks = super().sample_tasks(task_path, 1, 1)
        assert len(tasks) == 1
        tasks[-1]['worker_personality'] = self.single_task_worker
        self._update_reward_normaliser(tasks[-1])
        print(tasks[-1]['worker_personality'])
        return tasks
