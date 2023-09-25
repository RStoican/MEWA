import numpy as np
from gym import spaces

from mewa.mewa_symbolic import MEWASymbolic


class MEWASymbolicPrivileged(MEWASymbolic):
    DEFAULT_PRIVILEGE = 3

    def __init__(self,
                 task_path,
                 wide_tasks,
                 narrow_tasks,
                 complex_worker,
                 privilege_level=DEFAULT_PRIVILEGE,

                 seed=None,
                 split_dict=None,
                 tasks=None,
                 max_episode_steps=100,

                 verbose=0,
                 log_path=None):
        assert privilege_level in list(range(1, self.DEFAULT_PRIVILEGE + 1))
        super().__init__(task_path, wide_tasks, narrow_tasks, complex_worker, seed, split_dict, tasks,
                         max_episode_steps, verbose, log_path)
        self.privilege_level = privilege_level
        # FIXME Replace shape(27,) with shape (18+max_worker_probs_len+max_reward_len,)
        self.observation_space = spaces.Box(low=-20, high=20, shape=(27,), dtype=np.double)

    def _get_obs(self):
        obs = super()._get_obs()
        obs = self._add_privilege(obs)
        print(obs)
        print(obs.shape)
        return obs

    def _add_privilege(self, obs):
        worker_personality = [e[0] for e in self._task['worker_personality']]
        # worker_personality = [e for tpl in self._task['worker_personality'] for e in tpl]
        if self.privilege_level == 1:
            return np.append(obs, self._task['task']['worker_task']['rewards'])
        elif self.privilege_level == 2:
            return np.append(obs, worker_personality)
        elif self.privilege_level == 3:
            return np.append(obs, worker_personality + self._task['task']['worker_task']['rewards'])
        raise ValueError(f'The privilege level has an invalid value: {self.privilege_level}')
