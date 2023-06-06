import numpy as np
from gym.envs.registration import register

register(
    id='MEWASymbolic-v0',
    entry_point='mewa.mewa_symbolic:MEWASymbolic',
    kwargs={'task_path': 'optimal_tasks/reward-test/2022_12_21_12_18_40', 'wide_tasks': 1, 'narrow_tasks': 50,
            'complex_worker': True, 'seed': np.random.randint(0, 65536)}
)
