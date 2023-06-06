import numpy as np

from mewa.mewa_utils.utils import color_to_one_hot, ACTION_LABELS


class TestPolicy:
    """
    Critical state 1:
    When the policy has to choose between placing a structure's second block (RISKY) or starting a new structure (SAFE),
    it will take risk_type_1 SAFE actions, then repeat taking the RISKY action until it gets past the critical state.

    Critical state 2:
    When the policy has to choose between completing a structure ("SAFE") or adding the second block to another
    structure ("RISKY"), it will take risk_type_2 RISKY actions, then the SAFE action. We have:
        risk_type_1 <= risk_type_2,
    because we cannot take RISKY actions of type 2 if we don't take enough SAFE actions of type 1

    Extra safe blocks:
    If False, never take safe actions that would require blocks of already completed structures
    """

    def __init__(self, risk_type_1, risk_type_2, use_extra_safe_blocks, task_params):
        self.risk_type_1 = risk_type_1
        self.risk_type_2 = risk_type_2
        self.use_extra_safe_blocks = use_extra_safe_blocks
        self.task_params = task_params

        # Keep track of the number of SAFE type 1 and RISKY type 2 actions taken
        self._safe_count_1 = 0
        self._risky_count_2 = 0

    def reset(self):
        self._safe_count_1 = 0
        self._risky_count_2 = 0

    def get_action(self, s, env):
        progress = env.get_progress()
        deadline, deadline_colour = self._get_deadline(progress)

        if self._is_critical_state_1(s, deadline):
            self._risky_count_2 = 0
            return self._take_critical_action_1(s, progress, deadline_colour)

        if self._is_critical_state_2(s, progress, deadline):
            self._safe_count_1 = 0
            return self._take_critical_action_2(s, progress, deadline_colour)

        # Add a block to the current deadline if
        #   - the current structure has not been started yet; OR
        #   - the current structure and the structures of all other colours have 2 or more blocks
        self.reset()
        return self._take_deadline_action(deadline_colour)

    # This is a critical state of type 1 if we can add the second block to the current deadline
    def _is_critical_state_1(self, s, deadline):
        return s[4 + 2 * deadline] == 1

    # This is a critical state of type 2 if the current deadline has 2 blocks, and we can add the second block to one of
    # the future deadlines
    def _is_critical_state_2(self, s, progress, deadline):
        if s[4 + 2 * deadline] != 2:
            return False

        next_progress = progress + 1
        while next_progress < len(self.task_params['block_colors']):
            next_deadline, _ = self._get_deadline(next_progress)
            next_progress += 1
            if s[4 + 2 * next_deadline] == 1:
                return True
        return False

    def _take_critical_action_1(self, s, progress, deadline_colour):
        # Safe action
        if self._safe_count_1 < self.risk_type_1:
            self._safe_count_1 += 1
            action = self._take_safe_action_1(s, progress)
            if action is not None:
                return action

        # Risky action
        return self._take_deadline_action(deadline_colour)

    def _take_safe_action_1(self, s, progress):
        # Find the first structure from a future deadline to use as a safe block
        a = self._take_new_action(
            s,
            progress_start=progress + 1,
            progress_end=len(self.task_params['block_colors']),
            required_blocks=0
        )
        if a is not None:
            return a

        # If there are no future deadlines to use, try to use colours from completed deadlines as safe actions, but only
        # if the current policy allows it.
        # If there are none, then all structures already have at least 1 block, so there are no safe actions left
        if self.use_extra_safe_blocks:
            return self._take_new_action(
                s,
                progress_start=0,
                progress_end=progress,
                required_blocks=0
            )
        return None

    def _take_critical_action_2(self, s, progress, deadline_colour):
        if self._risky_count_2 < self.risk_type_2:
            # FIXME Maybe we should only increase self._risky_count_2 when the risky action is successful, but that is
            #  more difficult
            self._risky_count_2 += 1
            return self._take_risky_action_2(s, progress)
        return self._take_deadline_action(deadline_colour)

    def _take_risky_action_2(self, s, progress):
        return self._take_new_action(
            s,
            progress_start=progress + 1,
            progress_end=len(self.task_params['block_colors']),
            required_blocks=1
        )

    # Take the action that adds a new block to the current deadline
    def _take_deadline_action(self, deadline_colour):
        return color_to_one_hot(deadline_colour)

    # Take the action that adds a new block to a structure DIFFERENT from the current deadline
    def _take_new_action(self, s, progress_start, progress_end, required_blocks):
        new_progress = progress_start
        while new_progress < progress_end:
            new_deadline, new_deadline_colour = self._get_deadline(new_progress)
            if s[4 + 2 * new_deadline] == required_blocks:
                return color_to_one_hot(new_deadline_colour)
            new_progress += 1
        return None

    def _get_deadline(self, progress):
        subgoals = self.task_params['supervisor_task']['subgoals_requirements']
        if progress == len(subgoals):
            # The last colour required is not specified in the task description, so try to get it
            block_colours = self.task_params['block_colors']
            subgoal_colours = [subgoal[0] for subgoal in subgoals]
            colours_not_mentioned = [colour in subgoal_colours for colour in block_colours]
            deadline_colour = block_colours[colours_not_mentioned.index(False)]
        else:
            deadline_colour = subgoals[progress][0]
        deadline = ACTION_LABELS.index(deadline_colour)
        return deadline, deadline_colour


class SemiRandomPolicy(TestPolicy):
    def __init__(self, task_params):
        super(SemiRandomPolicy, self).__init__(0, 0, None, task_params)

    def _take_critical_action_1(self, s, progress, deadline_colour):
        self.use_extra_safe_blocks = np.random.random() < 0.5
        action = self._take_safe_action_1(s, progress) if np.random.random() < 0.5 \
            else self._take_deadline_action(deadline_colour)

        if action is not None:
            return action
        return self._take_deadline_action(deadline_colour)

    def _take_critical_action_2(self, s, progress, deadline_colour):
        return self._take_risky_action_2(s, progress) if np.random.random() < 0.5\
            else self._take_deadline_action(deadline_colour)
