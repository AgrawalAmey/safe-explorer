import gym
from gym.spaces import Box, Dict
import numpy as np
from numpy import linalg as LA

from safe_explorer.core.config import Config


class BallND(gym.Env):
    def __init__(self):
        self._config = Config.get().env.ballnd
        # Set the properties for spaces
        self.action_space = Box(low=-1, high=1, shape=(self._config.n,), dtype=np.float32)
        self.observation_space = Dict({
            'agent_position': Box(low=0, high=1, shape=(self._config.n,), dtype=np.float32),
            'target_position': Box(low=0, high=1, shape=(self._config.n,), dtype=np.float32)
        })

        # Sets all the episode specific variables
        self.reset()
        
    def reset(self):
        self._agent_position = 0.5 * np.ones(self._config.n,  dtype=np.float32)
        self._reset_target_location()
        self._current_time = 0.
        return self.step(np.zeros(self._config.n))[0]
    
    def _get_reward(self):
        if self._config.enable_reward_shaping and self._is_agent_outside_shaping_boundary():
            return -1
        else:
            return np.clip(1 - 10 * LA.norm(self._agent_position - self._target_position) ** 2, 0, 1)
    
    def _reset_target_location(self):
        self._target_position = \
            (1 - 2 * self._config.target_margin) * np.random.random(self._config.n) + self._config.target_margin
    
    def _move_agent(self, velocity):
        # Assume that frequency of motor is 1 (one action per second)
        self._agent_position += self._config.frequency_ratio * velocity
    
    def _is_agent_outside_boundary(self):
        return np.any(self._agent_position < 0) or np.any(self._agent_position > 1)
    
    def _is_agent_outside_shaping_boundary(self):
        return np.any(self._agent_position < self._config.reward_shaping_slack) \
               or np.any(self._agent_position > 1 - self._config.reward_shaping_slack)

    def _update_time(self):
        # Assume that frequency of motor is 1 (one action per second)
        self._current_time += self._config.frequency_ratio
    
    def _get_noisy_target_position(self):
        return self._target_position + \
               np.random.normal(0, self._config.target_noise_std, self._config.n)
    
    def get_num_constraints(self):
        return 2 * self._config.n

    def get_constraint_values(self):
        # For any given n, there will be 2 * n constraints
        # a lower and upper bound for each dim
        # We define all the constraints such that C_i = 0
        # _agent_position > 0 + _agent_slack => -_agent_position + _agent_slack < 0
        min_constraints = self._config.agent_slack - self._agent_position
        # _agent_position < 1 - _agent_slack => _agent_position + agent_slack- 1 < 0
        max_constraint = self._agent_position  + self._config.agent_slack - 1

        return np.concatenate([min_constraints, max_constraint])

    def step(self, action):
        # Check if the target needs to be relocated
        # Extract the first digit after decimal in current_time to add numerical stability
        if (int(100 * self._current_time) // 10) % (self._config.respawn_interval * 10) == 0:
            self._reset_target_location()

        # Increment time
        self._update_time()

        last_reward = self._get_reward()
        # Calculate new position of the agent
        self._move_agent(action)

        # Find reward         
        reward = self._get_reward()
        step_reward = reward - last_reward

        # Prepare return payload
        observation = {
            "agent_position": self._agent_position,
            "target_postion": self._get_noisy_target_position()
        }

        done = self._is_agent_outside_boundary() \
               or int(self._current_time // 1) > self._config.episode_length

        return observation, step_reward, done, {}
