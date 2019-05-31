import gym
from gym.spaces import Box, Dict
import numpy as np
from numpy import linalg as LA

from safe_explorer.core.config import Config


class Spaceship(gym.Env):
    def __init__(self):

        self._config = Config.get().env.spaceship

        self._width = self._config.length if self._config.is_arena else 1
        self._episode_length = self._config.arena_episode_length \
            if self._config.is_arena else self._config.corridor_episode_length
        # Set the properties for spaces
        self.action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = Dict({
            'agent_position': Box(low=0, high=1, shape=(2,), dtype=np.float32),
            'agent_velocity': Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
            'target_position': Box(low=0, high=1, shape=(2,), dtype=np.float32)
        })

        # Sets all the episode specific variables         
        self.reset()
        
    def reset(self):
        self._velocity = np.zeros(2, dtype=np.float32)
        self._agent_position = \
            (np.asarray([self._width , self._config.length / 3]) - 2 * self._config.margin) * np.random.random(2) \
                 + self._config.margin
        self._target_position = \
            (np.asarray([self._width , self._config.length]) - 2 * self._config.margin) * np.random.random(2) \
                 + self._config.margin
        self._current_time = 0.        

        return self.step(np.zeros(2))[0]

    def _get_reward(self):
        if self._config.enable_reward_shaping and self._is_agent_outside_shaping_boundary():
            reward = -1000
        elif LA.norm(self._agent_position - self._target_position) < self._config.target_radius:
            reward = 1000
        else:
            reward = 0

        return reward
    
    def _move_agent(self, acceleration):
        # Assume spaceship frequency to be one
        self._agent_position += self._velocity * self._config.frequency_ratio \
                                + 0.5 * acceleration * self._config.frequency_ratio ** 2
        self._velocity += self._config.frequency_ratio * acceleration
    
    def _is_agent_outside_boundary(self):
        return np.any(self._agent_position < 0) \
               or np.any(self._agent_position > np.asarray([self._width, self._config.length]))
    
    def _is_agent_outside_shaping_boundary(self):
        return np.any(self._agent_position < self._config.reward_shaping_slack) \
               or np.any(self._agent_position > np.asarray([self._width, self._config.length]) - self._config.reward_shaping_slack)

    def _update_time(self):
        # Assume spaceship frequency to be one
        self._current_time += self._config.frequency_ratio
    
    def _get_noisy_target_position(self):
        return self._target_position + \
               np.random.normal(0, self._config.target_noise_std, 2)

    def get_num_constraints(self):
        return 4

    def get_constraint_values(self):
        # There a a total of 4 constraints
        # a lower and upper bound for each dim
        # We define all the constraints such that C_i = 0
        # _agent_position > 0 + _agent_slack => -_agent_position + _agent_slack < 0
        min_constraints = self._config.agent_slack - self._agent_position
        # _agent_position < np.asarray([_width, length]) - agent_slack
        # => _agent_position + agent_slack - np.asarray([_width, length]) < 0
        max_constraint = self._agent_position  + self._config.agent_slack \
                         - np.asarray([self._width, self._config.length])

        return np.concatenate([min_constraints, max_constraint])

    def step(self, action):
        # Increment time
        self._update_time()

        # Calculate new position of the agent
        self._move_agent(action)

        # Find reward         
        reward = self._get_reward()
        
        # Prepare return payload
        observation = {
            "agent_position": self._agent_position,
            "agent_velocity": self._velocity,
            "target_postion": self._get_noisy_target_position()
        }

        done = self._is_agent_outside_boundary() \
               or int(self._current_time // 1) >= self._episode_length
        
        return observation, reward, done, {}