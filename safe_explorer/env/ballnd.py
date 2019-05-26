import gym
from gym.spaces import Box, Dict
import numpy as np
from numpy import linalg as LA

from safe_explorer.core.config import Config


class BallND(gym.Env):
    def __init__(self):
        self._config = Config.get().env.ballnd

        # Set the properties for spaces
        self.action_space = Box(low=-1, high=1, shape=(n,), dtype=np.float32)
        self.observation_space = Dict({
            'agent_position': Box(low=0, high=1, shape=(n,), dtype=np.float32),
            'target_position': Box(low=0, high=1, shape=(n,), dtype=np.float32)
        })

        # Sets all the episode specific variables      
        self.reset()
        
    def reset(self):
        self._agent_position = 0.5 * np.ones(self._config.n,  dtype=np.float32)
        self._reset_target_location()
        self._current_time = 0.
        self._last_reward = 0.
        return self.step(np.zeros(self._config.n))[0]
    
    def _get_reward(self):
        if self._config.enable_reward_shaping and self._is_agent_outside_slacked_boundary():
            return -1
        else:
            return np.absolute(1 - 10 * LA.norm(self._agent_position - self._target_position) ** 2)
    
    def _reset_target_location(self):
        self._target_position = \
            (1 - 2 * self._config.target_margin) * np.random.random(self._config.n) + self._config.target_margin
    
    def _move_agent(self, velocity):
        self._agent_position += self._config.frequency_ratio * (1 / self._config.frequency) * velocity
    
    def _is_agent_outside_boundary(self):
        return np.any(self._agent_position < 0 or self._agent_position > 1)
    
    def _is_agent_outside_slacked_boundary(self):
        return np.any(self._agent_position < self._config.agent_slack \
                      or self._agent_position > 1 - self._config.agent_slack)

    def _update_time(self):
        self._current_time += self._config.frequency_ratio * (1 / self._config.frequency)
    
    def _get_noisy_target_position(self):
        return self._target_position + \
               np.random.normal(0, np.power(self._config.target_noise_variance, 0.5), self._config.n)
    
    def step(self, action):
        # Check if the target needs to be relocated        
        if self._current_time % self._config.respawn_interval == 0:
            self._reset_target_location()

        # Increment time
        self._update_time()

        # Calculate new position of the agent
        self._move_agent(action)

        # Find reward         
        reward = self._get_reward()
        step_reward = reward - self._last_reward
        self._last_reward = reward
        
        # Prepare return payload
        observation = (self._agent_position, self._get_noisy_target_position())

        done = self._is_agent_outside_boundary() \
               or self._current_time == self._config.episode_length
        
        return observation, step_reward, done, {}
