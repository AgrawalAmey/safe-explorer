import gym
import numpy as np
from numpy import linalg as LA

from safe_explorer.core.config import Config


class Spaceship(gym.Env):
    def __init__(self):

        self._config = Config.get().env.spaceship

        self._width = self._config.length if self._config.is_arena else 1
        
        # Set the properties for spaces
        self.action_space = Box(low=-1, high=1, shape=(n,), dtype=np.float32)
        self.observation_space = Dict({
            'agent_position': Box(low=0, high=1, shape=(n,), dtype=np.float32),
            'agent_velocity': Box(low=-np.inf, high=np.inf, shape=(n,), dtype=np.float32),
            'target_position': Box(low=0, high=1, shape=(n,), dtype=np.float32)
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
        self._last_reward = 0.

        return self.step(np.zeros(self._config.n))[0]

    def _get_reward(self):
        if self._config.enable_reward_shaping and self._is_agent_outside_slacked_boundary():
            reward = -1000
        elif LA.norm(self._agent_position - self._target_position) < self._config.target_radius:
            reward = 1000
        else:
            reward = 0

        return reward
    
    def _move_agent(self, acceleration):
        time = self._config.frequency_ratio * (1 / self._config.frequency)
        self._agent_position += self._velocity * time + 0.5 * acceleration * time ** 2
        self._velocity += time * acceleration
    
    def _is_agent_outside_boundary(self):
        return np.any(self._agent_position < 0 \
                      or self._agent_position > np.asarray([self._width, self._config.length]))
    
    def _is_agent_outside_slacked_boundary(self):
        return np.any(self._agent_position < self._config.agent_slack \
                      or self._agent_position > np.asarray([self._width, self._config.length]) - self._config.agent_slack)

    def _update_time(self):
        self._current_time += self._config.frequency_ratio * (1 / self._config.frequency)
    
    def _get_noisy_target_position(self):
        return self._target_position + \
               np.random.normal(0, np.power(self._config.target_noise_variance, 0.5), 2)
    
    def step(self, action):
        # Increment time
        self._update_time()

        # Calculate new position of the agent
        self._move_agent(action)

        # Find reward         
        reward = self._get_reward()
        step_reward = reward - self._last_reward
        self._last_reward = reward
        
        # Prepare return payload
        observation = (self._agent_position, self._velocity, self._get_noisy_target_position())

        done = self._is_agent_outside_boundary() \
               or self._current_time == self._config.episode_length
        
        return observation, step_reward, done, {}