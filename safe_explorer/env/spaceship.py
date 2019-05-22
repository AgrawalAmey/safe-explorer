import numpy as np
from numpy import linalg as LA

from safe_explorer.core.config import Config


class Spaceship:
    def __init__(self):

        config = Config.get_conf()

        self.length = config.env.spaceship.length
        self.width = self.length if config.env.spaceship.is_arena else 1
        self.frequency = config.env.spaceship.frequency
        self.margin = config.env.spaceship.margin
        self.agent_slack = config.env.spaceship.agent_slack
        self.episode_length = config.env.spaceship.episode_length
        self.frequency_ratio = config.env.spaceship.frequency_ratio
        self.target_noise_variance = config.env.spaceship.target_noise_variance
        self.target_radius = config.env.spaceship.target_radius
        self.enable_reward_shaping = config.env.spaceship.enable_reward_shaping
        
        # Sets all the episode specific variables         
        self.reset()
        
    def reset(self):
        self.velocity = np.zeros(2)
        self.agent_position = 0.5 * np.ones(2)
        self.agent_position = \
            (np.asarray([self.width , self.length / 3]) - 2 * self.margin) * np.random.random(2) \
                 + self.margin
        self.target_position = \
            (np.asarray([self.width , self.length]) - 2 * self.margin) * np.random.random(2) \
                 + self.margin
        self.current_time = 0.
        self.last_reward = 0.
    
    def _get_reward(self):
        if self.enable_reward_shaping and self._is_agent_outside_slacked_boundary():
            reward = -1000
        elif LA.norm(self.agent_position - self.target_position) < self.target_radius:
            reward = 1000
        else:
            reward = 0

        return reward
    
    def _move_agent(self, acceleration):
        time = self.frequency_ratio * (1 / self.frequency)
        self.agent_position += self.velocity * time + 0.5 * acceleration * time ** 2
        self.velocity += time * acceleration
    
    def _is_agent_outside_boundary(self):
        return np.any(self.agent_position < 0 \
                      or self.agent_position > np.asarray([self.width, self.length]))
    
    def _is_agent_outside_slacked_boundary(self):
        return np.any(self.agent_position < self.agent_slack \
                      or self.agent_position > np.asarray([self.width, self.length]) - self.agent_slack)

    def _update_time(self):
        self.current_time += self.frequency_ratio * (1 / self.frequency)
    
    def _get_noisy_target_position(self):
        return self.target_position + \
               np.random.normal(0, np.power(self.target_noise_variance, 0.5), 2)
    
    def step(self, action):
        # Increment time
        self._update_time()

        # Calculate new position of the agent
        self._move_agent(action)

        # Find reward         
        reward = self._get_reward()
        step_reward = reward - self.last_reward
        self.last_reward = reward
        
        # Prepare return payload
        observation = (self.agent_position, self.velocity, self._get_noisy_target_position())

        done = self._is_agent_outside_boundary() \
               or self.current_time == self.episode_length
        
        return observation, step_reward, done, {}