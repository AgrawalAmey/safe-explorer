import numpy as np
from numpy import linalg as LA

from safe_explorer.core.config import Config


class BallND:
    def __init__(self):
        config = Config.get_conf()

        self.n = config.env.ballnd._n
        self.frequency = config.env.ballnd.frequency
        self.target_margin = config.env.ballnd.target_margin
        self.agent_slack = config.env.ballnd.agent_slack
        self.episode_length = config.env.ballnd.episode_length
        self.respawn_interval = config.env.ballnd.respawn_interval
        self.frequency_ratio = config.env.ballnd.frequency_ratio
        self.target_noise_variance = config.env.ballnd.target_noise_variance
        self.enable_reward_shaping = config.env.ballnd.enable_reward_shaping
        
        # Sets all the episode specific variables         
        self.reset()
        
    def reset(self):
        self.agent_position = 0.5 * np.ones(self.n)
        self._reset_target_location()
        self.current_time = 0.
        self.last_reward = 0.
    
    def _get_reward(self):
        if self.enable_reward_shaping and self._is_agent_outside_slacked_boundary():
            return -1
        else:
            return np.absolute(1 - 10 * LA.norm(self. agent_position - self.target_position) ** 2)
    
    def _reset_target_location(self):
        self.target_position = \
            (1 - 2 * self.target_margin) * np.random.random(self.n) + self.target_margin
    
    def _move_agent(self, velocity):
        self.agent_position += self.frequency_ratio * (1 / self.frequency) * velocity
    
    def _is_agent_outside_boundary(self):
        return np.any(self.agent_position < 0 or self.agent_position > 1)
    
    def _is_agent_outside_slacked_boundary(self):
        return np.any(self.agent_position < self.agent_slack or self.agent_position > 1 - self.agent_slack)

    def _update_time(self):
        self.current_time += self.frequency_ratio * (1 / self.frequency)
    
    def _get_noisy_target_position(self):
        return self.target_position + \
               np.random.normal(0, np.power(self.target_noise_variance, 0.5), self.n)
    
    def step(self, action):
        # Check if the target needs to be relocated        
        if self.current_time % self.respawn_interval == 0:
            self._reset_target_location()

        # Increment time
        self._update_time()

        # Calculate new position of the agent
        self._move_agent(action)

        # Find reward         
        reward = self._get_reward()
        step_reward = reward - self.last_reward
        self.last_reward = reward
        
        # Prepare return payload
        observation = (self.agent_position, self._get_noisy_target_position())

        done = self._is_agent_outside_boundary() \
               or self.current_time == self.episode_length
        
        return observation, step_reward, done, {}