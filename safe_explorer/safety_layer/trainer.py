import numpy as np
import time
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from safe_explorer.core.config import Config
from safe_explorer.core.replay_buffer import ReplayBuffer
from safe_explorer.safety_layer.constraint_model import ConstraintModel
from safe_explorer.utils.list import foreach

class Trainer:
    def __init__(self, env):
        self._env = env

        self._config = Config.get().safety_layer.trainer

        self._num_constraints = env.get_num_constraints()

        self._initialize_constraint_models()

        self._replay_buffer = ReplayBuffer(self._config.replay_buffer_size)

        # Tensorboard writer
        self._writer = SummaryWriter(self._config.tensorboard_dir)

        if self._config.use_gpu:
            self.cuda()

    def _as_tensor(self, ndarray, requires_grad=False):
        tensor = torch.Tensor(ndarray)
        tensor.requires_grad = requires_grad
        return tensor

    def cuda(self):
        foreach(lambda x: x.cuda(), self._models)

    def eval_mode(self):
        foreach(lambda x: x.eval(), self._models)

    def train_mode(self):
        foreach(lambda x: x.train(), self._models)

    def _initialize_constraint_models(self):
        self._models = [ConstraintModel(self._env.observation_space["agent_position"].shape[0],
                                        self._env.action_space.shape[0]) \
                        for _ in range(self._num_constraints)]
        self._optimizers = [Adam(x.parameters(), lr=self._config.lr) for x in self._models]

    def _sample_steps(self, num_steps):
        episode_length = 0

        observation = self._env.reset()

        for step in range(num_steps):            
            action = self._env.action_space.sample()
            c = self._env.get_constraint_values()
            observation_next, _, done, _ = self._env.step(action)
            c_next = self._env.get_constraint_values()

            self._replay_buffer.add({
                "action": action,
                "observation": observation["agent_position"],
                "c": c,
                "c_next": c_next 
            })
            
            observation = observation_next            
            episode_length += 1
            
            if done or (episode_length == self._config.max_episode_length):
                observation = self._env.reset()
                episode_length = 0

    def _evaluate_batch(self, batch):
        observation = self._as_tensor(batch["observation"])
        action = self._as_tensor(batch["action"])
        c = self._as_tensor(batch["c"])
        c_next = self._as_tensor(batch["c_next"])
        
        gs = [x(observation) for x in self._models]

        c_next_predicted = [c[:, i] + \
                            torch.bmm(x.view(x.shape[0], 1, -1), action.view(action.shape[0], -1, 1)).view(-1) \
                            for i, x in enumerate(gs)]
        losses = [torch.mean((c_next[:, i] - c_next_predicted[i]) ** 2) for i in range(self._num_constraints)]
        
        return losses

    def _update_batch(self, batch):
        batch = self._replay_buffer.sample(self._config.batch_size)

        # Update critic
        foreach(lambda x: x.zero_grad(), self._optimizers)
        losses = self._evaluate_batch(batch)
        foreach(lambda x: x.backward(), losses)
        foreach(lambda x: x.step(), self._optimizers)

        return np.asarray([x.item() for x in losses])

    def evaluate(self):

        self._sample_steps(self._config.evaluation_steps)

        self.eval_mode()

        losses = [list(map(lambda x: x.item(), self._evaluate_batch(batch))) for batch in \
                self._replay_buffer.get_sequential(self._config.batch_size)]

        losses = np.mean(np.concatenate(losses).reshape(-1, self._num_constraints), axis=0)

        self._replay_buffer.clear()

        foreach(lambda x: self._writer.add_scalar(f"constraint {x[0]} eval loss", x[1]), enumerate(losses))

        self.train_mode()

        print(f"Validation completed, average loss {losses}")

    def predict(self, observation):
        return self._model(self._as_tensor(observation["agent_position"]).reshape(1, -1))

    def train(self):
        
        start_time = time.time()

        print("==========================================================")
        print("Initializing constraint model training...")
        print("----------------------------------------------------------")        
        print(f"Start time: {start_time}")
        print("==========================================================")


        number_of_steps = self._config.steps_per_epoch * self._config.epochs

        for epoch in range(self._config.epochs):
            self._sample_steps(self._config.steps_per_epoch)

            losses = np.mean(np.concatenate([self._update_batch(batch) for batch in \
                    self._replay_buffer.get_sequential(self._config.batch_size)]).reshape(-1, self._num_constraints), axis=0)

            self._replay_buffer.clear()

            foreach(lambda x: self._writer.add_scalar(f"constraint {x[0]} training loss", x[1]), enumerate(losses))

            print(f"Finished epoch {epoch} with losses: {losses}. Running validation ...")
            self.evaluate()
            print("----------------------------------------------------------")
        
        self._writer.close()
        print("==========================================================")
        print(f"Finished training constraint model. Time spent: {time.time() - start_time}")
        print("==========================================================")