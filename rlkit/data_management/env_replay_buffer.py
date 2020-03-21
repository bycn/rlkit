from gym.spaces import Discrete

from rlkit.data_management.simple_replay_buffer import SimpleReplayBuffer
from rlkit.envs.env_utils import get_dim
import numpy as np
import torch


class EnvReplayBuffer(SimpleReplayBuffer):
    def __init__(
            self,
            max_replay_buffer_size,
            env,
            env_info_sizes=None
    ):
        """
        :param max_replay_buffer_size:
        :param env:
        """
        self.env = env
        self._ob_space = env.observation_space
        self._action_space = env.action_space

        if env_info_sizes is None:
            if hasattr(env, 'info_sizes'):
                env_info_sizes = env.info_sizes
            else:
                env_info_sizes = dict()

        super().__init__(
            max_replay_buffer_size=max_replay_buffer_size,
            observation_dim=get_dim(self._ob_space),
            action_dim=get_dim(self._action_space),
            env_info_sizes=env_info_sizes
        )

    def add_sample(self, observation, action, reward, terminal,
                   next_observation, **kwargs):
        if isinstance(self._action_space, Discrete):
            new_action = np.zeros(self._action_dim)
            new_action[action] = 1
        else:
            new_action = action
        return super().add_sample(
            observation=observation,
            action=new_action,
            reward=reward,
            next_observation=next_observation,
            terminal=terminal,
            **kwargs
        )

class TransformEnvReplayBuffer(SimpleReplayBuffer):
    def __init__(self,
            max_replay_buffer_size,
            env,
            env_info_sizes=None,
            transform_fn=None,
            #expect tuple (3,128,128)
            input_shape=None,
            output_shape=None,
            img_normalize=True,
            img_zerocenter=False,
            ):
        self.transform_fn = transform_fn
        self.input_shape = input_shape
        
        self.img_normalize = img_normalize
        self.img_zerocenter = img_zerocenter

        self.env = env
        self._ob_space = env.observation_space
        self._action_space = env.action_space

        if env_info_sizes is None:
            if hasattr(env, 'info_sizes'):
                env_info_sizes = env.info_sizes
            else:
                env_info_sizes = dict()
        super().__init__(
            max_replay_buffer_size=max_replay_buffer_size,
            observation_dim=(output_shape and np.prod(output_shape)) or get_dim(self._ob_space),
            action_dim=get_dim(self._action_space),
            env_info_sizes=env_info_sizes
        )

    def add_sample(self, observation, action, reward, terminal,
                   next_observation, **kwargs):

        if isinstance(self._action_space, Discrete):
            new_action = np.zeros(self._action_dim)
            new_action[action] = 1
        else:
            new_action = action
        def transform(observation):
            if self.img_normalize:
                observation = observation / 255.
            if self.img_zerocenter:
                observation = observation * 2 - 1
            if self.transform_fn:
                default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
                observation = torch.tensor(observation, device=default_device).float()
                observation = np.array(self.transform_fn(observation.reshape([1,*self.input_shape])).cpu()).flatten()
            return observation
        observation = transform(observation)
        next_observation = transform(next_observation)
        return super().add_sample(
            observation=observation,
            action=new_action,
            reward=reward,
            next_observation=next_observation,
            terminal=terminal,
            **kwargs
        )
