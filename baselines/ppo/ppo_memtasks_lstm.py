from collections import defaultdict
import os
import random
import time
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from colorama import Fore, Style

if os.path.exists("wandb_config.yaml"):
    import yaml
    with open("wandb_config.yaml") as f:
        wandb_config = yaml.load(f, Loader=yaml.FullLoader)
    os.environ['WANDB_API_KEY'] = wandb_config['wandb_api']

# ManiSkill specific imports
import mani_skill.envs
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

# Memory-ManiSkill specific imports
from mikasa_robo_suite.memory_envs import *
from mikasa_robo_suite.utils.wrappers import *


import copy
from typing import Dict
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore', message='.*env\\.\\w+ to get variables from other wrappers is deprecated.*')

class FlattenRGBDObservationWrapper(gym.ObservationWrapper):
    """
    Flattens the rgbd mode observations into a dictionary with two keys, "rgbd" and "state"

    Args:
        rgb (bool): Whether to include rgb images in the observation
        depth (bool): Whether to include depth images in the observation
        state (bool): Whether to include state data in the observation
    """

    def __init__(self, env, rgb=True, depth=True, state=True, oracle=False, joints=False) -> None:
        self.base_env: BaseEnv = StateOnlyTensorToDictWrapper(env.unwrapped)
        super().__init__(env)
        self.include_rgb = rgb
        self.include_depth = depth
        self.include_state = state
        self.include_oracle = oracle
        self.include_joints = joints

        sample_obs, _ = env.reset()
        new_obs = self.observation(sample_obs)
        self.base_env.update_obs_space(new_obs)

    def observation(self, observation: Dict):
        # Save oracle_info if it exists
        ret = dict()

        if self.include_rgb or self.include_depth:
            ret['oracle_info'] = observation['oracle_info']
            ret['prompt'] = observation['prompt']
            sensor_data = observation.pop("sensor_data")

            del observation["sensor_param"]
            images = []
            for cam_data in sensor_data.values():
                if self.include_rgb:
                    images.append(cam_data["rgb"])
                if self.include_depth:
                    images.append(cam_data["depth"])

            if len(images) > 0:
                images = torch.concat(images, axis=-1)

        # flatten the rest of the data which should just be state data
        if self.include_state and not (self.include_rgb or self.include_depth):
            if not self.include_oracle:
                observation.pop("oracle_info")
            else:
                observation = observation
        else:
            if not self.include_joints:
                filtered_obs = {k: v for k, v in observation.items() if k not in ['prompt', 'oracle_info']}
            else:
                # Create extra_agent dict with 'extra' and 'agent' keys
                extra_agent = {}
                for key in ['extra', 'agent']:
                    if key in observation:
                        extra_agent[key] = observation.pop(key)

                # Flatten the extra_agent dict
                extra_agent_flat = common.flatten_state_dict(extra_agent, use_torch=True, device=self.base_env.device)
                ret['joints'] = extra_agent_flat

                filtered_obs = {k: v for k, v in observation.items() if k not in ['prompt', 'oracle_info', 'extra']}

            observation = common.flatten_state_dict(
                filtered_obs, use_torch=True, device=self.base_env.device
            )
        
        if self.include_state and not (self.include_rgb or self.include_depth):
            ret = observation
        else:
            ret["state"] = observation
        if self.include_rgb and not self.include_depth:
            ret["rgb"] = images
        elif self.include_rgb and self.include_depth:
            ret["rgbd"] = images
        elif self.include_depth and not self.include_rgb:
            ret["depth"] = images

        if 'state' in ret.keys() and not self.include_state:
            ret.pop('state')

        if 'oracle_info' in ret.keys() and not self.include_oracle and ret['oracle_info'] is not None:
            ret.pop('oracle_info')

        if 'oracle_info' in ret.keys() and (ret['oracle_info'] == 4242424242).any().item():
            ret.pop('oracle_info')

        if 'prompt' in ret.keys() and (ret['prompt'] == 4242424242).any().item():
            ret.pop('prompt')

        if 'joints' in ret.keys() and not self.include_joints:
            ret.pop('joints')

        return ret


@dataclass
class Args:
    exp_name: Optional[str] = None
    """the name of this experiment"""
    seed: int = 123
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "ManiSkill-MemoryBench"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save model into the `checkpoints/ppo_memtasks/runs/{run_name}/{TIME}` folder"""
    evaluate: bool = False
    """if toggled, only runs evaluation with the given model checkpoint and saves the evaluation trajectories"""
    checkpoint: Optional[str] = None
    """path to a pretrained checkpoint file to start evaluation/training from"""
    render_mode: str = "all"
    """the environment rendering mode"""

    # Algorithm specific arguments
    env_id: str = "ShellGamePush-v1"
    """the id of the environment"""
    include_state: bool = False
    """whether to include state information in observations"""
    total_timesteps: int = 50_000_000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1024 # 512 | *256
    """the number of parallel environments"""
    num_eval_envs: int = 16
    """the number of parallel evaluation environments"""
    partial_reset: bool = False # True
    """whether to let parallel environments reset upon termination instead of truncation"""
    eval_partial_reset: bool = False
    """whether to let parallel evaluation environments reset upon termination instead of truncation"""
    num_steps: int = 90
    """the number of steps to run in each environment per policy rollout"""
    num_eval_steps: int = 270
    """the number of steps to run in each evaluation environment during evaluation"""
    reconfiguration_freq: Optional[int] = None
    """how often to reconfigure the environment during training"""
    eval_reconfiguration_freq: Optional[int] = 1
    """for benchmarking purposes we want to reconfigure the eval environment each reset to ensure objects are randomized in some tasks"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99 # ! 0.8 ! 
    """the discount factor gamma"""
    gae_lambda: float = 0.95 # ! 0.9 !
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32 # 32 | *8
    """the number of mini-batches"""
    update_epochs: int = 4 # 4 | *8
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = False # ! False !
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = 0.2
    """the target KL divergence threshold"""
    reward_scale: float = 1.0
    """Scale the reward by this factor"""
    eval_freq: int = 25
    """evaluation frequency in terms of iterations"""
    save_train_video_freq: Optional[int] = None
    """frequency to save training videos in terms of iterations"""
    finite_horizon_gae: bool = True

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


    include_oracle: bool = False
    """if toggled, oracle info (such as cup_with_ball_number in ShellGamePush-v0) will be used during the training, i.e. reducing memory task to MDP"""
    noop_steps: int = 1
    """if = 1, then no noops, if > 1, then noops for t ~ [0, noop_steps-1]"""
    include_rgb: bool = False
    """if toggled, rgb images will be included in the observation space"""
    include_joints: bool = False
    """[works only with include_rgb=True] if toggled, joints will be included in the observation space"""
    reward_mode: str = 'normalized_dense' # sparse | normalized_dense
    """the mode of the reward function"""
    
    lstm_hidden_size: int = 512
    """the hidden size of the LSTM"""
    lstm_num_layers: int = 1
    """the number of layers in the LSTM"""
    lstm_dropout: float = 0.0
    """the dropout rate of the LSTM"""

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def print_tensor_shapes(d, prefix=''):
    for k, v in d.items():
        if isinstance(v, dict):
            print_tensor_shapes(v, prefix=f'{prefix}{k}.')
        elif isinstance(v, torch.Tensor):
            print(f'{prefix}{k}: {v.shape}')


class DictArray(object):
    def __init__(self, buffer_shape, element_space, data_dict=None, device=None):
        self.buffer_shape = buffer_shape
        if data_dict:
            self.data = data_dict
        else:
            assert isinstance(element_space, gym.spaces.dict.Dict)
            self.data = {}
            for k, v in element_space.items():
                if isinstance(v, gym.spaces.dict.Dict):
                    self.data[k] = DictArray(buffer_shape, v)
                else:
                    self.data[k] = torch.zeros(buffer_shape + v.shape).to(device)

    def keys(self):
        return self.data.keys()

    def __getitem__(self, index):
        if isinstance(index, str):
            return self.data[index]
        return {
            k: v[index] for k, v in self.data.items()
        }

    def __setitem__(self, index, value):
        if isinstance(index, str):
            self.data[index] = value
        for k, v in value.items():
            self.data[k][index] = v

    @property
    def shape(self):
        return self.buffer_shape

    def reshape(self, shape):
        t = len(self.buffer_shape)
        new_dict = {}
        for k,v in self.data.items():
            if isinstance(v, DictArray):
                new_dict[k] = v.reshape(shape)
            else:
                new_dict[k] = v.reshape(shape + v.shape[t:])
        new_buffer_shape = next(iter(new_dict.values())).shape[:len(shape)]
        return DictArray(new_buffer_shape, None, data_dict=new_dict)

class NatureCNN(nn.Module):
    def __init__(self, sample_obs):
        """
        oracle_info: dict with keys: "cup_with_ball_number" for ShellGame
        include_oracle: bool, if True, oracle_info will be used during the training, i.e. reducing memory task to MDP
        """
        super().__init__()

        extractors = {}

        self.out_features = 0
        feature_size = 256

        self.list_of_obs_keys = list(sample_obs.keys()) # 'oracle_info', 'prompt', 'state', 'rgb'

        if 'rgb' in self.list_of_obs_keys:
            in_channels = sample_obs["rgb"].shape[-1]
            image_size = (sample_obs["rgb"].shape[1], sample_obs["rgb"].shape[2])

            # here we use a NatureCNN architecture to process images, but any architecture is permissble here
            cnn = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=32,
                    kernel_size=8,
                    stride=4,
                    padding=0,
                ),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0
                ),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0
                ),
                nn.ReLU(),
                nn.Flatten(),
            )

            # to easily figure out the dimensions after flattening, we pass a test tensor
            with torch.no_grad():
                n_flatten = cnn(sample_obs["rgb"].float().permute(0,3,1,2).cpu()).shape[1]
                fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
            extractors["rgb"] = nn.Sequential(cnn, fc)
            self.out_features += feature_size

        for key in self.list_of_obs_keys:
            if key in ['oracle_info', 'prompt']:
                extractors[key] =  nn.Sequential(
                    nn.Linear(sample_obs[key].shape[-1], 64),
                    nn.ReLU()
                )
                self.out_features += 64
            elif key == 'joints':
                extractors[key] =  nn.Sequential(
                    nn.Linear(sample_obs[key].shape[-1], 256),
                    nn.ReLU()
                )
                self.out_features += 256

        print(f'{sample_obs.keys()=}')
        print_tensor_shapes(sample_obs)
        print('\n')

        # for state data we simply pass it through a single linear layer
        if 'state' in sample_obs.keys():
            state_size = sample_obs["state"].shape[-1]
            extractors["state"] = nn.Linear(state_size, 256)
            self.out_features += 256

        self.extractors = nn.ModuleDict(extractors)

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []
        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            obs = observations[key]
            if key == "rgb" and 'rgb' in self.list_of_obs_keys:
                obs = obs.float().permute(0,3,1,2) # (N, H, W, C) -> (N, C, H, W)
                obs = obs / 255.
            elif key in ['oracle_info', 'prompt', 'joints']:
                obs = obs.float()

            encoded_tensor_list.append(extractor(obs))
        return torch.cat(encoded_tensor_list, dim=1)

class Agent(nn.Module):
    def __init__(self, envs, sample_obs):
        super().__init__()
        self.feature_net = NatureCNN(sample_obs=sample_obs)
        # latent_size = np.array(envs.unwrapped.single_observation_space.shape).prod()
        latent_size = self.feature_net.out_features

        self.lstm_hidden_size = args.lstm_hidden_size
        self.lstm = nn.LSTM(latent_size, 
                            self.lstm_hidden_size, 
                            args.lstm_num_layers,
                            dropout=args.lstm_dropout,
                            batch_first=False)
        
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0.0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)

        print('#'*50)
        print(f'{latent_size=}')
        print(f'{self.lstm.hidden_size=}')
        print('#'*50)
        print('\n')


        self.critic = nn.Sequential(
            layer_init(nn.Linear(self.lstm_hidden_size, 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, 1)),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(self.lstm_hidden_size, 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, np.prod(envs.unwrapped.single_action_space.shape)), std=0.01*np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, np.prod(envs.unwrapped.single_action_space.shape)) * -0.5)

    def get_states(self, x, lstm_state, done):
        hidden = self.feature_net(x).unsqueeze(0)
        batch_size = lstm_state[0].shape[1]
        hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))
        done = done.reshape((-1, batch_size))
        new_hidden = []
        for h, d in zip(hidden, done):
            d = d.float()
            h_ = (1.0 - d).view(1, -1, 1) * lstm_state[0]
            c_ = (1.0 - d).view(1, -1, 1) * lstm_state[1]
            h, lstm_state = self.lstm(h.unsqueeze(0), (h_, c_))

            new_hidden += [h]

        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)

        return new_hidden, lstm_state
    
    def get_value(self, x, lstm_state, done):
        hidden, _ = self.get_states(x, lstm_state, done)

        return self.critic(hidden)
    
    def get_action(self, x, lstm_state, done, deterministic=False):
        hidden, lstm_state = self.get_states(x, lstm_state, done)
        action_mean = self.actor_mean(hidden)
        if deterministic:
            return action_mean, lstm_state
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        return probs.sample(), lstm_state
    
    def get_action_and_value(self, x, lstm_state, done, action=None):
        hidden, lstm_state = self.get_states(x, lstm_state, done)
        action_mean = self.actor_mean(hidden)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(hidden), lstm_state
    


class AgentStateOnly(nn.Module):
    def __init__(self, envs):
        super().__init__()

        self.list_of_obs_keys = list(envs.single_observation_space.keys())
        print(f"{self.list_of_obs_keys=}")
        
        length = 0
        for key in self.list_of_obs_keys:
            l_ = np.array(envs.single_observation_space[key].shape).prod()
            print(f'{key}: {l_}')
            length += l_
        
        print(f'Total length: {length}')

        self.critic = nn.Sequential(
            layer_init(nn.Linear(length, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1)),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(length, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, np.prod(envs.single_action_space.shape)), std=0.01*np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, np.prod(envs.single_action_space.shape)) * -0.5)

        print(f'{envs.single_observation_space=}')
    
    def add_prompt_to_state(self, x):
        # Concatenate all observation tensors in order of self.list_of_obs_keys
        tensors = [x[key] for key in self.list_of_obs_keys]
        return torch.cat(tensors, dim=-1)

    def get_value(self, x):
        x = self.add_prompt_to_state(x)
        return self.critic(x)
    
    def get_action(self, x, deterministic=False):
        x = self.add_prompt_to_state(x)
        action_mean = self.actor_mean(x)
        if deterministic:
            return action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.sample()
    def get_action_and_value(self, x, action=None):
        x = self.add_prompt_to_state(x)
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)
    
class Logger:
    def __init__(self, log_wandb=False, tensorboard: SummaryWriter = None) -> None:
        self.writer = tensorboard
        self.log_wandb = log_wandb
    def add_scalar(self, tag, scalar_value, step):
        if self.log_wandb:
            wandb.log({tag: scalar_value}, step=step)
        self.writer.add_scalar(tag, scalar_value, step)
    def close(self):
        self.writer.close()

if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    TIME = time.strftime('%Y%m%d_%H%M%S')

    if args.env_id in ['ShellGamePush-v0', 'ShellGamePick-v0', 'ShellGameTouch-v0']:
        wrappers_list = [
            (InitialZeroActionWrapper, {"n_initial_steps": args.noop_steps-1}),
            (RenderStepInfoWrapper, {}),
            (ShellGameRenderCupInfoWrapper, {}),
            (RenderRewardInfoWrapper, {}),
            (DebugRewardWrapper, {}),
        ]
        oracle_info = 'cup_with_ball_number'
        prompt_info = None
    elif args.env_id in ['InterceptSlow-v0', 'InterceptMedium-v0', 'InterceptFast-v0', 
                         'InterceptGrabSlow-v0', 'InterceptGrabMedium-v0', 'InterceptGrabFast-v0']:
        wrappers_list = [
            (InitialZeroActionWrapper, {"n_initial_steps": args.noop_steps-1}),
            (RenderStepInfoWrapper, {}),
            (RenderRewardInfoWrapper, {}),
            (DebugRewardWrapper, {}),
        ]
        oracle_info = None
        prompt_info = None
    elif args.env_id in ['RotateLenientPos-v0', 'RotateLenientPosNeg-v0',
                         'RotateStrictPos-v0', 'RotateStrictPosNeg-v0']:
        wrappers_list = [
            (InitialZeroActionWrapper, {"n_initial_steps": args.noop_steps-1}),
            (RenderStepInfoWrapper, {}),
            (RenderRewardInfoWrapper, {}),
            (RotateRenderAngleInfoWrapper, {}),
            (DebugRewardWrapper, {}),
        ]
        oracle_info = 'angle_diff'
        prompt_info = 'target_angle'
    elif args.env_id in ['CameraShutdownPush-v0', 'CameraShutdownPick-v0']:
        wrappers_list = [
            (InitialZeroActionWrapper, {"n_initial_steps": args.noop_steps-1}),
            (CameraShutdownWrapper, {"n_initial_steps": 19}), # camera works only for t ~ [0, 19]
            (RenderStepInfoWrapper, {}),
            (RenderRewardInfoWrapper, {}),
        ]
        oracle_info = None
        prompt_info = None
    elif args.env_id in ['TakeItBack-v0']:
        wrappers_list = [
            (InitialZeroActionWrapper, {"n_initial_steps": args.noop_steps-1}),
            (RenderStepInfoWrapper, {}),
            (RenderRewardInfoWrapper, {}),
            (DebugRewardWrapper, {}),
        ]
        oracle_info = None
        prompt_info = None
    elif args.env_id in ['RememberColor3-v0', 'RememberColor5-v0', 'RememberColor9-v0']:
        wrappers_list = [
            (InitialZeroActionWrapper, {"n_initial_steps": args.noop_steps-1}),
            (RememberColorInfoWrapper, {}),
            (RenderStepInfoWrapper, {}),
            (RenderRewardInfoWrapper, {}),
            (DebugRewardWrapper, {}),
        ]
        oracle_info = None
        prompt_info = None
    elif args.env_id in ['RememberShape3-v0', 'RememberShape5-v0', 'RememberShape9-v0']:
        wrappers_list = [
            (InitialZeroActionWrapper, {"n_initial_steps": args.noop_steps-1}),
            (RememberShapeInfoWrapper, {}),
            (RenderStepInfoWrapper, {}),
            (RenderRewardInfoWrapper, {}),
            (DebugRewardWrapper, {}),
        ]
        oracle_info = None
        prompt_info = None
    elif args.env_id in ['RememberShapeAndColor3x2-v0', 'RememberShapeAndColor3x3-v0', 'RememberShapeAndColor5x3-v0']:
        wrappers_list = [
            (InitialZeroActionWrapper, {"n_initial_steps": args.noop_steps-1}),
            (RememberShapeAndColorInfoWrapper, {}),
            (RenderStepInfoWrapper, {}),
            (RenderRewardInfoWrapper, {}),
            (DebugRewardWrapper, {}),
        ]
        oracle_info = None
        prompt_info = None
    elif args.env_id in ['BunchOfColors3-v0', 'BunchOfColors5-v0', 'BunchOfColors7-v0']:
        wrappers_list = [
            (InitialZeroActionWrapper, {"n_initial_steps": args.noop_steps-1}),
            (MemoryCapacityInfoWrapper, {}),
            (RenderStepInfoWrapper, {}),
            (RenderRewardInfoWrapper, {}),
            (DebugRewardWrapper, {}),
        ]
        oracle_info = None
        prompt_info = None
    elif args.env_id in ['SeqOfColors3-v0', 'SeqOfColors5-v0', 'SeqOfColors7-v0']:
        wrappers_list = [
            (InitialZeroActionWrapper, {"n_initial_steps": args.noop_steps-1}),
            (MemoryCapacityInfoWrapper, {}),
            (RenderStepInfoWrapper, {}),
            (RenderRewardInfoWrapper, {}),
            (DebugRewardWrapper, {}),
        ]
        oracle_info = None
        prompt_info = None
    elif args.env_id in ['ChainOfColors3-v0', 'ChainOfColors5-v0', 'ChainOfColors7-v0']:
        wrappers_list = [
            (InitialZeroActionWrapper, {"n_initial_steps": args.noop_steps-1}),
            (MemoryCapacityInfoWrapper, {}),
            (RenderStepInfoWrapper, {}),
            (RenderRewardInfoWrapper, {}),
            (DebugRewardWrapper, {}),
        ]
        oracle_info = None
        prompt_info = None
    else:
        raise ValueError(f"Unknown environment: {args.env_id}")

    print('\n' + '='*75)
    print('║' + ' '*24 + 'Environment Configuration' + ' '*24 + '║')
    print('='*75)
    print('║' + f' Environment ID: {args.env_id}'.ljust(73) + '║')
    print('║' + f' Oracle Info:    {oracle_info}'.ljust(73) + '║')
    print('║ Wrappers:'.ljust(74) + '║')
    for wrapper, kwargs in wrappers_list:
        print('║    ├─ ' + wrapper.__name__.ljust(65) + '║')
        if kwargs:
            print('║    │  └─ ' + str(kwargs).ljust(65) + '║')
    print('║' + '-'*73 + '║')
    
    state_msg = 'state will be used' if args.include_state else 'state will not be used'
    print('║' + f' include_state:       {str(args.include_state):<5} │ {state_msg}'.ljust(68) + '║')
    
    rgb_msg = 'rgb images will be used' if args.include_rgb else 'rgb images will not be used'
    print('║' + f' include_rgb:         {str(args.include_rgb):<5} │ {rgb_msg}'.ljust(68) + '║')
    
    oracle_msg = 'oracle info will be used' if args.include_oracle else 'oracle info will not be used'
    print('║' + f' include_oracle:      {str(args.include_oracle):<5} │ {oracle_msg}'.ljust(68) + '║')
    
    joints_msg = 'joints will be used' if args.include_joints else 'joints will not be used'
    print('║' + f' include_joints:      {str(args.include_joints):<5} │ {joints_msg}'.ljust(68) + '║')
    print('='*75 + '\n')

    assert any([args.include_state, args.include_rgb]), "At least one of include_state or include_rgb must be True."
    assert not (args.include_joints and not args.include_rgb), "include_joints can only be True when include_rgb is True"

    if args.include_state and not args.include_rgb and not args.include_oracle and not args.include_joints:
        MODE = 'state'
    elif args.include_state and args.include_rgb and not args.include_oracle and not args.include_joints:
        raise NotImplementedError("state_rgb is not implemented and does not make sense, since any environment can be solved only by using state")
        MODE = 'state_rgb'
    elif args.include_state and not args.include_rgb and args.include_oracle and not args.include_joints:
        raise NotImplementedError("state_oracle is not implemented and does not make sense, since the state already contains oracle information")
        MODE = 'state_oracle'
    elif args.include_state and args.include_rgb and args.include_oracle and not args.include_joints:
        raise NotImplementedError("state_rgb_oracle is not implemented and does not make sense, since any environment can be solved only by using state")
        MODE = 'state_rgb_oracle'
    elif not args.include_state and args.include_rgb and not args.include_oracle and not args.include_joints:
        MODE = 'rgb'
    elif not args.include_state and args.include_rgb and args.include_oracle and not args.include_joints:
        MODE = 'rgb_oracle'
    elif not args.include_state and args.include_rgb and args.include_joints and args.include_oracle:
        MODE = 'rgb_joints_oracle' # TODO: check if this is correct
    elif not args.include_state and args.include_rgb and args.include_joints and not args.include_oracle:
        MODE = 'rgb_joints'
    else:
        raise NotImplementedError(f"Unknown mode: {args.include_state=} {args.include_rgb=} {args.include_oracle=} {args.include_joints=}")
    
    SAVE_DIR = f'checkpoints/ppo_memtasks/{MODE}/{args.reward_mode}/{args.env_id}'

    if 'state' in MODE:
        raise NotImplementedError("state mode is not implemented with LSTM model, use rgb mode instead")


    print(f'{MODE=}')
    print(f'{prompt_info=}')

    wrappers_list.insert(0, (StateOnlyTensorToDictWrapper, {})) # obs=torch.tensor -> dict with keys: state: obs, prompt: prompt, oracle_info: oracle_info


    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{MODE}__{TIME}"
    else:
        run_name = f"{args.exp_name}__{args.seed}__{MODE}__{TIME}"
        
    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    if MODE not in ['state', 'state_oracle']:
        env_kwargs = dict(obs_mode="rgb", control_mode="pd_joint_delta_pos", render_mode=args.render_mode, sim_backend="gpu", reward_mode=args.reward_mode)
    else:
        env_kwargs = dict(obs_mode="state", control_mode="pd_joint_delta_pos", render_mode=args.render_mode, sim_backend="gpu", reward_mode=args.reward_mode) # render_mode="rgb_array",

    eval_envs = gym.make(args.env_id, num_envs=args.num_eval_envs, reconfiguration_freq=args.eval_reconfiguration_freq,  **env_kwargs) # , reconfigure_freq=args.eval_reconfiguration_freq
    envs = gym.make(args.env_id, num_envs=args.num_envs if not args.evaluate else 1, reconfiguration_freq=args.reconfiguration_freq, **env_kwargs)

    for wrapper_class, wrapper_kwargs in wrappers_list:
        eval_envs = wrapper_class(eval_envs, **wrapper_kwargs)
        envs = wrapper_class(envs, **wrapper_kwargs)


    # rgbd obs mode returns a dict of data, we flatten it so there is just a rgbd key and state key
    envs = FlattenRGBDObservationWrapper(envs, rgb=args.include_rgb, depth=False, state=args.include_state, 
                                         oracle=args.include_oracle, joints=args.include_joints)
    eval_envs = FlattenRGBDObservationWrapper(eval_envs, rgb=args.include_rgb, depth=False, state=args.include_state, 
                                              oracle=args.include_oracle, joints=args.include_joints)

    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)
    if args.capture_video:
        eval_output_dir = f"{SAVE_DIR}/{run_name}/{TIME}/videos"
        if args.evaluate:
            eval_output_dir = f"{os.path.dirname(args.checkpoint)}/test_videos"
        print(f"Saving eval videos to {eval_output_dir}")
        if args.save_train_video_freq is not None:
            save_video_trigger = lambda x : (x // args.num_steps) % args.save_train_video_freq == 0
            envs = RecordEpisode(envs, output_dir=f"{SAVE_DIR}/{run_name}/{TIME}/train_videos", save_trajectory=False, save_video_trigger=save_video_trigger, max_steps_per_video=args.num_steps, video_fps=30)
        eval_envs = RecordEpisode(eval_envs, output_dir=eval_output_dir, save_trajectory=args.evaluate, trajectory_name="trajectory", max_steps_per_video=args.num_eval_steps, video_fps=30)
    envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=not args.partial_reset, record_metrics=True)
    eval_envs = ManiSkillVectorEnv(eval_envs, args.num_eval_envs, ignore_terminations=not args.eval_partial_reset, record_metrics=True)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_episode_steps = gym_utils.find_max_episode_steps_value(envs._env)
    print('='*70)
    print(f"Max Episode Steps: {max_episode_steps}")
    print('='*70 + '\n')
    logger = None
    if not args.evaluate:
        print("Running training")
        if args.track:
            import wandb
            config = vars(args)
            config["env_cfg"] = dict(**env_kwargs, num_envs=args.num_envs, env_id=args.env_id, env_horizon=max_episode_steps, partial_reset=args.partial_reset)
            config["eval_env_cfg"] = dict(**env_kwargs, num_envs=args.num_eval_envs, env_id=args.env_id, env_horizon=max_episode_steps, partial_reset=args.partial_reset)
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=False,
                config=config,
                name=run_name,
                save_code=True,
                group="PPO",
                tags=["ppo", "walltime_efficient"]
            )
        writer = SummaryWriter(f"{SAVE_DIR}/{run_name}/{TIME}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        logger = Logger(log_wandb=args.track, tensorboard=writer)
    else:
        print("Running evaluation")

    # ALGO Logic: Storage setup
    obs = DictArray((args.num_steps, args.num_envs), envs.single_observation_space, device=device)        
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    eval_obs, _ = eval_envs.reset(seed=args.seed)
    next_done = torch.zeros(args.num_envs, device=device)
    next_done_eval = torch.zeros(args.num_eval_envs, device=device)
    eps_returns = torch.zeros(args.num_envs, dtype=torch.float, device=device)
    video_iteration = 0

    print(f"\n####")
    print(f"args.num_iterations={args.num_iterations} args.num_envs={args.num_envs} args.num_eval_envs={args.num_eval_envs}")
    print(f"args.minibatch_size={args.minibatch_size} args.batch_size={args.batch_size} args.update_epochs={args.update_epochs}")
    print(f"####\n")

    if MODE not in ['state', 'state_oracle']:
        agent = Agent(envs, sample_obs=next_obs).to(device)
    else:
        agent = AgentStateOnly(envs).to(device)

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    if args.checkpoint:
        agent.load_state_dict(torch.load(args.checkpoint))

    initial_lstm_state = (
        torch.zeros(agent.lstm.num_layers, args.num_envs, agent.lstm.hidden_size).to(device),
        torch.zeros(agent.lstm.num_layers, args.num_envs, agent.lstm.hidden_size).to(device),
    )  # hidden and cell states (see https://youtu.be/8HyCNIVRbSU)

    initial_lstm_state_val = (
        torch.zeros(agent.lstm.num_layers, args.num_eval_envs, agent.lstm.hidden_size).to(device),
        torch.zeros(agent.lstm.num_layers, args.num_eval_envs, agent.lstm.hidden_size).to(device),
    )  # hidden and cell states (see https://youtu.be/8HyCNIVRbSU)

    # for iteration in range(1, args.num_iterations + 1):
    for iteration in tqdm(range(1, args.num_iterations + 1), total=args.num_iterations, desc="Training"):
        print(f"Epoch: {iteration}, global_step={global_step}")
        next_lstm_state = (initial_lstm_state[0].clone(), initial_lstm_state[1].clone())
        final_values = torch.zeros((args.num_steps, args.num_envs), device=device)
        agent.eval()
        if iteration % args.eval_freq == 1:
            print("Evaluating")
            next_lstm_state_val = (initial_lstm_state_val[0].clone(), initial_lstm_state_val[1].clone())
            eval_obs, _ = eval_envs.reset()
            eval_metrics = defaultdict(list)
            num_episodes = 0
            for _ in range(args.num_eval_steps):
                with torch.no_grad():
                    act_eval, next_lstm_state_val = agent.get_action(eval_obs, next_lstm_state_val, next_done_eval, deterministic=True)
                    eval_obs, eval_rew, eval_terminations, eval_truncations, eval_infos = eval_envs.step(act_eval)
                    next_done_eval = torch.logical_or(eval_terminations, eval_truncations).to(torch.float32)
                    if "final_info" in eval_infos:
                        mask = eval_infos["_final_info"]
                        num_episodes += mask.sum()
                        for k, v in eval_infos["final_info"]["episode"].items():
                            eval_metrics[k].append(v)
            print(f"Evaluated {args.num_eval_steps * args.num_eval_envs} steps resulting in {num_episodes} episodes")
            for k, v in eval_metrics.items():
                mean = torch.stack(v).float().mean()
                if logger is not None:
                    logger.add_scalar(f"eval/{k}", mean, global_step)
                print(f"{Fore.GREEN}Evaluation Metric: {k}{Style.RESET_ALL} | {Fore.CYAN}Mean: {mean:.4f}{Style.RESET_ALL}")
            if args.evaluate:
                break
        if args.save_model and iteration % args.eval_freq == 1:
            model_path = f"{SAVE_DIR}/{run_name}/{TIME}/ckpt_{video_iteration}_{iteration}.pt"
            video_iteration += 1
            torch.save(agent.state_dict(), model_path)
            print(f"model saved to {model_path}")

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
            
        rollout_time = time.time()
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value, next_lstm_state = agent.get_action_and_value(next_obs, next_lstm_state, next_done)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action)
            next_done = torch.logical_or(terminations, truncations).to(torch.float32)
            rewards[step] = reward.view(-1) * args.reward_scale

            if "final_info" in infos:
                final_info = infos["final_info"]
                done_mask = infos["_final_info"]
                for k, v in final_info["episode"].items():
                    logger.add_scalar(f"train/{k}", v[done_mask].float().mean(), global_step)
                for k in infos["final_observation"]:
                    infos["final_observation"][k] = infos["final_observation"][k][done_mask]
                with torch.no_grad():
                    final_values[step, torch.arange(args.num_envs, device=device)[done_mask]] = agent.get_value(infos["final_observation"],
                                                                                                                next_lstm_state,
                                                                                                                next_done).view(-1)

        rollout_time = time.time() - rollout_time

        # bootstrap value according to termination and truncation
        with torch.no_grad():
            next_value = agent.get_value(next_obs, next_lstm_state, next_done).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    next_not_done = 1.0 - next_done
                    nextvalues = next_value
                else:
                    next_not_done = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                real_next_values = next_not_done * nextvalues + final_values[t] # t instead of t+1
                # next_not_done means nextvalues is computed from the correct next_obs
                # if next_not_done is 1, final_values is always 0
                # if next_not_done is 0, then use final_values, which is computed according to bootstrap_at_done
                if args.finite_horizon_gae:
                    """
                    See GAE paper equation(16) line 1, we will compute the GAE based on this line only
                    1             *(  -V(s_t)  + r_t                                                               + gamma * V(s_{t+1})   )
                    lambda        *(  -V(s_t)  + r_t + gamma * r_{t+1}                                             + gamma^2 * V(s_{t+2}) )
                    lambda^2      *(  -V(s_t)  + r_t + gamma * r_{t+1} + gamma^2 * r_{t+2}                         + ...                  )
                    lambda^3      *(  -V(s_t)  + r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + gamma^3 * r_{t+3}
                    We then normalize it by the sum of the lambda^i (instead of 1-lambda)
                    """
                    if t == args.num_steps - 1: # initialize
                        lam_coef_sum = 0.
                        reward_term_sum = 0. # the sum of the second term
                        value_term_sum = 0. # the sum of the third term
                    lam_coef_sum = lam_coef_sum * next_not_done
                    reward_term_sum = reward_term_sum * next_not_done
                    value_term_sum = value_term_sum * next_not_done

                    lam_coef_sum = 1 + args.gae_lambda * lam_coef_sum
                    reward_term_sum = args.gae_lambda * args.gamma * reward_term_sum + lam_coef_sum * rewards[t]
                    value_term_sum = args.gae_lambda * args.gamma * value_term_sum + args.gamma * real_next_values

                    advantages[t] = (reward_term_sum + value_term_sum) / lam_coef_sum - values[t]
                else:
                    delta = rewards[t] + args.gamma * real_next_values - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * next_not_done * lastgaelam # Here actually we should use next_not_terminated, but we don't have lastgamlam if terminated
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_dones = dones.reshape(-1)

        # Optimizing the policy and value network
        assert args.num_envs % args.num_minibatches == 0
        agent.train()
        envsperbatch = args.num_envs // args.num_minibatches
        envinds = np.arange(args.num_envs)
        flatinds = np.arange(args.batch_size).reshape(args.num_steps, args.num_envs)
        clipfracs = []
        update_time = time.time()
        for epoch in range(args.update_epochs):
            np.random.shuffle(envinds)
            for start in range(0, args.num_envs, envsperbatch):
                end = start + envsperbatch
                mbenvinds = envinds[start:end]
                mb_inds = flatinds[:, mbenvinds].ravel()  # be really careful about the index

                _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
                    b_obs[mb_inds], 
                    (next_lstm_state[0][:, mbenvinds], 
                     next_lstm_state[1][:, mbenvinds]),
                    b_dones[mb_inds],
                    b_actions[mb_inds] # TODO .long()?
                    )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        update_time = time.time() - update_time

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        logger.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        logger.add_scalar("charts/global_step", global_step, global_step)
        logger.add_scalar("losses/value_loss", v_loss.item(), global_step)
        logger.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        logger.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        logger.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        logger.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        logger.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        logger.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        logger.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        logger.add_scalar("time/step", global_step, global_step)
        logger.add_scalar("time/update_time", update_time, global_step)
        logger.add_scalar("time/rollout_time", rollout_time, global_step)
        logger.add_scalar("time/rollout_fps", args.num_envs * args.num_steps / rollout_time, global_step)

    if args.save_model and not args.evaluate:
        model_path = f"{SAVE_DIR}/{run_name}/{TIME}/final_ckpt.pt"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

    if logger is not None: 
        logger.close()