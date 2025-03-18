import os
import torch
import gymnasium as gym
from tqdm import tqdm
from pathlib import Path
import shutil
import tyro
from dataclasses import dataclass
from typing import Optional

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from mani_skill.utils.wrappers import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from baselines.ppo.ppo_memtasks import AgentStateOnly, FlattenRGBDObservationWrapper
from mikasa_robo_suite.memory_envs import *
from mikasa_robo_suite.utils.wrappers import *


def env_info(env_id):
    noop_steps = 1
    if env_id in ['ShellGamePush-v0', 'ShellGamePick-v0', 'ShellGameTouch-v0']:
        wrappers_list = [
            (InitialZeroActionWrapper, {"n_initial_steps": noop_steps-1}),
            (RenderStepInfoWrapper, {}),
            (ShellGameRenderCupInfoWrapper, {}),
            (RenderRewardInfoWrapper, {}),
            (DebugRewardWrapper, {}),
        ]
        oracle_info = 'cup_with_ball_number'
        prompt_info = None
        EPISODE_TIMEOUT = 90
    elif env_id in ['InterceptSlow-v0', 'InterceptMedium-v0', 'InterceptFast-v0', 
                    'InterceptGrabSlow-v0', 'InterceptGrabMedium-v0', 'InterceptGrabFast-v0']:
        wrappers_list = [
            (InitialZeroActionWrapper, {"n_initial_steps": noop_steps-1}),
            (RenderStepInfoWrapper, {}),
            (RenderRewardInfoWrapper, {}),
            (DebugRewardWrapper, {}),
        ]
        oracle_info = None
        prompt_info = None
        EPISODE_TIMEOUT = 90
    elif env_id in ['RotateLenientPos-v0', 'RotateLenientPosNeg-v0',
                    'RotateStrictPos-v0', 'RotateStrictPosNeg-v0']:
        wrappers_list = [
            (InitialZeroActionWrapper, {"n_initial_steps": noop_steps-1}),
            (RenderStepInfoWrapper, {}),
            (RenderRewardInfoWrapper, {}),
            (RotateRenderAngleInfoWrapper, {}),
            (DebugRewardWrapper, {}),
        ]
        oracle_info = 'angle_diff'
        prompt_info = 'target_angle'
        EPISODE_TIMEOUT = 90
    elif env_id in ['CameraShutdownPush-v0', 'CameraShutdownPick-v0']:
        wrappers_list = [
            (InitialZeroActionWrapper, {"n_initial_steps": noop_steps-1}),
            (CameraShutdownWrapper, {"n_initial_steps": 19}), # camera works only for t ~ [0, 19]
            (RenderStepInfoWrapper, {}),
            (RenderRewardInfoWrapper, {}),
        ]
        oracle_info = None
        prompt_info = None
        EPISODE_TIMEOUT = 90
    elif env_id in ['TakeItBack-v0']:
        wrappers_list = [
            (InitialZeroActionWrapper, {"n_initial_steps": noop_steps-1}),
            (RenderStepInfoWrapper, {}),
            (RenderRewardInfoWrapper, {}),
            (DebugRewardWrapper, {}),
        ]
        oracle_info = None
        prompt_info = None
        EPISODE_TIMEOUT = 180
    elif env_id in ['RememberColor3-v0', 'RememberColor5-v0', 'RememberColor9-v0']:
        wrappers_list = [
            (InitialZeroActionWrapper, {"n_initial_steps": noop_steps-1}),
            (RememberColorInfoWrapper, {}),
            (RenderStepInfoWrapper, {}),
            (RenderRewardInfoWrapper, {}),
            (DebugRewardWrapper, {}),
        ]
        oracle_info = None
        prompt_info = None
        EPISODE_TIMEOUT = 60
    elif env_id in ['RememberShape3-v0', 'RememberShape5-v0', 'RememberShape9-v0']:
        wrappers_list = [
            (InitialZeroActionWrapper, {"n_initial_steps": noop_steps-1}),
            (RememberShapeInfoWrapper, {}),
            (RenderStepInfoWrapper, {}),
            (RenderRewardInfoWrapper, {}),
            (DebugRewardWrapper, {}),
        ]
        oracle_info = None
        prompt_info = None
        EPISODE_TIMEOUT = 60
    elif env_id in ['RememberShapeAndColor3x2-v0', 'RememberShapeAndColor3x3-v0', 'RememberShapeAndColor5x3-v0']:
        wrappers_list = [
            (InitialZeroActionWrapper, {"n_initial_steps": noop_steps-1}),
            (RememberShapeAndColorInfoWrapper, {}),
            (RenderStepInfoWrapper, {}),
            (RenderRewardInfoWrapper, {}),
            (DebugRewardWrapper, {}),
        ]
        oracle_info = None
        prompt_info = None
        EPISODE_TIMEOUT = 60
    elif env_id in ['BunchOfColors3-v0', 'BunchOfColors5-v0', 'BunchOfColors7-v0']:
        wrappers_list = [
            (InitialZeroActionWrapper, {"n_initial_steps": noop_steps-1}),
            (MemoryCapacityInfoWrapper, {}),
            (RenderStepInfoWrapper, {}),
            (RenderRewardInfoWrapper, {}),
            (DebugRewardWrapper, {}),
        ]
        oracle_info = None
        prompt_info = None
        EPISODE_TIMEOUT = 120
    elif env_id in ['SeqOfColors3-v0', 'SeqOfColors5-v0', 'SeqOfColors7-v0']:
        wrappers_list = [
            (InitialZeroActionWrapper, {"n_initial_steps": noop_steps-1}),
            (MemoryCapacityInfoWrapper, {}),
            (RenderStepInfoWrapper, {}),
            (RenderRewardInfoWrapper, {}),
            (DebugRewardWrapper, {}),
        ]
        oracle_info = None
        prompt_info = None
        EPISODE_TIMEOUT = 120
    elif env_id in ['ChainOfColors3-v0', 'ChainOfColors5-v0', 'ChainOfColors7-v0']:
        wrappers_list = [
            (InitialZeroActionWrapper, {"n_initial_steps": noop_steps-1}),
            (MemoryCapacityInfoWrapper, {}),
            (RenderStepInfoWrapper, {}),
            (RenderRewardInfoWrapper, {}),
            (DebugRewardWrapper, {}),
        ]
        oracle_info = None
        prompt_info = None
        EPISODE_TIMEOUT = 120
    else:
        raise ValueError(f"Unknown environment: {env_id}")
    
    wrappers_list.insert(0, (StateOnlyTensorToDictWrapper, {}))

    return wrappers_list, EPISODE_TIMEOUT


def collect_batched_data_from_ckpt(
    env_id="ShellGameTouch-v0", 
    checkpoint_path=None, 
    path_to_save_data="data",
    num_train_data=1000
):    
    """
    Collect batched data, consequent unbatching required!!!
    """
    # env_id = "ShellGameTouch-v0"
    
    NUMBER_OF_TRAIN_DATA = num_train_data
    batch_size = 250
    NUMBER_OF_BATCHES = NUMBER_OF_TRAIN_DATA // batch_size
    # render = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env_kwargs_state = dict(
        obs_mode="state",
        control_mode="pd_joint_delta_pos",
        render_mode="all",
        sim_backend="gpu",
        reward_mode="normalized_dense"
    )

    env_kwargs_rgb = dict(
        obs_mode="rgb",
        control_mode="pd_joint_delta_pos",
        render_mode="all",
        sim_backend="gpu",
        reward_mode="normalized_dense"
    )

    env_state = gym.make(env_id, num_envs=batch_size, **env_kwargs_state)
    env_rgb = gym.make(env_id, num_envs=batch_size, **env_kwargs_rgb)

    state_wrappers_list, episode_timeout = env_info(env_id)

    for wrapper_class, wrapper_kwargs in state_wrappers_list:
        env_state = wrapper_class(env_state, **wrapper_kwargs)

    env_state = FlattenRGBDObservationWrapper(
        env_state, 
        rgb=False,
        depth=False,
        state=True,
        oracle=False,
        joints=False
    )

    rgb_wrappers_list, _ = env_info(env_id)

    for wrapper_class, wrapper_kwargs in rgb_wrappers_list:
        env_rgb = wrapper_class(env_rgb, **wrapper_kwargs)

    env_rgb = FlattenRGBDObservationWrapper(
        env_rgb, 
        rgb=True,
        depth=False,
        state=False,
        oracle=False,
        joints=True
    )

    if isinstance(env_state.action_space, gym.spaces.Dict):
        env_state = FlattenActionSpaceWrapper(env_state)
    if isinstance(env_rgb.action_space, gym.spaces.Dict):
        env_rgb = FlattenActionSpaceWrapper(env_rgb)

    # env_state = RecordEpisode(
    #     env_state,
    #     output_dir="dataset_collection_videos/state",
    #     save_trajectory=True,
    #     video_fps=30
    # )

    # env_rgb = RecordEpisode(
    #     env_rgb,
    #     output_dir="dataset_collection_videos/rgb",
    #     save_trajectory=True,
    #     video_fps=30
    # )

    env_state = ManiSkillVectorEnv(env_state, batch_size, ignore_terminations=True, record_metrics=True)
    env_rgb = ManiSkillVectorEnv(env_rgb, batch_size, ignore_terminations=True, record_metrics=True)

    agent = AgentStateOnly(env_state).to(device)
    agent.load_state_dict(torch.load(checkpoint_path, map_location=device))
    agent.eval()


    # save_dir = f'data/MIKASA-Robo/batched/{env_id}'
    save_dir = f'{path_to_save_data}/MIKASA-Robo/batched/{env_id}'
    os.makedirs(save_dir, exist_ok=True)

    # Dataset collection
    print(f"Generating {NUMBER_OF_TRAIN_DATA} episodes in {NUMBER_OF_BATCHES} batches (batched with batch size {batch_size})")
    for episode in tqdm(range(NUMBER_OF_BATCHES)):
        rgbList, jointsList,actList, rewList, succList, doneList = [], [], [], [], [], []
        
        # Reset of both environments with the same seed for synchronization
        # seed = np.random.randint(0, 10000)
        seed = episode
        obs_state, _ = env_state.reset(seed=seed)
        obs_rgb, _ = env_rgb.reset(seed=seed)
        
        done = False
        for t in range(episode_timeout):
            # Get action from agent based on state
            rgbList.append(obs_rgb['rgb'].cpu().numpy())
            jointsList.append(obs_rgb['joints'].cpu().numpy())
            with torch.no_grad():
                for key, value in obs_state.items():
                    obs_state[key] = value.to(device)
                action = agent.get_action(obs_state, deterministic=True)
            
            # Make a step in both environments with the same action
            obs_state, reward_state, term_state, trunc_state, info_state = env_state.step(action)
            obs_rgb, reward_rgb, term_rgb, trunc_rgb, info_rgb = env_rgb.step(action)

            rewList.append(reward_rgb.cpu().numpy())
            succList.append(info_rgb['success'].cpu().numpy().astype(int))
            actList.append(action.cpu().numpy())
            done = torch.logical_or(term_rgb, trunc_rgb)
            doneList.append(done.cpu().numpy().astype(int))
            
            # Check synchronization of environments
            # assert np.allclose(reward_state.cpu().numpy(), reward_rgb.cpu().numpy()), "Environments desynchronized!"

        DATA = {'rgb': np.array(rgbList), # (15, 6, 128, 128)
                'joints': np.array(jointsList), # (15, 25)
                'action': np.array(actList), # (15, 8)
                'reward': np.array(rewList), # (15,)
                'success': np.array(succList), # (15,)
                'done': np.array(doneList)} # (15,)
        
        file_path = f'{save_dir}/train_data_{episode}.npz'
        np.savez(file_path, **DATA)
        
        # print(f"Episode completed")
        # if "final_info" in info_rgb:
        #     for k, v in info_rgb["final_info"]["episode"].items():
        #         print(f"{k}: {v.item()}")


    env_state.close()
    env_rgb.close()

    print(f"\nDataset saved to {save_dir}")


def collect_unbatched_data_from_batched(env_id="ShellGameTouch-v0", path_to_save_data="data"):
    dir_with_batched_data = f'{path_to_save_data}/MIKASA-Robo/batched/{env_id}'
    NUMBER_OF_BATCHES = len(list(Path(dir_with_batched_data).glob('*')))
    print(f"Unbatching {dir_with_batched_data}, {NUMBER_OF_BATCHES} batches")

    traj_cnt = 0
    save_dir_unbatched = f'{path_to_save_data}/MIKASA-Robo/unbatched/{env_id}'
    os.makedirs(save_dir_unbatched, exist_ok=True)

    for episode in tqdm(range(NUMBER_OF_BATCHES)):
        episode = np.load(f'{dir_with_batched_data}/train_data_{episode}.npz')
        episode = {key: episode[key] for key in episode.keys()}
        for trajectory_num in range(episode['reward'].shape[1]):
            unbatched_rgb = episode['rgb'][:, trajectory_num, :, :, :]
            unbatched_joints = episode['joints'][:, trajectory_num, :]
            unbatched_action = episode['action'][:, trajectory_num, :]
            unbatched_reward = episode['reward'][:, trajectory_num]
            unbatched_success = episode['success'][:, trajectory_num]
            unbatched_done = episode['done'][:, trajectory_num]

            DATA = {'rgb': unbatched_rgb,
                    'joints': unbatched_joints,
                    'action': unbatched_action,
                    'reward': unbatched_reward,
                    'success': unbatched_success,
                    'done': unbatched_done}

            file_path = f'{save_dir_unbatched}/train_data_{traj_cnt}.npz'
            np.savez(file_path, **DATA)

            traj_cnt += 1


def get_list_of_all_checkpoints_available(ckpt_dir="."):
    oracle_checkpoints_dir = os.path.join(ckpt_dir, "oracle_checkpoints")

    # 1) Check if oracle_checkpoints_dir exists
    if not os.path.exists(oracle_checkpoints_dir):
        raise FileNotFoundError(f"Directory {oracle_checkpoints_dir} does not exist.")
    
    checkpoint_paths = []
    
    # 2) Iterate over all directories in oracle_checkpoints/ppo_memtasks/state/normalized_dense/
    normalized_dense_dir = os.path.join(oracle_checkpoints_dir, "ppo_memtasks", "state", "normalized_dense")
    for env_dir in os.listdir(normalized_dense_dir):
        env_path = os.path.join(normalized_dense_dir, env_dir)
        if os.path.isdir(env_path):
            # 3) Create list of checkpoint paths
            for root, _, files in os.walk(env_path):
                for file in files:
                    if file == "final_success_ckpt.pt":
                        checkpoint_paths.append([env_dir, os.path.join(root, file)])
    
    return checkpoint_paths


@dataclass
class Args:
    env_id: Optional[str] = "ShellGameTouch-v0"
    path_to_save_data: str = "data"
    ckpt_dir: str = "."
    num_train_data: int = 1000

if __name__ == "__main__":
    args = tyro.cli(Args)
    path_to_save_data = args.path_to_save_data
    ckpt_dir = args.ckpt_dir
    ENV_ID = args.env_id

    # Check if num_train_data is divisible by batch_size
    batch_size = 250
    if args.num_train_data % batch_size != 0:
        raise ValueError(f"num_train_data ({args.num_train_data}) must be divisible by batch_size ({batch_size})")

    # * 0. Get list of all checkpoints available
    checkpoints = get_list_of_all_checkpoints_available(ckpt_dir=ckpt_dir)
    # print("Available checkpoints:")
    # for i, (env_id, checkpoint) in enumerate(checkpoints):
    #     print(f"{i+1}. {env_id} - {checkpoint}")

    for env_id, checkpoint in checkpoints:
        if env_id == ENV_ID:
            print(f"Collecting data for {env_id} from {checkpoint}")
            
            # * 1. Collect batched data from ckpt
            collect_batched_data_from_ckpt(
                env_id=env_id, 
                checkpoint_path=checkpoint, 
                path_to_save_data=path_to_save_data,
                num_train_data=args.num_train_data
            )

            # * 2. Unbatch batched data
            collect_unbatched_data_from_batched(
                env_id=env_id, 
                path_to_save_data=path_to_save_data
            )

            # * 3. Remove batched data
            dir_with_batched_data = f'{path_to_save_data}/MIKASA-Robo/batched/{env_id}'
            shutil.rmtree(dir_with_batched_data)
            print(f"Deleted batched data for {env_id} at {dir_with_batched_data}")

# python3 mikasa_robo_suite/dataset_collectors/get_mikasa_robo_datasets.py --env-id=ShellGameTouch-v0 --path-to-save-data="data" --ckpt-dir="." --num-train-data=1000