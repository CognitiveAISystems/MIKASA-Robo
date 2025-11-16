import json
import logging
import os
import sys
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union
from collections import defaultdict

import draccus
import numpy as np
import tensorflow as tf
import tqdm

import wandb
import tyro

sys.path.append("../..")
sys.path.append("../")

from experiments.robot.openvla_utils import (
    get_action_head,
    get_noisy_action_projector,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK

from mikasa_utils import get_mikasa_eval_env

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class Args:

    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path

    use_l1_regression: bool = True                   # If True, uses continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, uses continuous action head with diffusion modeling objective (DDIM)
    num_diffusion_steps_train: int = -1              # (When `diffusion==True`) Number of diffusion steps used for training
    num_diffusion_steps_inference: int = -1          # (When `diffusion==True`) Number of diffusion steps used for inference
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 2                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = False                         # Whether to include proprio state in input

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 1                     # Number of actions to execute open-loop before requerying policy

    lora_rank: int = 32                              # Rank of LoRA weight matrix (MAKE SURE THIS MATCHES TRAINING!)

    unnorm_key: Union[str, Path] = "mikasa_dataset"                # Action un-normalization key

    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization


    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    # use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_entity: str = "anon"          # Name of WandB entity
    wandb_project: str = "MIKASA-Robo-VLA"           # Name of WandB project
    use_wandb: bool = True

    #################################################################################################################
    # MIKASA environment-specific parameters
    #################################################################################################################
    env_img_res: int = 256                           # Resolution for environment images (not policy input resolution)
    exp_name: Optional[str] = None
    env_id: str = "RememberColor3-v0" #'RememberColor3-v0' #"ShellGamePush-v0"
    language_instruction: str = ""
    """The environment ID of the task you want to simulate."""
    shader: str = "default"
    num_episodes: int = 100
    """Number of episodes to run and record evaluation metrics over"""
    record_dir: str = "videos"
    """The directory to save videos and results"""
    model: Optional[str] = "octo-base"
    """The model to evaluate on the given environment. Can be one of octo-base, octo-small, rt-1x. If not given, random actions are sampled."""
    ckpt_path: str = "" 
    """Checkpoint path for models. Only used for RT models"""
    seed: int = 0
    """Seed the model and environment. Default seed is 0"""
    reset_by_episode_id: bool = True
    """Whether to reset by fixed episode ids instead of random sampling initial states."""
    info_on_video: bool = False
    """Whether to write info text onto the video"""
    save_video: bool = True
    """Whether to save videos"""
    device: str = 'cuda:0'
    camera_width: Optional[int] = 128
    """the width of the camera image. If none it will use the default the environment specifies"""
    camera_height: Optional[int] = 128
    """the height of the camera image. If none it will use the default the environment specifies."""
    include_oracle: bool = False
    """if toggled, oracle info (such as cup_with_ball_number in ShellGamePush-v0) will be used during the training, i.e. reducing memory task to MDP"""
    noop_steps: int = 1
    """if = 1, then no noops, if > 1, then noops for t ~ [0, noop_steps-1]"""
    include_rgb: bool = True
    """if toggled, rgb images will be included in the observation space"""
    include_joints: bool = False
    """[works only with include_rgb=True] if toggled, joints will be included in the observation space"""
    reward_mode: str = 'normalized_dense' # sparse | normalized_dense
    """the mode of the reward function"""
    control_mode: Optional[str] = "pd_ee_delta_pose"
    """the control mode to use for the environment"""
    render_mode: str = "all"
    """the environment rendering mode"""
    """the id of the environment"""
    include_state: bool = False
    """whether to include state information in observations"""
    num_eval_steps: int = 60
    num_eval_episodes: int = 100
    sim_backend: str = 'gpu'



def validate_config(args: Args) -> None:
    """Validate configuration parameters."""
    assert args.pretrained_checkpoint is not None, "pretrained_checkpoint must not be None!"

    if "image_aug" in str(args.pretrained_checkpoint):
        assert args.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"

    assert not (args.load_in_8bit and args.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"



def initialize_model(args: Args):
    """Initialize model and associated components."""
    # Load model
    model = get_model(args)

    # Load proprio projector if needed
    proprio_projector = None
    if args.use_proprio:
        proprio_projector = get_proprio_projector(
            args,
            model.llm_dim,
            proprio_dim=8,  # 8-dimensional proprio for LIBERO
        )

    # Load action head if needed
    action_head = None
    if args.use_l1_regression or args.use_diffusion:
        action_head = get_action_head(args, model.llm_dim)

    # Load noisy action projector if using diffusion
    noisy_action_projector = None
    if args.use_diffusion:
        noisy_action_projector = get_noisy_action_projector(args, model.llm_dim)

    # Get OpenVLA processor if needed
    processor = None
    if args.model_family == "openvla":
        processor = get_processor(args)
        check_unnorm_key(args, model)

    return model, action_head, proprio_projector, noisy_action_projector, processor


def check_unnorm_key(args: Args, model) -> None:
    """Check that the model contains the action un-normalization key."""
    # Initialize unnorm_key
    unnorm_key = args.unnorm_key

    # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
    # with the suffix "_no_noops" in the dataset name)
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"

    assert unnorm_key in model.norm_stats, f"Action un-norm key {unnorm_key} not found in VLA `norm_stats`!"



def setup_logging(args: Args):
    """Set up logging to file and optionally to wandb."""
    # Create run ID
    run_id = f"EVAL-{args.env_id}-{args.model_family}-{DATE_TIME}"
    if args.run_id_note is not None:
        run_id += f"--{args.run_id_note}"

    # Set up local logging
    os.makedirs(args.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(args.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging if enabled
    if args.use_wandb:
        wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=run_id,
        )

    return log_file, local_log_filepath, run_id


def log_message(message: str, log_file=None):
    """Log a message to console and optionally to a log file."""
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()


def prepare_observation(obs, resize_size):
    """Prepare observation for policy input."""
    # Get preprocessed images
    img = get_libero_image(obs)
    wrist_img = get_libero_wrist_image(obs)

    # Resize images to size expected by model
    img_resized = resize_image_for_policy(img, resize_size)
    wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)

    # Prepare observations dict
    observation = {
        "full_image": img_resized,
        "wrist_image": wrist_img_resized,
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
    }

    return observation, img  # Return both processed observation and original image for replay


def process_action(action, model_family):
    """Process action before sending to environment."""
    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
    action = normalize_gripper_action(action, binarize=True)

    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
    if model_family == "openvla":
        action = invert_gripper_action(action)

    return action

if __name__ == "__main__":

    args = tyro.cli(Args)

    # Validate configuration
    validate_config(args)

    # Set random seed
    set_seed_everywhere(args.seed)

    env = get_mikasa_eval_env(args)

    # Initialize model and components
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(args)

    # Get expected image dimensions
    resize_size = get_image_resize_size(args)

    # Setup logging
    log_file, local_log_filepath, run_id = setup_logging(args)

    eval_metrics = defaultdict(list)
    num_episodes = 0

    seeds = list(range(1, args.num_eval_episodes + 1))

    for j in range(args.num_eval_episodes):


        # Initialize action queue
        if args.num_open_loop_steps != NUM_ACTIONS_CHUNK:
            print(f"WARNING: args.num_open_loop_steps ({args.num_open_loop_steps}) does not match the NUM_ACTIONS_CHUNK "
                    f"({NUM_ACTIONS_CHUNK}) constant defined in prismatic.vla.constants! For best performance (in terms of "
                    "both speed and success rate), we recommend executing the full action chunk.")
        action_queue = deque(maxlen=args.num_open_loop_steps)


        print(f'Eval episode {j}')

        obs, info = env.reset(seed = seeds[j], options={})
        # images = [obs["image_primary"][0]]


        for i in range(args.num_eval_steps):
            observation = {}

            image_primary = tf.convert_to_tensor(obs['image_primary'].cpu().numpy()[0])
            observation['full_image'] = resize_image_for_policy(image_primary, resize_size)

            image_wrist = tf.convert_to_tensor(obs['image_wrist'].cpu().numpy()[0])
            observation['wrist_image'] = resize_image_for_policy(image_wrist, resize_size)

            # If action queue is empty, requery model
            if len(action_queue) == 0:
                # Query model to get action
                actions = get_action(
                    args,
                    model,
                    observation,
                    args.language_instruction,
                    processor=processor,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    use_film=args.use_film,
                )
                action_queue.extend(actions)

            # Get action from queue and process action
            action = action_queue.popleft()
            action = process_action(action, args.model_family)


            obs, reward, done, trunc, info = env.step(np.array(action))


            if "final_info" in info:
                mask = info["_final_info"][0]
                num_episodes += mask.sum()
                for k, v in info["final_info"]["episode"].items():
                    eval_metrics[k].append(float(v.item()))
                    wandb.log({k:float(v.item())}, step = j)
                    wandb.log({f"mean_{k}": np.array(eval_metrics[k]).mean()}, step = j)
                    wandb.log({f"std_{k}": np.array(eval_metrics[k]).std()}, step = j)

                break

        succ = eval_metrics['success_once'][-1]
        print(f'Episode {j} success_once: {succ}')