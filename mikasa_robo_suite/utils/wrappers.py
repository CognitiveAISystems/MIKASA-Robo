import gymnasium as gym
from mani_skill.envs.sapien_env import BaseEnv
import cv2
import numpy as np
import torch
from gymnasium import spaces


class StateOnlyTensorToDictWrapper(gym.ObservationWrapper):
    """Wrapper that converts tensor observation to a dictionary with 'state' key."""
    
    def __init__(self, env):
        super().__init__(env)
        
        orig_obs_space = env.observation_space
        
        
        self.observation_space = spaces.Dict({
            'state': orig_obs_space
        })

    def observation(self, obs):
        if not isinstance(obs, dict):
            obs = {'state': obs}
            b_ = obs['state'].shape[0]
        else:
            obs = obs.copy()
            b_ = obs['agent']['qpos'].shape[0]
            # obs.update({'rgb': self.unwrapped.rgb.unsqueeze(-1)})

        prompt_ = self.unwrapped.prompt
        oracle_info_ = self.unwrapped.oracle_info

        if prompt_ is not None:
            if len(prompt_.shape) == 1:
                prompt_ = prompt_.unsqueeze(-1)
        else:
            prompt_ = torch.ones(b_, 1) * 4242424242

        if oracle_info_ is not None:
            if len(oracle_info_.shape) == 1:
                oracle_info_ = oracle_info_.unsqueeze(-1)
        else:
            oracle_info_ = torch.ones(b_, 1) * 4242424242
        
        obs.update({
            'prompt': prompt_,
            'oracle_info': oracle_info_
        })
        return obs
    
# class StateOnlyTensorToDictWrapper(gym.ObservationWrapper):
#     """Wrapper that converts tensor observation to a dictionary with 'state' key."""
    
#     def __init__(self, env):
#         super().__init__(env)
        
#         orig_obs_space = env.observation_space
        
#         self.observation_space = spaces.Dict({
#             'state': orig_obs_space
#         })

#     def observation(self, obs):
#         return {'state': obs, 'prompt': self.unwrapped.prompt.unsqueeze(-1)}
    
# class RotateAddAngleObservationWrapper(gym.ObservationWrapper):
#     def __init__(self, env):
#         super().__init__(env)

#         init_obs = self.observation(self.base_env._init_raw_obs)
#         self.base_env.update_obs_space(init_obs)
    
#     @property
#     def base_env(self) -> BaseEnv:
#         return self.env.unwrapped

    
#     def observation(self, obs):
#         if isinstance(obs, dict):
#             obs = obs.copy()
#             obs['oracle_info'] = self.angle_diff.unsqueeze(-1)
#         return obs
    
# class RotateAddAngleObservationWrapper(gym.ObservationWrapper):
#     def __init__(self, env):
#         super().__init__(env)

#         init_obs = self.observation(self.base_env._init_raw_obs)
#         self.base_env.update_obs_space(init_obs)
    
#     @property
#     def base_env(self) -> BaseEnv:
#         return self.env.unwrapped

    
#     def observation(self, obs):
#         if isinstance(obs, dict):
#             obs = obs.copy()
#             obs['target_angle'] = self.target_angle
#         return obs
    

class RotateRenderAngleInfoWrapper(gym.Wrapper):
    """
    A wrapper that renders the current step count and target cup on the screen.
    """
    def __init__(self, env):
        super().__init__(env)
        self.step_count = 0
        self.current_obs = None
        
    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.info = info
        self.step_count = info['elapsed_steps'].detach().cpu().numpy()
        self.current_obs = obs
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self.info = info
        self.current_obs = obs
        self.step_count = info['elapsed_steps'].detach().cpu().numpy()
        return obs, reward, terminated, truncated, info
    
    def render(self):
        # Get the base render from the environment
        frame = self.env.render()

        # Add text
        for i in range(len(frame)):
            # if isinstance(self.current_obs, dict):
            target_angle = str(
                np.round(self.info["prompt"][i].item() * 180 / np.pi, 2)
                )
            current_angle = str(
                np.round(self.info['relative_angle'][i].item() * 180 / np.pi, 2)
                )
            cv2.putText(
                frame[i],
                'Target : ' + target_angle + ' deg',
                (10, 60),  # position
                cv2.FONT_HERSHEY_SIMPLEX,  # font
                1.0,  # font scale
                (255, 255, 255),  # color (white)
                2,  # thickness
                cv2.LINE_AA
            )

            cv2.putText(
                frame[i],
                'Current: ' + current_angle + ' deg',
                (10, 90),  # position
                cv2.FONT_HERSHEY_SIMPLEX,  # font
                1.0,  # font scale
                (255, 255, 255),  # color (white)
                2,  # thickness
                cv2.LINE_AA
            )

            # cv2.putText(
            #     frame[i],
            #     'Error: ' + error_angle + ' deg',
            #     (10, 120),  # position
            #     cv2.FONT_HERSHEY_SIMPLEX,  # font
            #     1.0,  # font scale
            #     (255, 255, 255),  # color (white)
            #     2,  # thickness
            #     cv2.LINE_AA
                # )
            
        return frame


class RenderStepInfoWrapper(gym.Wrapper):
    """
    A wrapper that renders the current step count and target cup on the screen.
    """
    def __init__(self, env):
        super().__init__(env)
        self.step_count = 0
        self.current_obs = None
        
    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.step_count = info['elapsed_steps'].detach().cpu().numpy()
        self.current_obs = obs
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self.current_obs = obs
        self.step_count = info['elapsed_steps'].detach().cpu().numpy()
        return obs, reward, terminated, truncated, info
    
    def render(self):
        # Get the base render from the environment
        frame = self.env.render()
        # frame = frame.detach().cpu().numpy()
        if torch.is_tensor(frame):
            frame = frame.detach().cpu().numpy()

        # Add text
        for i in range(len(frame)):
            # Env. step
            cv2.putText(
                frame[i],
                f'Step: {self.step_count[i]}',
                (10, 30),  # position
                cv2.FONT_HERSHEY_SIMPLEX,  # font
                1.0,  # font scale
                (255, 255, 255),  # color (white)
                2,  # thickness
                cv2.LINE_AA
            )
            
        return frame
    
class RenderRewardInfoWrapper(gym.Wrapper):
    """
    A wrapper that renders the current reward on the screen.
    """
    def __init__(self, env):
        super().__init__(env)
        self.reward = None
        
    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.reward = None
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self.reward = reward
        return obs, reward, terminated, truncated, info
    
    def render(self):
        # Get the base render from the environment
        frame = self.env.render()
        if torch.is_tensor(frame):
            frame = frame.detach().cpu().numpy()

        for i in range(len(frame)):
            if self.reward is not None:
                render_reward = self.reward[i].detach().cpu().numpy()
            else:
                render_reward = 0.0
            cv2.putText(
                frame[i],
                f'Reward: {render_reward:.3f}',
                (10, 120),  # position
                cv2.FONT_HERSHEY_SIMPLEX,  # font
                1.0,  # font scale
                (255, 255, 255),  # color (white)
                2,  # thickness
                cv2.LINE_AA
            )

        return frame


class CameraShutdownWrapper(gym.Wrapper):
    """Wrapper that zeros out all camera observations
    
    if n_initial_steps = 4 then t \in [0, 4] (5 steps) action is zero
    if n_initial_steps = 9 then t \in [0, 9] (10 steps) action is zero
    if n_initial_steps = 19 then t \in [0, 19] (20 steps) action is zero
    
    """
    
    def __init__(self, env, n_initial_steps=19):
        super().__init__(env)

        render_camera_config = env.unwrapped._default_human_render_camera_configs
        self.width = render_camera_config.width
        self.height = render_camera_config.height

        self.n_initial_steps = n_initial_steps
        self.current_steps = None
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.current_steps = info['elapsed_steps'].detach().cpu().numpy()
        
        # Zero out camera observations if they exist
        if (self.current_steps > self.n_initial_steps).any():
            if isinstance(obs, dict):
                for key in obs:
                    if 'sensor_data' in key:
                        for key2 in obs['sensor_data']:
                            if 'hand_camera' in key2:
                                for key3 in obs[key][key2]:
                                    obs[key][key2][key3] *= 0
                            if 'base_camera' in key2:
                                for key3 in obs[key][key2]:
                                    obs[key][key2][key3] *= 0

        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.current_steps = info['elapsed_steps'].detach().cpu().numpy()
                    
        return obs, info
    
    def render(self):
        img = self.env.render()
        if (self.current_steps > self.n_initial_steps).any():
            img[:, :, self.width:, :] *= 0
            
        return img


# class ShellGameAddBallInfoWrapper(gym.ObservationWrapper):
# ! not need now
#     """
#     A wrapper for the ShellGamePush and ShellGamePick environments that adds oracle information about the ball's position to the observation space.
    
#     This wrapper is intended for use during testing or oracle training only. It should not be used during memory evaluation 
#     as it provides additional information that would not be available in a real-world scenario.

#     Attributes:
#         env (gym.Env): The environment to be wrapped.

#     Methods:
#         observation(obs): Modifies the observation to include the ball's position.
#     """
#     def __init__(self, env):
#         super().__init__(env)

#         init_obs = self.observation(self.base_env._init_raw_obs)
#         self.base_env.update_obs_space(init_obs)
    
#     @property
#     def base_env(self) -> BaseEnv:
#         return self.env.unwrapped

    
#     def observation(self, obs):
#         if isinstance(obs, dict):
#             obs = obs.copy()
#             obs['cup_with_ball_number'] = self.cup_with_ball_number
#         return obs


class InitialZeroActionWrapper(gym.ActionWrapper):
    def __init__(self, env, n_initial_steps=1):
        """
        A wrapper that forces zero actions for a specified number of initial steps in the environment.

        Args:
            env: environment
            n_initial_steps: number of steps with zero actions
        """
        super().__init__(env)
        self.n_initial_steps = n_initial_steps
        self.current_steps = None
        
    def action(self, action):
        """Modifies action before sending it to the environment"""
        if self.current_steps is None or (self.current_steps < self.n_initial_steps).any():
            # Zero out actions for environments still in initial phase
            mask = (self.current_steps < self.n_initial_steps)
            modified_action = action.clone()
            modified_action[mask] = 0
            return modified_action
        return action
        
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self.current_steps = info['elapsed_steps'].detach().cpu().numpy()
        return obs, reward, terminated, truncated, info
        
    def reset(self, **kwargs):
        """Resets the step counter"""
        obs, info = super().reset(**kwargs)
        self.current_steps = info['elapsed_steps'].detach().cpu().numpy()
        return obs, info
    

class ShellGameRenderCupInfoWrapper(gym.Wrapper):
    """
    A wrapper that renders the current step count and target cup on the screen.
    """
    def __init__(self, env):
        super().__init__(env)
        self.step_count = 0
        self.current_obs = None
        self.info = None
        
    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.info = info
        self.step_count = info['elapsed_steps'].detach().cpu().numpy()
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self.info = info
        self.step_count = info['elapsed_steps'].detach().cpu().numpy()
        return obs, reward, terminated, truncated, info
    
    def render(self):
        # Get the base render from the environment
        frame = self.env.render()
        if torch.is_tensor(frame):
            frame = frame.detach().cpu().numpy()

        # Add text
        for i in range(len(frame)):
            # if self.current_obs is not None:
            if self.info["oracle_info"][i] == 0:
                cup = 'Target: Left'
            elif self.info["oracle_info"][i] == 1:
                cup = 'Target: Center'
            elif self.info["oracle_info"][i] == 2:
                cup = 'Target: Right'
            # Target cup
            cv2.putText(
                frame[i],
                cup,
                (10, 60),  # position
                cv2.FONT_HERSHEY_SIMPLEX,  # font
                1.0,  # font scale
                (255, 255, 255),  # color (white)
                2,  # thickness
                cv2.LINE_AA
            )
            
        return frame
    



class DebugRewardWrapper(gym.Wrapper):
    """
    A wrapper that renders the current step count and target cup on the screen.
    """
    def __init__(self, env):
        super().__init__(env)
        self.info = None
        
    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.info = info
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self.info = info
        return obs, reward, terminated, truncated, info
    
    def render(self):
        # Get the base render from the environment
        frame = self.env.render()
        if torch.is_tensor(frame):
            frame = frame.detach().cpu().numpy()
        
        for i in range(len(frame)):
            if 'reward_dict' in self.info and self.info['reward_dict'] is not None:
                for reward_num, (reward_key, reward_value) in enumerate(self.info['reward_dict'].items()):
                    
                    cv2.putText(
                        frame[i],
                        f"{reward_key}: {reward_value[i].detach().cpu().numpy():.3f}",
                        (10, 150+(reward_num+1)*20),  # position
                        cv2.FONT_HERSHEY_SIMPLEX,  # font
                        0.5,  # font scale
                        (255, 255, 255),  # color (white)
                        1,  # thickness
                        cv2.LINE_AA
                )
            
        return frame
    



class RememberColorInfoWrapper(gym.Wrapper):
    """
    A wrapper that renders the current step count and target cup on the screen.
    """
    def __init__(self, env):
        super().__init__(env)
        self.step_count = 0
        self.current_obs = None
        self.info = None
        self.colors_names = {
            0: 'Red',
            1: 'Lime',
            2: 'Blue',
            3: 'Yellow',
            4: 'Magenta',
            5: 'Cyan',
            6: 'Maroon',
            7: 'Olive',
            8: 'Teal',
        }
        
    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.info = info
        self.step_count = info['elapsed_steps'].detach().cpu().numpy()
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self.info = info
        self.step_count = info['elapsed_steps'].detach().cpu().numpy()
        return obs, reward, terminated, truncated, info
    
    def render(self):
        # Get the base render from the environment
        frame = self.env.render()
        if torch.is_tensor(frame):
            frame = frame.detach().cpu().numpy()

        # Add text
        for i in range(len(frame)):
            # if self.current_obs is not None:
            color_name = self.colors_names[self.info["oracle_info"][i].item()]
            color_name = f"Target: {color_name}"

            # Target cup
            cv2.putText(
                frame[i],
                color_name,
                (10, 60),  # position
                cv2.FONT_HERSHEY_SIMPLEX,  # font
                1.0,  # font scale
                (255, 255, 255),  # color (white)
                2,  # thickness
                cv2.LINE_AA
            )
            
        return frame
    

class RememberShapeInfoWrapper(gym.Wrapper):
    """
    A wrapper that renders the current step count and target cup on the screen.
    """
    def __init__(self, env):
        super().__init__(env)
        self.step_count = 0
        self.current_obs = None
        self.info = None
        self.SHAPES_names = {
            0: "cube",
            1: "sphere",
            2: "cylinder",
            3: "cross",
            4: "torus",
            5: "star",
            6: "pyramide",
            7: "t_shape",
            8: "crescent",
        }
        
    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.info = info
        self.step_count = info['elapsed_steps'].detach().cpu().numpy()
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self.info = info
        self.step_count = info['elapsed_steps'].detach().cpu().numpy()
        return obs, reward, terminated, truncated, info
    
    def render(self):
        # Get the base render from the environment
        frame = self.env.render()
        if torch.is_tensor(frame):
            frame = frame.detach().cpu().numpy()

        # Add text
        for i in range(len(frame)):
            # if self.current_obs is not None:
            color_name = self.SHAPES_names[self.info["oracle_info"][i].item()]
            color_name = f"Target: {color_name}"

            # Target cup
            cv2.putText(
                frame[i],
                color_name,
                (10, 60),  # position
                cv2.FONT_HERSHEY_SIMPLEX,  # font
                1.0,  # font scale
                (255, 255, 255),  # color (white)
                2,  # thickness
                cv2.LINE_AA
            )
            
        return frame
    

import cv2

class RememberShapeAndColorInfoWrapper(gym.Wrapper):
    """
    A wrapper that renders the current step count and target cup on the screen.
    """
    def __init__(self, env):
        super().__init__(env)
        self.step_count = 0
        self.current_obs = None
        self.info = None

        self._env = env
        # while not isinstance(self._env, RememberShapeAndColorBaseEnv):
        #     self._env = self._env.env
            
        self.shape_dict = self._env.base_shapes
        print(self.shape_dict)
        # self.color_dict = self._env.color_dict
        
    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.info = info
        self.step_count = info['elapsed_steps'].detach().cpu().numpy()
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self.info = info
        self.step_count = info['elapsed_steps'].detach().cpu().numpy()
        return obs, reward, terminated, truncated, info
    
    def decode_color(self, color_id):
        """
        Converts color ID to color name
        """
        if color_id == 0:
            return "Red"
        elif color_id == 1:
            return "Green"
        elif color_id == 2:
            return "Blue"
        else:
            return "Unknown"
        
    def decode_shape(self, shape_id):
        """
        Converts shape ID to shape name
        """
        return self.shape_dict.get(shape_id, "Unknown")
    
    def render(self):
        # Get the base render from the environment
        frame = self.env.render()
        if torch.is_tensor(frame):
            frame = frame.detach().cpu().numpy()

        # Add text
        for i in range(len(frame)):
            shape_id = self.info["oracle_info"][i][0].item()  # shape info
            color_id = self.info["oracle_info"][i][1].item()  # color info
            
            color_name = self.decode_color(color_id)
            shape_name = self.decode_shape(shape_id)
            
            # Draw "Target: " in white
            cv2.putText(
                frame[i],
                "Target: ",
                (10, 60),  # position
                cv2.FONT_HERSHEY_SIMPLEX,  # font
                1.0,  # font scale
                (255, 255, 255),  # white color
                2,  # thickness
                cv2.LINE_AA
            )
            
            # Get the size of "Target: " text
            (text_width, text_height), _ = cv2.getTextSize(
                "Target: ", 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1.0, 
                2
            )
            # Draw color name in its corresponding color
            # Use self.shape_color_dict to get the color
            color = self._env.shape_color_dict[color_id]['color'][:3] * 255  # convert to BGR
            cv2.putText(
                frame[i],
                f"{color_name} ",
                (10 + text_width, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                color,  # use the color from shape_color_dict
                2,
                cv2.LINE_AA
            )
            
            # Get width of color text
            (color_width, _), _ = cv2.getTextSize(
                f"{color_name} ", 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1.0, 
                2
            )
            
            # Draw shape name in white
            cv2.putText(
                frame[i],
                shape_name,
                (10 + text_width + color_width, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),  # white color
                2,
                cv2.LINE_AA
            )
            
        return frame
    




class MemoryCapacityInfoWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.step_count = 0
        self.current_obs = None
        self.info = None

        self._env = env
        self.color_dict = self._env.color_dict
        self.colors_names = {
            0: "Red",
            1: "Green",
            2: "Blue",
            3: "Yellow",
            4: "Magenta",
            5: "Cyan",
            6: "Maroon",
            7: "Olive",
            8: "Teal"
        }
        
    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.info = info
        self.step_count = info['elapsed_steps'].detach().cpu().numpy()
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self.info = info
        self.step_count = info['elapsed_steps'].detach().cpu().numpy()
        return obs, reward, terminated, truncated, info
    
    def render(self):
        frame = self.env.render()
        if torch.is_tensor(frame):
            frame = frame.detach().cpu().numpy()

        for i in range(len(frame)):
            seq_of_cubes = self.info["oracle_info"][i].detach().cpu().numpy()
            touched_cubes = self._env.touched_cubes[i].detach().cpu().numpy()
            
            # Draw "Target: " in white
            cv2.putText(
                frame[i],
                "Target: ",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )
            
            # Get the width of "Target: " text
            (text_width, _), _ = cv2.getTextSize(
                "Target: ", 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1.0, 
                2
            )
            
            current_x = 10 + text_width
            
            # # Draw each color circle with outline based on touch status
            # for color_id in seq_of_cubes:
            #     # Draw filled circle in cube's color
            #     cv2.circle(
            #         frame[i],
            #         (current_x + 10, 55),  # center point
            #         10,  # radius
            #         self.color_dict[color_id][:3] * 255,  # color
            #         -1  # filled
            #     )
                
            #     # Draw white outline if touched, black if not
            #     outline_color = (255, 255, 255) if touched_cubes[color_id] else (0, 0, 0)
            #     cv2.circle(
            #         frame[i],
            #         (current_x + 10, 55),  # center point
            #         10,  # radius
            #         outline_color,  # outline color
            #         2  # thickness
            #     )
                
            #     current_x += 30  # space between circles


            # Draw each color square with outline based on touch status
            for color_id in seq_of_cubes:
                square_size = 15
                x1 = current_x
                y1 = 45
                x2 = x1 + square_size
                y2 = y1 + square_size
                
                # Draw filled square in cube's color
                cv2.rectangle(
                    frame[i],
                    (x1, y1),  # top-left point
                    (x2, y2),  # bottom-right point
                    self.color_dict[color_id][:3] * 255,  # color
                    -1  # filled
                )
                
                # Draw white outline if touched, black if not
                outline_color = (255, 255, 255) if touched_cubes[color_id] else (0, 0, 0)
                cv2.rectangle(
                    frame[i],
                    (x1, y1),  # top-left point
                    (x2, y2),  # bottom-right point
                    outline_color,  # outline color
                    2  # thickness
                )
                
                current_x += square_size + 10  # space between squares
                
        return frame
