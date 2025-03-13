import subprocess
import os
import time
import torch
import psutil
import numpy as np
from typing import List, Dict
import GPUtil
import tyro
from dataclasses import dataclass
from typing import Optional

# python3 dataset_collectors/parallel_training_manager.py

class TrainingManager:
    def __init__(
        self, 
        max_parallel_processes: int = 4,
        gpu_memory_threshold: float = 0.8,  # 80% of GPU memory
        check_interval: float = 30,  # seconds
    ):
        self.max_parallel_processes = max_parallel_processes
        self.gpu_memory_threshold = gpu_memory_threshold
        self.check_interval = check_interval
        self.running_processes: Dict[str, subprocess.Popen] = {}
        
    # def get_gpu_memory_usage(self) -> float:
    #     """Returns GPU memory usage as a fraction (0-1)"""
    #     try:
    #         gpu = GPUtil.getGPUs()[0]  # Assuming single GPU
    #         return gpu.memoryUsed / gpu.memoryTotal
    #     except:
    #         return 0.0

    def get_gpu_memory_usage(self) -> List[float]:
        """Returns GPU memory usage as a list of fractions (0-1) for each GPU"""
        try:
            gpus = GPUtil.getGPUs()  # Get all available GPUs
            return [gpu.memoryUsed / gpu.memoryTotal for gpu in gpus]
        except:
            return [0.0] * len(GPUtil.getGPUs())  # Return a list of zeros for each GPU

    # def can_start_new_process(self) -> bool:
    #     """Check if we can start a new training process"""
    #     if len(self.running_processes) >= self.max_parallel_processes:
    #         return False
            
    #     gpu_usage = self.get_gpu_memory_usage()
    #     return gpu_usage < self.gpu_memory_threshold

    def can_start_new_process(self) -> bool:
        """Check if we can start a new training process considering multiple GPUs"""
        if len(self.running_processes) >= self.max_parallel_processes:
            return False

        gpu_usages = self.get_gpu_memory_usage()
        # Check if any GPU has available memory
        for usage in gpu_usages:
            if usage < self.gpu_memory_threshold:
                return True
        return False

    def run_training(self, env_ids: List[str], path_to_save_data: str, ckpt_dir: str):
        """Run training for multiple environments in parallel"""
        remaining_envs = env_ids.copy()
        
        while remaining_envs or self.running_processes:
            # Check running processes
            completed_envs = []
            for env_id, process in self.running_processes.items():
                if process.poll() is not None:  # Process finished
                    print(f"Training completed for {env_id}")
                    completed_envs.append(env_id)
                    
            # Remove completed processes
            for env_id in completed_envs:
                del self.running_processes[env_id]

            # Start new processes if possible
            while remaining_envs and self.can_start_new_process():
                env_id = remaining_envs.pop(0)
                cmd = [
                    "python3",
                    "dataset_collectors/get_mikasa_robo_datasets.py",
                    f"--env_id={env_id}",
                    f"--path-to-save-data={path_to_save_data}",
                    f"--ckpt-dir={ckpt_dir}"
                ]
            
                process = subprocess.Popen(cmd)
                self.running_processes[env_id] = process
                print(f"Started dataset collection from ckpt for {env_id}")

            time.sleep(self.check_interval)

@dataclass
class Args:
    path_to_save_data: str = "data"
    ckpt_dir: str = "."

def main():
    args = tyro.cli(Args)
    path_to_save_data = args.path_to_save_data
    ckpt_dir = args.ckpt_dir

    # List of environments to train

    env_ids = [
        "ShellGameTouch-v0",
        "ShellGamePush-v0",
        "ShellGamePick-v0",
        "InterceptSlow-v0",
        "InterceptMedium-v0",
        "InterceptFast-v0",
        "InterceptGrabSlow-v0",
        "InterceptGrabMedium-v0",
        "InterceptGrabFast-v0",
        "RotateLenientPos-v0",
        "RotateLenientPosNeg-v0",
        "RotateStrictPos-v0",
        "RotateStrictPosNeg-v0",
        "TakeItBack-v0",
        "RememberColor3-v0",
        "RememberColor5-v0",
        "RememberColor9-v0",
        "RememberShape3-v0",
        "RememberShape5-v0",
        "RememberShape9-v0",
        "RememberShapeAndColor3x2-v0",
        "RememberShapeAndColor3x3-v0",
        "RememberShapeAndColor5x3-v0",
        "BunchOfColors3-v0",
        "BunchOfColors5-v0",
        "BunchOfColors7-v0",
        "SeqOfColors3-v0",
        "SeqOfColors5-v0",
        "SeqOfColors7-v0",
        "ChainOfColors3-v0",
        "ChainOfColors5-v0",
        "ChainOfColors7-v0",
    ]

    # Initialize training manager
    manager = TrainingManager(
        max_parallel_processes=4,  # Run 2 training processes at once
        gpu_memory_threshold=0.8,  # Use up to 80% of GPU memory
        check_interval=30  # Check status every 30 seconds
    )

    # Start parallel training
    manager.run_training(env_ids, path_to_save_data, ckpt_dir)

if __name__ == "__main__":
    main()

# python3 dataset_collectors/parallel_dataset_collection_manager.py --path-to-save-data="data" --ckpt-dir="."