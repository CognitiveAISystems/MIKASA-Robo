import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

try:
    import GPUtil
except ImportError:
    GPUtil = None
import argparse

DEFAULT_ENV_IDS = [
    "ShellGameTouch-VLA-v0",
    "ShellGamePush-VLA-v0",
    "ShellGamePick-VLA-v0",
    "InterceptSlow-VLA-v0",
    "InterceptMedium-VLA-v0",
    "InterceptFast-VLA-v0",
    "InterceptGrabSlow-VLA-v0",
    "InterceptGrabMedium-VLA-v0",
    "InterceptGrabFast-VLA-v0",
    "RotateLenientPos-VLA-v0",
    "RotateLenientPosNeg-VLA-v0",
    "RotateStrictPos-VLA-v0",
    "RotateStrictPosNeg-VLA-v0",
    "TakeItBack-VLA-v0",
    "RememberColor3-VLA-v0",
    "RememberColor5-VLA-v0",
    "RememberColor9-VLA-v0",
    "RememberShape3-VLA-v0",
    "RememberShape5-VLA-v0",
    "RememberShape9-VLA-v0",
    "RememberShapeAndColor3x2-VLA-v0",
    "RememberShapeAndColor3x3-VLA-v0",
    "RememberShapeAndColor5x3-VLA-v0",
    "BunchOfColors3-VLA-v0",
    "BunchOfColors5-VLA-v0",
    "BunchOfColors7-VLA-v0",
    "SeqOfColors3-VLA-v0",
    "SeqOfColors5-VLA-v0",
    "SeqOfColors7-VLA-v0",
    "ChainOfColors3-VLA-v0",
    "ChainOfColors5-VLA-v0",
    "ChainOfColors7-VLA-v0",
    "ShellGameShuffleTouch-VLA-v0",
    "ShellGameShuffleColorLampTouch-VLA-v0",
    "ShellGameColorLampTouch-VLA-v0",
    "FindImposterColor3-VLA-v0",
    "FindImposterColor5-VLA-v0",
    "FindImposterColor9-VLA-v0",
    "FindImposterShape3-VLA-v0",
    "FindImposterShape5-VLA-v0",
    "FindImposterShape9-VLA-v0",
    "FindImposterShapeAndColor3x2-VLA-v0",
    "FindImposterShapeAndColor3x3-VLA-v0",
    "FindImposterShapeAndColor5x3-VLA-v0",
]


class DatasetCollectionManager:
    def __init__(
        self,
        max_parallel_processes: int = 4,
        gpu_memory_threshold: float = 0.8,
        check_interval: float = 20.0,
        logs_dir: str = "logs/dataset_collection",
    ):
        self.max_parallel_processes = max_parallel_processes
        self.gpu_memory_threshold = gpu_memory_threshold
        self.check_interval = check_interval
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.running_processes: Dict[str, subprocess.Popen] = {}
        self.running_gpus: Dict[str, Optional[int]] = {}

    def _get_gpu_usages(self) -> List[float]:
        if GPUtil is None:
            return []
        try:
            gpus = GPUtil.getGPUs()
            return [gpu.memoryUsed / max(gpu.memoryTotal, 1) for gpu in gpus]
        except Exception:
            return []

    def _pick_gpu(self) -> Optional[int]:
        gpu_usages = self._get_gpu_usages()
        if not gpu_usages:
            return None

        # Count running processes per GPU.
        procs_per_gpu = [0] * len(gpu_usages)
        for gpu_id in self.running_gpus.values():
            if gpu_id is not None and gpu_id < len(procs_per_gpu):
                procs_per_gpu[gpu_id] += 1

        # Pick GPU with fewest running processes (among those below memory threshold).
        candidates = [
            (idx, procs_per_gpu[idx], usage)
            for idx, usage in enumerate(gpu_usages)
            if usage < self.gpu_memory_threshold
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda x: (x[1], x[2]))  # fewest procs, then least memory
        return int(candidates[0][0])

    def _can_start_new_process(self) -> bool:
        return len(self.running_processes) < self.max_parallel_processes

    def run_collection(
        self,
        env_ids: List[str],
        path_to_save_data: str,
        ckpt_dir: str,
        num_train_data: int,
    ):
        script_path = Path(__file__).resolve().parent / "get_mikasa_robo_datasets.py"
        if not script_path.exists():
            raise FileNotFoundError(f"Dataset collector script not found: {script_path}")

        pending = list(env_ids)
        completed: List[str] = []
        failed: List[str] = []

        print(f"Will collect datasets for {len(pending)} envs")
        print(f"Max parallel processes: {self.max_parallel_processes}")

        while pending or self.running_processes:
            finished_envs: List[str] = []
            for env_id, process in self.running_processes.items():
                code = process.poll()
                if code is None:
                    continue

                gpu_id = self.running_gpus.get(env_id)
                if code == 0:
                    print(f"[DONE] {env_id} (gpu={gpu_id})")
                    completed.append(env_id)
                else:
                    print(f"[FAIL] {env_id} (gpu={gpu_id}) exit_code={code}")
                    failed.append(env_id)
                finished_envs.append(env_id)

            for env_id in finished_envs:
                self.running_processes.pop(env_id, None)
                self.running_gpus.pop(env_id, None)

            while pending and self._can_start_new_process():
                env_id = pending[0]
                gpu_id = self._pick_gpu()

                # If GPUs exist but all busy, wait.
                if self._get_gpu_usages() and gpu_id is None:
                    break

                pending.pop(0)
                cmd = [
                    sys.executable,
                    str(script_path),
                    f"--env-id={env_id}",
                    f"--path-to-save-data={path_to_save_data}",
                    f"--ckpt-dir={ckpt_dir}",
                    f"--num-train-data={num_train_data}",
                ]

                env = os.environ.copy()
                if gpu_id is not None:
                    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

                log_path = self.logs_dir / f"{env_id}.log"
                log_file = open(log_path, "w", encoding="utf-8")
                process = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)

                self.running_processes[env_id] = process
                self.running_gpus[env_id] = gpu_id
                print(f"[START] {env_id} (gpu={gpu_id}) -> {log_path}")

            time.sleep(self.check_interval)

        print("\nSummary:")
        print(f"Completed: {len(completed)}")
        print(f"Failed: {len(failed)}")
        if failed:
            print("Failed envs:", failed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-to-save-data", default="data_mikasa_robo", type=str)
    parser.add_argument("--ckpt-dir", default=".", type=str)
    parser.add_argument("--num-train-data", default=1000, type=int)
    parser.add_argument("--max-parallel-processes", default=4, type=int)
    parser.add_argument("--gpu-memory-threshold", default=0.8, type=float)
    parser.add_argument("--check-interval", default=20.0, type=float)
    parser.add_argument("--logs-dir", default="logs/dataset_collection", type=str)
    parser.add_argument("--env-ids", nargs="*", default=[])
    return parser.parse_args()


def main():
    args = parse_args()

    env_ids = args.env_ids if args.env_ids else DEFAULT_ENV_IDS

    manager = DatasetCollectionManager(
        max_parallel_processes=args.max_parallel_processes,
        gpu_memory_threshold=args.gpu_memory_threshold,
        check_interval=args.check_interval,
        logs_dir=args.logs_dir,
    )

    manager.run_collection(
        env_ids=env_ids,
        path_to_save_data=args.path_to_save_data,
        ckpt_dir=args.ckpt_dir,
        num_train_data=args.num_train_data,
    )


if __name__ == "__main__":
    main()

# Example:
# python3 mikasa_robo_suite/vla/dataset_collectors/parallel_dataset_collection_manager.py \
#   --path-to-save-data=data_mikasa_robo --ckpt-dir=. --num-train-data=250 --max-parallel-processes=4
