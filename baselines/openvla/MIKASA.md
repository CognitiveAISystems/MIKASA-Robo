# OpenVLA-OFT

This section provides instructions for fine-tuning and evaluating the OpenVLA-OFT model.

## Dataset Collection for MIKASA-Robo

To begin, collect datasets for each MIKASA-Robo task:

1. Follow the dataset collection instructions provided in the original repository.
2. Don't forget to set `--control_mode='pd_ee_delta_pose'` in `get_dataset_collectors_ckpt.py`

## Conversion to RLDS Format

1. Install conda enviroment for RLDS dataset builder:

```bash

cd openvla/rlds_dataset_builder
conda env create -f environment_ubuntu.yml
```

2. Add data paths to scritp mikasa_dataset_dataset_builder.py:

* add .npz dataset path in function `_split_generators`, for example: `self._generate_examples(path='MIKASA-Robo/unbatched/RememberColor9-v0/train_data_*.npz')`
* add task language instruction in function `_parse_example`: 

3. To convert .npz dataset to rlds, execute the following command:

```bash

conda activate rlds_env

cd openvla/rlds_dataset_builder/mikasa_dataset
tfds build --overwrite

cd ../../
mkdir -p datasets
mv -T ~/tensorflow_datasets/mikasa_dataset datasets/remember_color_9/mikasa_dataset
```

## Fine-tuning OpenVLA-OFT

1. Set up the conda environment and install all required packages by running:

```bash
bash openvla/env_setup.sh
```

2. Execute the fine-tuning script (`openvla/vla-scripts/finetune.py`) with your specific dataset paths and parameters. Modify `data_root_dir`, `run_root_dir`, and adjust `--nproc-per-node` to match your GPU availability:

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir "datasets/remember_color_9" \
  --dataset_name mikasa_dataset \
  --run_root_dir "mikasa_finetune_remember_color_9" \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio False \
  --batch_size 8 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 100000 \
  --max_steps 100_000 \
  --save_freq 10000 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 32 \
  --run_id_note parallel_dec--8_acts_chunk--continuous_acts--L1_regression--base_img--wrist_img
```

## OpenVLA Evaluation

For evaluation on all environments reported in the paper, run the following evaluation script, replacing `pretrained_checkpoint` with the path to your trained model checkpoint:

```bash
bash openvla/eval_all.sh
```