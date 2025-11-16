#!/bin/bash

echo $(nvidia-smi)
echo $(cat /usr/share/vulkan/icd.d/nvidia_icd.json)

# Set environment variable to skip prompts
export MS_SKIP_ASSET_DOWNLOAD_PROMPT=1

# Get home directory
HOME_DIR="$HOME"

# echo "Downloading assets..."
echo "HOME_DIR: $HOME_DIR"

# Initialize conda for the current shell session
eval "$(conda shell.bash hook)"

# Install huggingface-cli if not already installed
if ! command -v huggingface-cli &> /dev/null; then
    pip install huggingface_hub
fi

# Create directory
mkdir -p "$HOME_DIR/.maniskill/data/assets"

# Download assets using Python API instead of CLI
python3 -c "
from huggingface_hub import hf_hub_download
import os

output_file = hf_hub_download(
    repo_id='haosulab/ManiSkill2',
    filename='data/mani_skill2_ycb.zip',
    repo_type='dataset',
    local_dir='/tmp/maniskill_download'
)
print(f'Downloaded to: {output_file}')
"

# Copy the downloaded file
cp /tmp/maniskill_download/data/mani_skill2_ycb.zip "$HOME_DIR/.maniskill/data/assets/"

# Extract the zip file
unzip "$HOME_DIR/.maniskill/data/assets/mani_skill2_ycb.zip" -d "$HOME_DIR/.maniskill/data/assets/"

# Clean up temporary files
rm -rf /tmp/maniskill_download

conda init bash
source ~/.bashrc
conda create -n openvla-oft python=3.10 -y
conda activate openvla-oft
pip install torch torchvision torchaudio
pip install -e .

# Install Flash Attention 2 for training (https://github.com/Dao-AILab/flash-attention)
#   =>> If you run into difficulty, try `pip cache remove flash_attn` first
pip3 install packaging ninja
ninja --version; echo $?  # Verify Ninja --> should return exit code "0"
pip install "flash-attn==2.5.5" --no-build-isolation
