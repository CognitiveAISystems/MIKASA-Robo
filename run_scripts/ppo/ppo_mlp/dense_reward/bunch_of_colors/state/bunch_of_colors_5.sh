#!/bin/bash

for seed in 123 231 321
do
    echo "Running experiment with seed $seed"
    python3 baselines/ppo/ppo_memtasks.py \
        --env_id=BunchOfColors5-v0 \
        --exp-name=ppo-mlp-state-dense-bunch-of-colors-5-v0 \
        --capture-video \
        --save-model \
        --track \
        --num-steps=120 \
        --num_eval_steps=360 \
        --include-state \
        --seed=$seed \
        --total-timesteps=75_000_000 \
        --no-finite-horizon-gae \
        --eval-freq=25 \
        --gamma=0.9 \
        --update-epochs=4
        
done