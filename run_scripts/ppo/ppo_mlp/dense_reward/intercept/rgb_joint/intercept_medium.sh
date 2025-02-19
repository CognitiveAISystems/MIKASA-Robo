#!/bin/bash

for seed in 123 231 321
do
    echo "Running experiment with seed $seed"
    python3 baselines/ppo/ppo_memtasks.py \
        --env_id=InterceptMedium-v0 \
        --exp-name=intercept-medium-v0 \
        --capture-video \
        --save-model \
        --track \
        --num-steps=90 \
<<<<<<< HEAD:run_scripts/ppo/ppo_mlp/dense_reward/intercept/rgb_joint/intercept_medium.sh
        --num_eval_steps=270 \
        --include-rgb \
        --include-joints \
        --seed=$seed \
        --total-timesteps=10_000_000 \
        --eval-freq=25 \
        --num-envs=256 \
        --num-minibatches=8
=======
        --num_eval_steps=90 \
        --include-state \
        --seed=$seed
>>>>>>> origin/dev5:run_scripts/ppo/ppo_mlp/sparse_reward/intercept/state/intercept_medium.sh
done