
action_chunking_size = 8

echo "===RememberColor3 ==="
python experiments/robot/mikasa/run_mikasa_eval.py \
  --pretrained_checkpoint "" \
  --language_instruction "Remember the color of the cube and then pick the matching one" \
  --env_id "RememberColor3-v0" \
  --num_eval_steps 60 \
  --num_eval_episodes 100 \
  --num_open_loop_steps $action_chunking_size --run_id_note "action_chunk=$action_chunking_size"

echo "===RememberColor5 ==="
python experiments/robot/mikasa/run_mikasa_eval.py \
  --pretrained_checkpoint "" \
  --language_instruction "Remember the color of the cube and then pick the matching one" \
  --env_id "RememberColor5-v0" \
  --num_eval_steps 60 \
  --num_eval_episodes 100 \
  --num_open_loop_steps $action_chunking_size --run_id_note "action_chunk=$action_chunking_size"

echo "===RememberColor9 ==="
python experiments/robot/mikasa/run_mikasa_eval.py \
  --pretrained_checkpoint "" \
  --language_instruction "Remember the color of the cube and then pick the matching one" \
  --env_id "RememberColor9-v0" \
  --num_eval_steps 60 \
  --num_eval_episodes 100 \
  --num_open_loop_steps $action_chunking_size --run_id_note "action_chunk=$action_chunking_size"


echo "===ShellGameTouch ==="
python experiments/robot/mikasa/run_mikasa_eval.py \
  --pretrained_checkpoint "" \
  --language_instruction "Memorize the position of the cup covering the ball, then pick that cup" \
  --env_id "ShellGameTouch-v0" \
  --num_eval_steps 90 \
  --num_eval_episodes 100 \
  --num_open_loop_steps $action_chunking_size --run_id_note "action_chunk=$action_chunking_size"

echo "===InterceptMedium ==="
python experiments/robot/mikasa/run_mikasa_eval.py \
  --pretrained_checkpoint "" \
  --language_instruction "Track the ball’s movement, estimate its velocity, then aim the ball at the target" \
  --env_id "InterceptMedium-v0" \
  --num_eval_steps 90 \
  --num_eval_episodes 100 \
  --num_open_loop_steps $action_chunking_size --run_id_note "action_chunk=$action_chunking_size"

echo "=== Done ==="
