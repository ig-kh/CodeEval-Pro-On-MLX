set -ex
WORK_DIR=/Users/igorkharchikov/Codes/Thesis26/CodeEval-Pro-On-MLX/depy_sql_baselines/result
# MODEL=QWEN35-9B-Q4-1k
# MODEL=QWEN35-9B-Q4-2k
# MODEL=QWEN35-9B-Q4-3k
# MODEL=QWEN35-9B-Q4-?k-hot
MODEL=QWEN35-9B-Q4-8k

TASK_TYPE=depy_sql      
mkdir -p ${WORK_DIR}/${MODEL}/${TASK_TYPE}/outputs/

python -m eval.inference \
  --use_mlx \
  --model_name_or_path mlx-community/Qwen3.5-9B-MLX-4bit \
  --dataset $TASK_TYPE \
  --save_path ${WORK_DIR}/${MODEL}/${TASK_TYPE}/outputs/results.jsonl \
  --n_batches 17 \
  --n_problems_per_batch 4 \
  --n_samples_per_problem 1 \
  --max_new_tokens 8192 \
  --temperature 0.1 \
  --top_p 0.95 \
  --do_sample \
  --repetition_penalty 1.05 \
  --lazy \
  --max_kv_size 8192
### OTHER OPTIONS CHECKED DURING BASELINE EVAL ###
  # --max_new_tokens 2048 \
  # --max_new_tokens 3072 \
  # --temperature 0.5 \

python -m sanitize \
    --model_name $MODEL \
    --source_path ${WORK_DIR}/${MODEL}/${TASK_TYPE}/outputs/ \

python -m strong_sanitize \
    --model_name $MODEL \
    --source_path ${WORK_DIR}/${MODEL}/${TASK_TYPE}/outputs/ \

python -m harness \
    --model_name $MODEL \
    --task $TASK_TYPE \
    --dataset_path ./dataset/${TASK_TYPE}.json \
    --source_path ${WORK_DIR}/${MODEL}/${TASK_TYPE}/outputs/ \
    --save_path ${WORK_DIR}/${MODEL}/${TASK_TYPE} \
    --run_code