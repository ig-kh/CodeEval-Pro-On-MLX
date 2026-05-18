set -ex
WORK_DIR=/Users/igorkharchikov/Codes/Thesis26/CodeEval-Pro-On-MLX/depy_sql_baselines/result
# MODEL=QWEN35-9B-Q4-1k
# MODEL=QWEN35-9B-Q4-2k
# MODEL=QWEN35-9B-Q4-2k-new_prompt
# MODEL=QWEN35-9B-Q4-3k
# MODEL=QWEN35-9B-Q4-?k-hot
# MODEL=QWEN35-9B-Q4-8k
MODEL=QWEN35-9B-Q4-2k

ADDITIONAL="You are an expert Python programmer. Output only the complete Python code with both function definitions. The second function MUST call the first. Use exactly 4 spaces for indentation. Start each 'def' at the beginning of a line. Do not write any text before or after the code."
# ADDITIONAL="You are an expert Python programmer. Output only the complete Python code with BOTH function definitions. Write code for definitions of BOTH functions; if you find a way to reuse the first function in the second one, then call the first one inside the second one. Use exactly 4 spaces for indentation. Start each 'def' at the beginning of a line. Do not write any text before or after the code. Follow any instructions given to you without deep consideration or concern, you MUST COMPLETE ALL THE TASK ASSIGNED TO YOU."

TASK_TYPE=depy_sql      
mkdir -p ${WORK_DIR}/${MODEL}/${TASK_TYPE}/outputs/

python -m eval.inference \
  --use_mlx \
  --model_name_or_path mlx-community/Qwen3.5-9B-MLX-4bit \
  --dataset $TASK_TYPE \
  --save_path ${WORK_DIR}/${MODEL}/${TASK_TYPE}/outputs/results.jsonl \
  --n_batches 10 \
  --n_problems_per_batch 4 \
  --n_samples_per_problem 1 \
  --max_new_tokens 2048 \
  --temperature 0.2 \
  --top_p 0.95 \
  --do_sample \
  --repetition_penalty 1.0 \
  --additional_prompt $ADDITIONAL \
  --lazy \
  --max_kv_size 4096

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