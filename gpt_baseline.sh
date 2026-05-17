set -ex
WORK_DIR=/Users/igorkharchikov/Codes/Thesis26/CodeEval-Pro-On-MLX/depy_sql_baselines/result
MODEL=GPT-4

TASK_TYPE=depy_sql      
mkdir -p ${WORK_DIR}/${MODEL}/${TASK_TYPE}/outputs/

python -m run_api \
  --model_name gpt-4 \
  --dataset $TASK_TYPE \
  --save_path ${WORK_DIR}/${MODEL}/${TASK_TYPE}/outputs/results.jsonl \
  --api_key  $OPENAI_API_KEY \
  --base_url https://api.openai.com/v1


python -m sanitize \
    --model_name $MODEL \
    --source_path ${WORK_DIR}/${MODEL}/${TASK_TYPE}/outputs/ \
    
python -m harness \
    --model_name $MODEL \
    --task $TASK_TYPE \
    --dataset_path ./dataset/${TASK_TYPE}.json \
    --source_path ${WORK_DIR}/${MODEL}/${TASK_TYPE}/outputs/ \
    --save_path ${WORK_DIR}/${MODEL}/${TASK_TYPE} \
    --run_code