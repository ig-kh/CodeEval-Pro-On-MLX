import json
import os
import shutil

input_file = "dataset/depy_sql.json"
output_dir = "./gt/dummy"
output_file = os.path.join(output_dir, "results.jsonl")
output_sanitized_file = os.path.join(output_dir, "sanitized_results.jsonl")

# This is the utilitary script to produce runnable codes from gt labeling of depy_sql dataset
# I. First, run this script
# python creadte_gt_codes.py
# > This will generate gt codes at ./gt/dummy
# II. Then run harness against the generated codes to verify them
# python -m eval.harness \
#   --model_name ground_truth \
#   --task depy_sql \
#   --dataset_path ./dataset/depy_sql.json \
#   --source_path ./gt/dummy \
#   --save_path ./gt_eval_results \
#   --run_code
# > After all runs check you shall see the result "Result of Your sanitized Outputs : {'pass@1': np.float64(1.0)}"

with open(input_file, "r") as f:
    problems = json.load(f)

os.makedirs(output_dir, exist_ok=True)

with open(output_file, "w") as f_out:
    for prob in problems:
        raw_code = prob["raw_solution"].strip()
        new_code = prob["new_solution"].strip()

        # Merge raw and new (remove duplicate imports)
        def extract_imports(code):
            lines = code.split('\n')
            imports = []
            rest = []
            for line in lines:
                if line.startswith('import ') or line.startswith('from '):
                    imports.append(line)
                else:
                    rest.append(line)
            return imports, '\n'.join(rest)

        raw_imports, raw_body = extract_imports(raw_code)
        new_imports, new_body = extract_imports(new_code)
        all_imports = list(dict.fromkeys(raw_imports + new_imports))
        merged_code = '\n'.join(all_imports) + '\n\n' + raw_body + '\n\n' + new_body

        # Use 'solution' key as a string
        record = {
            "task_id": prob["id"],
            "solution": merged_code
        }
        f_out.write(json.dumps(record) + "\n")

shutil.copy(output_file, output_sanitized_file)