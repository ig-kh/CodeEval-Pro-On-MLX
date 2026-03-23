import json
import argparse
import re
from pathlib import Path


# Standard modules commonly used in HumanEval tasks
STD_MODULES = {
    'math': ['math.', 'sqrt', 'sin', 'cos', 'tan', 'log', 'exp', 'ceil', 'floor'],
    'random': ['random.', 'randint', 'choice', 'shuffle'],
    'itertools': ['itertools.', 'chain', 'combinations', 'permutations', 'product'],
    'collections': ['collections.', 'Counter', 'defaultdict', 'deque', 'OrderedDict'],
    'functools': ['functools.', 'lru_cache', 'reduce', 'partial'],
    'heapq': ['heapq.', 'heappush', 'heappop', 'heapify'],
    'bisect': ['bisect.', 'bisect_left', 'bisect_right'],
    're': ['re.', 'search', 'match', 'findall', 'sub'],
}


def add_missing_imports(code: str) -> str:
    """Adds missing imports of standard libraries and typing."""
    lines = code.split('\n')
    imported_modules = set()
    for line in lines:
        match = re.match(r'^\s*import\s+([\w\.]+)', line)
        if match:
            imported_modules.add(match.group(1).split('.')[0])
        match = re.match(r'^\s*from\s+([\w\.]+)\s+import', line)
        if match:
            imported_modules.add(match.group(1).split('.')[0])

    used_modules = set()
    typing_types = {'List', 'Tuple', 'Dict', 'Set', 'Optional', 'Union', 'Any', 'Callable'}
    for type_name in typing_types:
        if re.search(rf'\b{type_name}\b', code):
            used_modules.add('typing')
            break
    for mod, patterns in STD_MODULES.items():
        for pat in patterns:
            if re.search(rf'\b{re.escape(pat)}', code):
                used_modules.add(mod)
                break

    imports_to_add = []
    for mod in sorted(used_modules):
        if mod not in imported_modules:
            if mod == 'typing':
                types_used = [t for t in typing_types if re.search(rf'\b{t}\b', code)]
                if types_used:
                    imports_to_add.append(f"    from typing import {', '.join(sorted(types_used))}")
                else:
                    imports_to_add.append("    import typing")
            else:
                imports_to_add.append(f"    import {mod}")

    if not imports_to_add:
        return code

    insert_pos = 0
    for i, line in enumerate(lines):
        if re.match(r'^\s*(import|from)\s+', line):
            insert_pos = i + 1
        else:
            if insert_pos > 0 and line.strip() != '':
                break
    new_lines = lines[:insert_pos] + imports_to_add + [''] + lines[insert_pos:]
    return '\n'.join(new_lines)


def normalize_indents(code: str) -> str:
    lines = code.split('\n')
    result = []
    first_found = False

    for line in lines:
        stripped = line.lstrip(' ')
        if not stripped:
            result.append('')
            continue

        # Normalize indentation for the remaining lines
        indent = len(line) - len(stripped)
        if indent % 4 != 0:
            new_indent = ((indent // 4) + 1) * 4
            line = ' ' * new_indent + stripped
        result.append(line)

    return '\n'.join(result)


def remove_example_lines(code: str) -> str:
    """Removes lines starting with >>> or print( (often added by the model)."""
    lines = []
    for line in code.split('\n'):
        stripped = line.lstrip()
        if stripped.startswith(('>>>', 'print(')):
            continue
        lines.append(line)
    return '\n'.join(lines)


def truncate_after_code_block(code: str) -> str:
    """Truncates code after a closing ``` (if any)."""
    code_blocks = re.findall(r'```(?:python)?\s*\n(.*?)\n```', code, re.DOTALL)
    if code_blocks:
        return code_blocks[0].strip()
    idx = code.find('```')
    if idx != -1:
        return code[:idx].strip()
    return code.strip()


def post_process_humaneval(output: str) -> str:
    """Main code sanitization function."""
    if not output:
        return ""

    # 1. Trim extra text
    for marker in ['if __name__', '# Test', '# Example']:
        if marker in output:
            output = output.split(marker, 1)[0]

    # 2. Extract code from markdown blocks
    # output = truncate_after_code_block(output)

    # # 3. Remove example lines
    # output = remove_example_lines(output)

    # # 4. Normalize indentation
    output = normalize_indents(output)

    # # 5. Add missing imports
    output = add_missing_imports(output)

    # # 6. Remove extra blank lines
    # output = output.strip()

    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True, help='Model name')
    parser.add_argument('--source_path', type=str, required=True, help='Path to the log directory.')
    args = parser.parse_args()

    source_dir = Path(args.source_path)
    input_file = source_dir / 'results.jsonl'
    output_file = source_dir / 'sanitized_results.jsonl'

    with open(input_file, 'r') as f:
        ds = [json.loads(line) for line in f]

    with open(output_file, 'w') as f:
        for d in ds:
            if 'completion' in d:
                d['completion'] = post_process_humaneval(d['completion'])
            if 'solution' in d:
                d['solution'] = post_process_humaneval(d['solution'])
            f.write(json.dumps(d) + '\n')

    print(f"Sanitized results saved to {output_file}")