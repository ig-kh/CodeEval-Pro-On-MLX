import json
import argparse
import re
import ast
from pathlib import Path

# Common standard modules and their typical symbols
STD_MODULES = {
    'math': ['math.', 'sqrt', 'sin', 'cos', 'tan', 'log', 'exp', 'ceil', 'floor'],
    'random': ['random.', 'randint', 'choice', 'shuffle'],
    'itertools': ['itertools.', 'chain', 'combinations', 'permutations', 'product'],
    'collections': ['collections.', 'Counter', 'defaultdict', 'deque', 'OrderedDict'],
    'functools': ['functools.', 'lru_cache', 'reduce', 'partial'],
    'heapq': ['heapq.', 'heappush', 'heappop', 'heapify'],
    'bisect': ['bisect.', 'bisect_left', 'bisect_right'],
    're': ['re.', 'search', 'match', 'findall', 'sub'],
    'sqlite3': ['sqlite3.', 'fetchone', 'cursor', 'fetchall', 'execute'],
    'json': ['json.', 'dumps', 'loads'],
}

TYPING_TYPES = {'List', 'Tuple', 'Dict', 'Set', 'Optional', 'Union', 'Any', 'Callable'}


def extract_code_from_markdown(text: str) -> str:
    """Extract Python code from markdown code blocks."""
    # Find ```python ... ``` blocks
    pattern = r'```(?:python)?\s*\n(.*?)\n```'
    blocks = re.findall(pattern, text, re.DOTALL)
    if blocks:
        return '\n\n'.join(blocks)
    # If no block, try to find any ``` ... ``` block
    blocks = re.findall(r'```\s*\n(.*?)\n```', text, re.DOTALL)
    if blocks:
        return '\n\n'.join(blocks)
    # Fallback: return whole text (assume it's code)
    return text


def remove_explanatory_text(code: str) -> str:
    """Remove common prefixes like 'Here is the code:' and trailing explanations."""
    # Remove lines before the first 'def ' or 'import ' or 'from '
    lines = code.splitlines()
    start_idx = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(('def ', 'import ', 'from ', '@')):
            start_idx = i
            break
    code = '\n'.join(lines[start_idx:])

    # Remove trailing text after certain markers
    markers = ['if __name__', '# Test', '# Example', '```', '"""Example', "'''Example"]
    for marker in markers:
        if marker in code:
            code = code.split(marker, 1)[0]

    # Remove lines that start with '>>>' or '...' (interactive prompts)
    lines = []
    for line in code.splitlines():
        stripped = line.strip()
        if stripped.startswith(('>>>', '...', 'print(')) and not stripped.startswith('print('):
            # Actually keep print('...') if it's inside a function? Safer to remove only >>> and ...
            continue
        lines.append(line)
    code = '\n'.join(lines)

    # Remove trailing backticks
    code = re.sub(r'```\s*$', '', code)
    return code.strip()


def fix_indentation(code: str) -> str:
    """
    Convert 3-space indentation to 4 spaces, ensure consistent spacing,
    and fix lines that start with spaces but should be at column 0.
    """
    lines = code.splitlines()
    fixed = []
    for line in lines:
        if not line.strip():
            fixed.append('')
            continue

        # Remove any leading spaces that are exactly 3 and replace with 4
        # But only if the line contains code (not just spaces)
        if line.startswith('   ') and not line.startswith('    '):
            line = '    ' + line[3:]

        # Ensure that lines starting with 'def ', 'class ', 'if __name__' have no leading spaces?
        # Actually, they can be indented if inside a class. We'll trust the model, but remove
        # leading spaces from top-level constructs if they are the first line.
        # We'll not force column 0 – that could break nested definitions.
        fixed.append(line)

    # Join back
    return '\n'.join(fixed)


def add_missing_imports(code: str) -> str:
    """Add missing imports for standard modules and typing (no indentation)."""
    lines = code.splitlines()

    # Find existing imports
    imported_modules = set()
    for line in lines:
        m = re.match(r'^\s*import\s+([\w\.]+)', line)
        if m:
            imported_modules.add(m.group(1).split('.')[0])
        m = re.match(r'^\s*from\s+([\w\.]+)\s+import', line)
        if m:
            imported_modules.add(m.group(1).split('.')[0])

    used_modules = set()
    # Check for typing
    for typ in TYPING_TYPES:
        if re.search(rf'\b{typ}\b', code):
            used_modules.add('typing')
            break
    # Check for standard modules
    for mod, patterns in STD_MODULES.items():
        for pat in patterns:
            if re.search(rf'\b{re.escape(pat)}', code):
                used_modules.add(mod)
                break

    imports_to_add = []
    for mod in sorted(used_modules):
        if mod not in imported_modules:
            if mod == 'typing':
                # Find which types are actually used
                used_types = [t for t in TYPING_TYPES if re.search(rf'\b{t}\b', code)]
                if used_types:
                    imports_to_add.append(f"from typing import {', '.join(sorted(used_types))}")
                else:
                    imports_to_add.append("import typing")
            else:
                imports_to_add.append(f"import {mod}")

    if not imports_to_add:
        return code

    # Insert imports at the top (first line of code, after any shebang or encoding comment)
    insert_idx = 0
    for i, line in enumerate(lines):
        if line.startswith('#!') or line.startswith('# -*- coding:'):
            insert_idx = i + 1
        else:
            break
    new_lines = lines[:insert_idx] + imports_to_add + [''] + lines[insert_idx:]
    return '\n'.join(new_lines)


def remove_extra_blank_lines(code: str) -> str:
    """Replace multiple consecutive blank lines with at most one."""
    return re.sub(r'\n\s*\n', '\n\n', code)


def validate_syntax(code: str, fix_missing_def: bool = False) -> str:
    """
    Check if the code can be parsed. If fix_missing_def is True, attempt
    to prepend a dummy function definition? Better not – just return as is.
    We'll only raise a warning but not modify.
    """
    try:
        ast.parse(code)
    except SyntaxError as e:
        # Optionally log, but don't alter code
        print(f"Syntax warning: {e}", file=sys.stderr)
    return code


def post_process(output: str, validate: bool = False) -> str:
    """Main sanitization pipeline."""
    if not output:
        return ""

    # 1. Extract code from markdown blocks
    output = extract_code_from_markdown(output)

    # 2. Remove explanatory text and prompt leftovers
    output = remove_explanatory_text(output)

    # 3. Fix indentation (3->4 spaces)
    output = fix_indentation(output)

    # 4. Add missing imports
    output = add_missing_imports(output)

    # 5. Remove extra blank lines
    output = remove_extra_blank_lines(output)

    # 6. Optional syntax validation (no modification)
    if validate:
        output = validate_syntax(output)

    return output.strip()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Enhanced code sanitizer for model outputs.')
    parser.add_argument('--model_name', type=str, required=True, help='Model name (unused but required for compatibility)')
    parser.add_argument('--source_path', type=str, required=True, help='Path to the directory containing results.jsonl')
    parser.add_argument('--validate', action='store_true', help='Check syntax after sanitization (prints warnings)')
    args = parser.parse_args()

    source_dir = Path(args.source_path)
    input_file = source_dir / 'results.jsonl'
    output_file = source_dir / 'strongly_sanitized_results.jsonl'

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    with open(input_file, 'r') as f:
        data = [json.loads(line) for line in f]

    with open(output_file, 'w') as f_out:
        for entry in data:
            if 'completion' in entry:
                entry['completion'] = post_process(entry['completion'], validate=args.validate)
            if 'solution' in entry:
                entry['solution'] = post_process(entry['solution'], validate=args.validate)
            f_out.write(json.dumps(entry) + '\n')

    print(f"Enhanced sanitized results written to {output_file}")