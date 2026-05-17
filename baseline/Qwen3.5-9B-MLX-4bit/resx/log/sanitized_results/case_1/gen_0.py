from typing import List


def separate_paren_groups(paren_string: str) -> List[str]:
    """ Input to this function is a string containing multiple groups of nested parentheses. Your goal is to
    separate those group into separate strings and return the list of those.
    Separate groups are balanced (each open brace is properly closed) and not nested within each other
    Ignore any spaces in the input string.
    >>> separate_paren_groups('( ) (( )) (( )( ))')
    ['()', '(())', '(()())']
    """
    result = []
    current_group = ""
    balance = 0

    for char in paren_string:
        if char == ' ':
            continue

        if char == '(':
            balance += 1
            current_group += char
        elif char == ')':
            current_group += char
            balance -= 1

            if balance == 0:
                result.append(current_group)
                current_group = ""

    return result


def count_balanced_paren_groups(paren_string:

assert count_balanced_paren_groups('( ) (( )) (( )( ))') == 3
assert count_balanced_paren_groups('((())) (()) (()(()))') == 3
assert count_balanced_paren_groups('()()()') == 3
assert count_balanced_paren_groups('') == 0
assert count_balanced_paren_groups('(()) ((()))') == 2