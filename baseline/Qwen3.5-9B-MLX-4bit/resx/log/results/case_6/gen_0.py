from typing import List


def parse_nested_parens(paren_string: str) -> List[int]:
    """ Input to this function is a string represented multiple groups for nested parentheses separated by spaces.
    For each of the group, output the deepest level of nesting of parentheses.
    E.g. (()()) has maximum two levels of nesting while ((())) has three.

    >>> parse_nested_parens('(()()) ((())) () ((())()())')
    [2, 3, 1, 3]
    """
    result = []
    
    for group in paren_string.split():
        level = 0
        max_level = 0
        
        for char in group:
            if char == '(':
                level += 1
                max_level = max(max_level, level)
            elif char == ')':
                level -= 1
        
        result.append(max_level)
    
    return result


def sum_deepest_nesting(strings: List[str]) -> List[int]:
    """ Given a list of strings, each representing multiple groups of nested parentheses separated by spaces, return the sum of the deepest levels of nesting for each group in each string. For example, if a string contains '(()()) ((())) () ((())()())', the sum of the

assert sum_deepest_nesting(['(()()) ((())) () ((())()())']) == [9]
assert sum_deepest_nesting(['() (())', '((()))']) == [3, 3]
assert sum_deepest_nesting(['(()(()))', '()']) == [3, 1]
assert sum_deepest_nesting(['']) == [0]
assert sum_deepest_nesting(['(((())))', '(()())']) == [4, 2]