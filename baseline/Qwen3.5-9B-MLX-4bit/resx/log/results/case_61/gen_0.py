

def correct_bracketing(brackets: str):
    """ brackets is a string of "(" and ")".
    return True if every opening bracket has a corresponding closing bracket.

    >>> correct_bracketing("(")
    False
    >>> correct_bracketing("()")
    True
    >>> correct_bracketing("(()())")
    True
    >>> correct_bracketing(")(()")
    False
    """
    count = 0
    for char in brackets:
        if char == "(":
            count += 1
        elif char == ")":
            count -= 1
        if count < 0:
            return False
    return count == 0

def check_bracket_strings(bracket_strings: list) -> list:
    """check_bracket_strings is a function that takes a list of bracket strings and returns a list of booleans indicating whether each string is correctly bracketed.
    >>> check_bracket_strings(["()", "(()())", ")"])
    [True, True, False]
    >>> check_bracket_strings(["(", ")", ""])
    [False, False, True]
    >>> check_bracket_strings(["(()())", ")(()"])
    [True, False]
    >>> check_bracket

assert check_bracket_strings(['(', '()', '(()())', ')()']) == [False, True, True, False]