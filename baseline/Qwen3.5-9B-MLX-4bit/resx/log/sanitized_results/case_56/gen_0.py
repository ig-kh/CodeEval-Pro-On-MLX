

def correct_bracketing(brackets: str):
    """ brackets is a string of "<" and ">".
    return True if every opening bracket has a corresponding closing bracket.

    >>> correct_bracketing("<")
    False
    >>> correct_bracketing("<>")
    True
    >>> correct_bracketing("<<><>>")
    True
    >>> correct_bracketing("><<>")
    False
    """
    count = 0
    for ch in brackets:
        if ch == "<":
            count += 1
        elif ch == ">":
            count -= 1
            if count < 0:
                return False
    return count == 0


def check_bracket_strings(bracket_list: list) -> list:
    """
    Given a list of bracket strings, determine if each string in the list is correctly bracketed and then return a

assert check_bracket_strings(['<>', '<<><>>', '><<>']) == [True, True, False]
assert check_bracket_strings(['<', '>', '<<>>']) == [False, False, True]
assert check_bracket_strings([]) == []
assert check_bracket_strings(['<><>', '><><']) == [True, False]