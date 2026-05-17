from typing import List


def concatenate(strings: List[str]) -> str:
    """ Concatenate list of strings into a single string
    >>> concatenate([])
    ''
    >>> concatenate(['a', 'b', 'c'])
    'abc'
    """
    import math
    import re
    from typing import List

    return ''.join(strings)


def concatenate_nested_lists(nested_lists: List[List[str]]) -> str:
    """ Given a list of lists of strings, concatenate each sublist into a single string and then concatenate all the resulting strings into a single string. Return the final concatenated string.
    >>> concatenate_nested_lists([['a', 'b'], ['c', 'd']])
    'abcd'
    """
    concatenated_sublists = [concatenate(sublist) for sublist in nested_lists]
    return concatenate(concatenated_sublists)


assert concatenate_lists_of_strings([['a', 'b'], ['c', 'd'], ['e', 'f']]) == 'abcdef'
assert concatenate_lists_of_strings([['hello', 'world'], ['python', 'programming']]) == 'helloworldpythonprogramming'
assert concatenate_lists_of_strings([[], ['a'], ['b', 'c'], []]) == 'abc'
assert concatenate_lists_of_strings([[], [], []]) == ''