from typing import List


def all_prefixes(string: str) -> List[str]:
    """ Return list of all prefixes from shortest to longest of the input string
    >>> all_prefixes('abc')
    ['a', 'ab', 'abc']
    """
    import re
    from typing import List

    prefixes = []
    for i in range(1, len(string) + 1):
        prefixes.append(string[:i])

    return prefixes


def all_prefixes_list(strings: List[str]) -> List[List[str]]:
    """ Given a list of strings, return a list of lists where each sublist contains all prefixes of the corresponding string in the input list, sorted from shortest to longest. If the input list is empty, return an empty list.
    """
    if not strings:
        return []

    result = []
    for s in strings:
        prefixes = all_prefixes(s)
        result.append(prefixes)

    return result


assert all_prefixes_for_list(['abc', 'def']) == [['a', 'ab', 'abc'], ['d', 'de', 'def']]
assert all_prefixes_for_list(['']) == [[]]
assert all_prefixes_for_list([]) == []
assert all_prefixes_for_list(['single']) == [['s', 'si', 'sin', 'sing', 'singl', 'single']]