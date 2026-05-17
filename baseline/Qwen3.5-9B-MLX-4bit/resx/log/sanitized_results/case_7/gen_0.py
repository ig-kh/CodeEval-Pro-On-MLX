from typing import List


def filter_by_substring(strings: List[str], substring: str) -> List[str]:
    """ Filter an input list of strings only for ones that contain given substring
    >>> filter_by_substring([], 'a')
    []
    >>> filter_by_substring(['abc', 'bacd', 'cde', 'array'], 'a')
    ['abc', 'bacd', 'array']
    """
    import re
    from typing import List

    result = []

    for string in strings:
        if substring in string:
            result.append(string)

    return result


def filter_by_multiple_substrings(strings: List[str], substrings: List[str]) -> List[str]:
    """ Given a list of strings and a list of substrings, return a set of strings that contain at least one of the

assert filter_by_multiple_substrings([], ['a']) == set([])
assert filter_by_multiple_substrings(['abc', 'bacd', 'cde', 'array'], ['a', 'd']) == set(['abc', 'bacd', 'cde', 'array'])
assert filter_by_multiple_substrings(['hello', 'world', 'python', 'programming'], ['o', 'n']) == set(['hello', 'world', 'python', 'programming'])
assert filter_by_multiple_substrings(['apple', 'banana', 'cherry'], ['p', 'a']) == set(['apple', 'banana'])