from typing import List


def filter_by_prefix(strings: List[str], prefix: str) -> List[str]:
    """ Filter an input list of strings only for ones that start with a given prefix.
    >>> filter_by_prefix([], 'a')
    []
    >>> filter_by_prefix(['abc', 'bcd', 'cde', 'array'], 'a')
    ['abc', 'array']
    """
    return [s for s in strings if s.startswith(prefix)]


def filter_by_multiple_prefixes(strings: List[str], prefixes: List[str]) -> List

assert filter_by_multiple_prefixes([], ['a', 'b']) == set([])
assert filter_by_multiple_prefixes(['abc', 'bcd', 'cde', 'array'], ['a', 'b']) == set(['abc', 'array', 'bcd'])
assert filter_by_multiple_prefixes(['apple', 'banana', 'cherry', 'apricot'], ['a', 'b']) == set(['apple', 'apricot', 'banana'])
assert filter_by_multiple_prefixes(['apple', 'banana', 'cherry', 'apricot'], ['c', 'd']) == set(['cherry'])
assert filter_by_multiple_prefixes(['apple', 'banana', 'cherry', 'apricot'], ['a', 'b', 'c']) == set(['apple', 'apricot', 'banana', 'cherry'])
