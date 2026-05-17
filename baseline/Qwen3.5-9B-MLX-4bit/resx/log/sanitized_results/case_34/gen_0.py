

def unique(l: list):
    """Return sorted unique elements in a list
    >>> unique([5, 3, 5, 2, 3, 3, 9, 0, 123])
    [0, 2, 3, 5, 9, 123]
    """
    import re

    return sorted(set(l))

def unique_from_lists(lists: list[list]) -> list:
    """Given a list of lists, return a list of sorted unique elements from all sublists. Each sublist may contain duplicate elements,

assert unique_from_lists([[1, 2, 2], [3, 3, 4], [5, 6, 6]]) == [1, 2, 3, 4, 5, 6]
assert unique_from_lists([[10, 10], [20, 30, 20], [40, 40, 50]]) == [10, 20, 30, 40, 50]
assert unique_from_lists([[1, 1, 1], [1, 1], [1]]) == [1]