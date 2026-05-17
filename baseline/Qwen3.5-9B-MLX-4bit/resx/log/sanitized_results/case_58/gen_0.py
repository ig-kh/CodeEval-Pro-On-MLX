

def common(l1: list, l2: list):
    """Return sorted unique common elements for two lists.
    >>> common([1, 4, 3, 34, 653, 2, 5], [5, 7, 1, 5, 9, 653, 121])
    [1, 5, 653]
    >>> common([5, 3, 2, 8], [3, 2])
    [2, 3]

    """
    return sorted(set(l1) & set(l2))


def common_elements(lists: list):
    """
    Given a list of lists, find the sorted unique common elements across all lists. For example, given [[1, 

assert common_elements([[1, 4, 3, 34, 653, 2, 5], [5, 7, 1, 5, 9, 653, 121], [5, 3, 2, 8]]) == [5]
assert common_elements([[5, 3, 2, 8], [3, 2], [2, 3, 5]]) == [2, 3]
assert common_elements([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) == []
assert common_elements([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) == [1]