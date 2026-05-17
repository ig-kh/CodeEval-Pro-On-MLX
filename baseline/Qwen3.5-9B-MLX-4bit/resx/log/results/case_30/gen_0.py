

def get_positive(l: list):
    """Return only positive numbers in the list.
    >>> get_positive([-1, 2, -4, 5, 6])
    [2, 5, 6]
    >>> get_positive([5, 3, -5, 2, -3, 3, 9, 0, 123, 1, -10])
    [5, 3, 2, 3, 9, 123, 1]
    """
    return [x for x in l if x > 0]

def get_all_positives(list_of_lists: list) -> list:
    """Given a list of lists, return a list containing only the positive numbers from each sublist. If a sublist is empty, it should be ignored in the final result. The order of the numbers should be maintained as they have appeared in the original lists.
    >>> get_all_positives([[1, 2], [3, -4, 5], [], [6, 7]])
    [1, 2, 3, 5, 6, 7]
    >>> get_all_positives([[1], [], [2, 3]])
    [1, 2, 3]
    >>> get_all_positives([[], [], []])
    []
    """
    result = []
    for sublist in list_of_lists:
        if len(sublist) > 0:
            result.extend(get_positive(sublist))
    return result


assert get_all_positives([[-1, 2, -4, 5, 6], [5, 3, -5, 2, -3, 3, 9, 0, 123, 1, -10]]) == [2, 5, 6, 5, 3, 2, 3, 9, 123, 1]
assert get_all_positives([[1, 2, 3], [], [4, -5, 6], [-7, -8, -9]]) == [1, 2, 3, 4, 6]
assert get_all_positives([[], [10, -10], [], [11, -11, 12]]) == [10, 11, 12]
assert get_all_positives([[0, -1, -2], [-3, -4, -5], [-6, -7, -8]]) == []