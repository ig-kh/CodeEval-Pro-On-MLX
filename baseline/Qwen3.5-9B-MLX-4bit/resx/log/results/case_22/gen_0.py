from typing import List, Any


def filter_integers(values: List[Any]) -> List[int]:
    """ Filter given list of any python values only for integers
    >>> filter_integers(['a', 3.14, 5])
    [5]
    >>> filter_integers([1, 2, 3, 'abc', {}, []])
    [1, 2, 3]
    """
    return [val for val in values if isinstance(val, int) and not isinstance(val, bool)]


def filter_integers_from_nested_lists(nested_lists: List[List[Any]]) -> List[int]:
    """ Given a list of lists containing any Python values, filter out all non-integer values from each sublist and return a new list of lists containing only the integers. If a sublist becomes empty after filtering, it should not be included in the final result.
    >>> filter_integers_from_nested_lists([[1, 2, 'a'], [3.14, 5], [6, 7, 8]])
    [[1, 2], [5]]
    """
    filtered_lists = []
    for sublist in nested_lists:
        filtered_list = filter_integers(sublist)
        if len(filtered_list) > 0:
            filtered_lists.append(filtered_list)
    return filtered_lists


assert filter_nested_lists([['a', 3.14, 5], [1, 2, 'abc', {}, []]]) == [[5], [1, 2]]
assert filter_nested_lists([[1, 'b', 3], [4, 5, 'c'], [6, 7, 8]]) == [[1, 3], [4, 5], [6, 7, 8]]
assert filter_nested_lists([[1, 2, 3], ['a', 'b', 'c'], [4, 5, 6]]) == [[1, 2, 3], [4, 5, 6]]
assert filter_nested_lists([[1, 'a', 2], [], [3, 'b', 4]]) == [[1, 2], [3, 4]]
assert filter_nested_lists([[], [1, 2, 3], []]) == [[1, 2, 3]]