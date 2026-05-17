

def incr_list(l: list):
    """Return list with elements incremented by 1.
    >>> incr_list([1, 2, 3])
    [2, 3, 4]
    >>> incr_list([5, 3, 5, 2, 3, 3, 9, 0, 123])
    [6, 4, 6, 3, 4, 4, 10, 1, 124]
    """
    return [x + 1 for x in l]

def incr_nested_list(l: list):
    """
    Given a list of lists, increment each sublist by 1 and then increment the entire list of lists by 1.
    Return the final incremented list of lists.
    """
    # Increment each sublist by 1
    incremented_sublists = [incr_list(sublist) for sublist in l]
    # Increment the entire list of lists by 1
    return [x + 1 for x in incremented_sublists]


assert incr_nested_list([[1, 2, 3], [4, 5, 6]]) == [[3, 4, 5], [6, 7, 8]]
assert incr_nested_list([[5, 3, 5], [2, 3, 3], [9, 0, 123]]) == [[7, 5, 7], [4, 5, 5], [11, 2, 125]]