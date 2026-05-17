

def max_element(l: list):
    """Return maximum element in the list.
    >>> max_element([1, 2, 3])
    3
    >>> max_element([5, 3, -5, 2, -3, 3, 9, 0, 123, 1, -10])
    123
    """
    return max(l)

def max_element_in_lists(list_of_lists: list):
    """Given a list of lists, return the maximum element among all the sublists. If the list of lists is empty, return None.
    >>> max_element_in_lists([[1, 2], [3, -4, 5], [], [6, 7]])
    7
    >>> max_element_in_lists([[]])
    None
    """

assert max_element_in_lists([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) == 9
assert max_element_in_lists([[5, 3, -5, 2, -3, 3, 9, 0, 123, 1, -10], [100, 200, 300], [-1, -2, -3]]) == 300
assert max_element_in_lists([]) == None
assert max_element_in_lists([[], [1, 2], [3, 4, 5]]) == 5
assert max_element_in_lists([[1], [2], [3]]) == 3