

def monotonic(l: list):
    """Return True is list elements are monotonically increasing or decreasing.
    >>> monotonic([1, 2, 4, 20])
    True
    >>> monotonic([1, 20, 4, 10])
    False
    >>> monotonic([4, 1, 0, -10])
    True
    """
    if len(l) <= 1:
        return True
    
    increasing = all(l[i] <= l[i+1] for i in range(len(l)-1))
    decreasing = all(l[i] >= l[i+1] for i in range(len(l)-1))
    
    return increasing or decreasing


def check_monotonic_lists(lists: list[list]) -> tuple[list[bool], bool]:
    """
    Given a list of lists, determine if each sublist is monotonic and if the entire list of lists is also monotonic when concatenated. Return a tuple where the first element is a list of boolean values indicating whether each sublist is monotonic,

assert check_monotonic_lists([[1, 2, 4, 20], [4, 1, 0, -10]]) == ([True, True], False)
assert check_monotonic_lists([[1, 20, 4, 10], [4, 1, 0, -10]]) == ([False, True], False)
assert check_monotonic_lists([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) == ([True, True, True], True)
assert check_monotonic_lists([[9, 8, 7], [6, 5, 4], [3, 2, 1]]) == ([True, True, True], True)
assert check_monotonic_lists([[1, 2, 3], [3, 2, 1], [1, 2, 3]]) == ([True, True, True], False)