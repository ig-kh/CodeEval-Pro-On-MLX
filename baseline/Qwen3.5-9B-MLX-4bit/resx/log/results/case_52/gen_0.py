

def below_threshold(l: list, t: int):
    """Return True if all numbers in the list l are below threshold t.
    >>> below_threshold([1, 2, 4, 10], 100)
    True
    >>> below_threshold([1, 20, 4, 10], 5)
    False
    """
    return all(num < t for num in l)


def all_below_threshold(ll: list, t: int):
    """
    Given a list of lists, where each sublist contains numbers, determine if all numbers in each sublist are below a given threshold. Return a list of boolean values where each value corresponds to whether all numbers in the fewest number of elements.
    """
    return [below_threshold(sublist, t) for sublist in ll]


assert all_below_threshold([[1, 2, 4, 10], [1, 20, 4, 10]], 100) == [True, True]
assert all_below_threshold([[1, 2, 4, 10], [1, 20, 4, 10]], 5) == [False, False]
assert all_below_threshold([[100, 200], [300, 400]], 50) == [False, False]
assert all_below_threshold([[1, 2, 3], [4, 5, 6]], 10) == [True, True]