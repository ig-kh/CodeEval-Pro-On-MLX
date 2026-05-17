
def make_a_pile(n):
    """
    Given a positive integer n, you have to make a pile of n levels of stones.
    The first level has n stones.
    The number of stones in the next level is:
        - the next odd number if n is odd.
        - the next even number if n is even.
    Return the number of stones in each level in a list, where element at index
    i represents the number of stones in the level (i+1).

    Examples:
    >>> make_a_pile(3)
    [3, 5, 7]
    """
    import re

    result = []
    current = n

    for _ in range(n):
        result.append(current)
        if current % 2 == 0:
            current += 1
        else:
            current += 2
    return result

def create_pyramid(n):
    """
    Given a positive integer n, you have to create a pyramid of stones. The pyramid consists of multiple piles, each pile representing a level in the pyramid. The first pile has 1 stone, the second pile has 2 stones, and so on up to the nth pile which has n stones. Each pile must be constructed according to the 'make_a_pile' function. Return a list of lists where each sublist represents the stones in each level of the pyramid.
    """
    result = []
    for i in range(1, n + 1):
        result.append(make_a_pile(i))
    return result


assert create_pyramid(1) == [[1]]
assert create_pyramid(2) == [[1], [2, 4]]
assert create_pyramid(3) == [[1], [2, 4], [3, 5, 7]]
assert create_pyramid(4) == [[1], [2, 4], [3, 5, 7], [4, 6, 8, 10]]