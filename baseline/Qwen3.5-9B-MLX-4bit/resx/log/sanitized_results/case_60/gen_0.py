

def sum_to_n(n: int):
    """sum_to_n is a function that sums numbers from 1 to n.
    >>> sum_to_n(30)
    465
    >>> sum_to_n(100)
    5050
    >>> sum_to_n(5)
    15
    >>> sum_to_n(10)
    55
    >>> sum_to_n(1)
    1
    """
    return n * (n + 1) // 2

def sum_of_sums_to_n(lst: list):
    """sum_of_sums_to_n is a function that returns the sum of the sums of numbers from 1 to each integer in the list.
    >>> sum_of_sums_to_n([3, 5])
    30
    >>> sum_of_sums_to_n([1, 2, 3])
    9
    >>> sum_of_sums_to_n([])
    0
    >>> sum_of_sums_to_n([10, 20])
    1240
    >>> sum_of_sums to_n([5, 5])
    30
    """
    return sum(sum_to_n(x) for x in lst)


assert sum_of_sums_to_n([3, 5]) == 21
assert sum_of_sums_to_n([10, 20]) == 265
assert sum_of_sums_to_n([1, 2, 3]) == 10
assert sum_of_sums_to_n([5, 10, 15]) == 190
assert sum_of_sums_to_n([100, 200, 300]) == 70300