from typing import List, Tuple


def sum_product(numbers: List[int]) -> Tuple[int, int]:
    """ For a given list of integers, return a tuple consisting of a sum and a product of all the integers in a list.
    Empty sum should be equal to 0 and empty product should be equal to 1.
    >>> sum_product([])
    (0, 1)
    >>> sum_product([1, 2, 3, 4])
    (10, 24)
    """
    import itertools
    import re
    from typing import List, Tuple

    total_sum = 0
    total_product = 1

    for num in numbers:
        total_sum += num
        total_product *= num

    return (total_sum, total_product)


def sum_product_of_lists(lists: List[List[int]]) -> Tuple[List[Tuple[int, int]], int, int]:
    """ Given a list of lists of integers, return a list of tuples where each tuple consists of the sum and product of the integers in the corresponding sublist. Additionally, calculate the total sum and total product of all integers across all sublists.
    >>> sum_product_of_lists([[1, 2], [3, 4]])
    ([(3, 2), (7, 12)], 10, 24)
    >>> sum_product_of_lists([[], [1]])
    ([(0, 1), (1, 1)], 1, 2)
    """
    result_tuples = []
    total_sum = 0
    total_product = 1

    for sublist in lists:
        sum_prod = sum_product(sublist)
        result_tuples.append(sum_prod)
        total_sum += sum_prod[0]
        total_product *= sum_prod[

assert sum_product_of_lists([[1, 2], [3, 4]]) == ([(3, 2), (7, 12)], 10, 24)
assert sum_product_of_lists([[0], [1], [2]]) == ([(0, 0), (1, 1), (2, 2)], 3, 0)
assert sum_product_of_lists([[], [1, 2, 3], [4, 5]]) == ([(0, 1), (6, 6), (9, 20)], 15, 120)
assert sum_product_of_lists([[1], [1, 1], [1, 1, 1]]) == ([(1, 1), (2, 1), (3, 1)], 6, 1)