from typing import List, Tuple


def rolling_max(numbers: List[int]) -> List[int]:
    """ From a given list of integers, generate a list of rolling maximum element found until given moment
    in the sequence.
    >>> rolling_max([1, 2, 3, 2, 3, 4, 2])
    [1, 2, 3, 3, 3, 4, 4]
    """
    from typing import List, Tuple

    if not numbers:
        return []

    result = []
    current_max = numbers[0]

    for num in numbers:
        if num > current_max:
            current_max = num

        result.append(current_max)

    return result


def rolling_max_across_lists(list_of_lists: List[List[int]]) -> Tuple[List[List[int]], List[int]]:
    """ Given a

assert rolling_max_across_lists([[1, 2, 3, 2, 3, 4, 2], [5, 6, 2, 8, 3, 1, 9]]) == ([[1, 2, 3, 3, 3, 4, 4], [5, 6, 6, 8, 8, 8, 9]], [5, 6, 6, 8, 8, 8, 9])
assert rolling_max_across_lists([[4, 3, 2, 1], [1, 2, 3, 4]]) == ([[4, 4, 4, 4], [1, 2, 3, 4]], [4, 4, 4, 4])
assert rolling_max_across_lists([[1], [1]]) == ([[1], [1]], [1])