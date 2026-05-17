from typing import List, Tuple


def find_closest_elements(numbers: List[float]) -> Tuple[float, float]:
    """ From a supplied list of numbers (of length at least two) select and return two that are the closest to each
    other and return them in order (smaller number, larger number).
    >>> find_closest_elements([1.0, 2.0, 3.0, 4.0, 5.0, 2.2])
    (2.0, 2.2)
    >>> find_closest_elements([1.0, 2.0, 3.0, 4.0, 5.0, 2.0])
    (2.0, 2.0)
    """
    if len(numbers) < 2:
        raise ValueError("List must contain at least two elements.")
    
    min_diff = float('inf')
    closest_pair = (numbers[0], numbers[1])
    
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            diff = abs(numbers[i] - numbers[j])
            if diff < min_diff:
                min_diff = diff
                closest_pair = (numbers[i], numbers[j])
    
    return closest_pair


def find_closest_elements_across_lists(lists: List[List[float]]) -> Tuple[float, float]:
    """ Given a list of lists of numbers, find the pair of numbers across all lists that are the closest to each other. Return the pair in the order (smaller number,
    smaller number, larger number). If there are multiple pairs with the same smallest difference, return the first one found.
    >>> find_closest_elements_across_lists([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    (2.0, 3.0)
    """
    if not lists:
        raise ValueError("Input list of lists must not be empty.")
    
    # Flatten the list of lists into a single list
    flattened_list = []
    for sublist in lists:
        if len(sublist) < 2:
            raise ValueError("Each sublist must contain at least two elements.")
        flattened_list.extend(sublist)
    
    # Find the closest pair in the flattened list
    return find_closest_elements(flattened_list)


assert find_closest_pair_across_lists([[1.0, 2.0, 3.0], [4.0, 5.0, 2.2]]) == (2.0, 2.2)
assert find_closest_pair_across_lists([[1.0, 2.0, 3.0], [4.0, 5.0, 2.0]]) == (2.0, 2.0)
assert find_closest_pair_across_lists([[1.0, 2.0, 3.0], [4.0, 5.0, 1.1]]) == (1.0, 1.1)
assert find_closest_pair_across_lists([[1.0, 2.0, 3.0], [4.0, 5.0, 1.0]]) == (1.0, 1.0)