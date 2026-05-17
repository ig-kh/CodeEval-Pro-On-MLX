from typing import List


def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """
    if len(numbers) < 2:
        return False
    
    numbers.sort()
    
    for i in range(len(numbers) - 1):
        if abs(numbers[i] - numbers[i + 1]) < threshold:
            return True
    
    return False


def find_close_elements_lists(list_of_lists: List[List[float]], threshold: float) -> List[int]:
    """ Given a list of lists of floats, determine if there exists any list where at least two numbers are closer to each other than a given threshold. If such a list meets the condition, return the indices of the lists where this condition is met. If no such list exists, return an empty list.
    >>> find_close_elements_lists([[1.0, 2.0, 3.0], [1.0, 2.8, 3.0, 4.0, 5.0, 2.0]], 0.5)
    [1]
    >>> find_close_elements_lists([[1.0, 2.0, 3.], [1.0, 2.8, 3.0, 4.0, 5.0, 2.0]], 0.3)
    [1]
    >>> find_close


assert find_close_elements_lists([[1.0, 2.0, 3.0], [1.0, 2.8, 3.0, 4.0, 5.0, 2.0]], 0.5) == [1]
assert find_close_elements_lists([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], 0.3) == []
assert find_close_elements_lists([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], 0.5) == []