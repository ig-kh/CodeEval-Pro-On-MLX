from typing import List


def intersperse(numbers: List[int], delimeter: int) -> List[int]:
    """ Insert a number 'delimeter' between every two consecutive elements of input list `numbers'
    >>> intersperse([], 4)
    []
    >>> intersperse([1, 2, 3], 4)
    [1, 4, 2, 4, 3]
    """
    if not numbers:
        return []
    
    result = [numbers[0]]
    
    for num in numbers[1:]:
        result.append(delimeter)
        result.append(num)
    
    return result


def intersperse_twice(numbers: List[int], delimeter1: int, delimeter2: int) -> List[int]:
    """ Given a list of integers, intersperse a specified delimiter between every two consecutive elements of the list. Then, repeat this process with a different delimiter for the resulting list. Finally, return the list after both interspersing operations.
    >>> intersperse_twice([1, 2, 3], 4, 5)
    [1, 4, 2, 5, 3, 4, 5, 2, 5, 3]
    """
    intermediate = intersperse(numbers, delimeter1)
    return intersperse(intermediate, delimeter2)


assert double_intersperse([], 4, 5) == []
assert double_intersperse([1, 2, 3], 4, 5) == [1, 5, 4, 5, 2, 5, 4, 5, 3]
assert double_intersperse([1, 1, 1], 2, 3) == [1, 3, 2, 3, 1, 3, 2, 3, 1]
assert double_intersperse([1], 4, 5) == [1]
assert double_intersperse([1, 2], 4, 5) == [1, 5, 4, 5, 2]