from typing import List


def remove_duplicates(numbers: List[int]) -> List[int]:
    """ From a list of integers, remove all elements that occur more than once.
    Keep order of elements left the same as in the input.
    >>> remove_duplicates([1, 2, 3, 2, 4])
    [1, 3, 4]
    """
    seen = set()
    result = []
    
    for num in numbers:
        if num in seen:
            continue
        else:
            seen.add(num)
            result.append(num)
    
    return result


def remove_duplicates_across_lists(lists: List[List[int]]) -> List[List[int]]:
    """ Given a list of lists of integers, remove all elements that occur more than once across all lists. Keep the order of elements left the same as in the input. For example, given [[1, 2, 3, 2, 4], [2, 3, 5]], the output should be [[1, 4], [5]].
    >>> remove_duplicates_across_lists([[1, 2, 3, 2, 4], [2, 3, 5]])
    [[1, 4], [5]]
    """
    # Flatten all lists into a single list
    flattened = []
    for sublist in lists:
        flattened.extend(sublist)
    
    # Remove duplicates from the flattened list
    unique_elements = remove_duplicates(flattened)
    
    # Filter each sublist to keep only the unique elements
    result = []
    for sublist in lists:
        filtered_sublist = remove_duplicates(sublist)
        result.append(filtered_sublist)
    
    return result


assert remove_duplicates_across_lists([[1, 2, 3, 2, 4], [2, 3, 5]]) == [[1, 4], [5]]
assert remove_duplicates_across_lists([[1, 1, 2, 2], [3, 3, 4, 4]]) == [[], []]
assert remove_duplicates_across_lists([[1, 2, 3], [4, 5, 6]]) == [[1, 2, 3], [4, 5, 6]]
assert remove_duplicates_across_lists([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]]) == [[], []]