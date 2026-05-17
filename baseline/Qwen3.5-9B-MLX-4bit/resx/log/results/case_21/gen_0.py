from typing import List


def rescale_to_unit(numbers: List[float]) -> List[float]:
    """ Given list of numbers (of at least two elements), apply a linear transform to that list,
    such that the smallest number will become 0 and the largest will become 1
    >>> rescale_to_unit([1.0, 2.0, 3.0, 4.0, 5.0])
    [0.0, 0.25, 0.5, 0.75, 1.0]
    """
    if len(numbers) < 2:
        raise ValueError("List must contain at least two elements.")
    
    min_val = min(numbers)
    max_val = max(numbers)
    
    if min_val == max_val:
        return [0.0] * len(numbers)
    
    rescaled = [(num - min_val) / (max_val - min_val) for num in numbers]
    return rescaled


def rescale_to_unit_across_lists(lists: List[List[float]]) -> List[float]:
    """ Given a list of lists of numbers (each sublist containing at least two elements), apply a linear transform to each sublist such that the smallest number in each sublist will become 0 and the largest will become 1. Then, combine all the transformed sublists into a single list and apply the same linear transform to this combined list, ensuring the smallest number in the combined list becomes 0 and the largest becomes 1.
    >>> rescale_to_unit_across_lists([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    [0.0, 0.5, 0.0, 0.5, 0.0, 1.0]
    """
    if not lists:
        raise ValueError("Input list of lists must not be empty.")
    
    # Step 1: Rescale each sublist individually
    rescaled_lists = []
    for sublist in lists:
        if len(sublist) < 2:
            raise ValueError("Each sublist must contain at least two elements.")
        rescaled_lists.append(rescale_to_unit(sublist))
    
    # Step 2: Combine all rescaled sublists into a single list
    combined_list = []
    for rescaled_list in rescaled_lists:
        combined_list.extend(rescaled_list)
    
    # Step 3: Apply the same linear transform to the combined list
    return rescale_to_unit(combined_list)


assert rescale_nested_lists([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]) == [0.0, 0.5, 1.0, 0.0, 0.5, 1.0, 0.0, 0.5, 1.0]
assert rescale_nested_lists([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]) == [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]