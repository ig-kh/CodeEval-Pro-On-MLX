
def next_smallest(lst):
    """
    You are given a list of integers.
    Write a function next_smallest() that returns the 2nd smallest element of the list.
    Return None if there is no such element.
    
    next_smallest([1, 2, 3, 4, 5]) == 2
    next_smallest([5, 1, 4, 3, 2]) == 2
    next_smallest([]) == None
    next_smallest([1, 1]) == None
    """
    import re

    if len(lst) < 2:
        return None

    # Sort the list to find the smallest and second smallest elements
    sorted_lst = sorted(lst)

    # Return the second element (index 1) after sorting
    return sorted_lst[1]

def next_smallest_in_lists(lst_of_lsts):
    """
    Given a list of lists of integers, write a function `next_smallest_in_lists()` that returns a list containing the 2nd smallest element from each sublist. If a sublist does not have a 2nd smallest element, append `None` to the result list for that sublist. The function should use the `next_smallest` function to find the 2nd smallest element for each sublist.
    """
    result = []

    for sublist in lst_of_lsts:
        # Use the next_smallest function to find the 2nd smallest element in each sublist
        result.append(next_smallest(sublist))

    return result


assert next_smallest_in_lists([[1, 2, 3, 4, 5], [5, 1, 4, 3, 2], [], [1, 1]]) == [2, 2, None, None]
assert next_smallest_in_lists([[10, 20, 30], [40, 50], [60]]) == [20, 50, None]
assert next_smallest_in_lists([[1, 1, 1], [2, 2], [3]]) == [None, None, None]
assert next_smallest_in_lists([[7, 8, 9], [10, 11, 12], [13, 14, 15]]) == [8, 11, 14]