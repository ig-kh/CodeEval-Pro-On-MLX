
def unique_digits(x):
    """Given a list of positive integers x. return a sorted list of all 
    elements that hasn't any even digit.

    Note: Returned list should be sorted in increasing order.
    
    For example:
    >>> unique_digits([15, 33, 1422, 1])
    [1, 15, 33]
    >>> unique_digits([152, 323, 1422, 10])
    []
    """
    import re

    result = []
    for num in x:
        has_even_digit = False
        temp = num
        while temp > 0:
            digit = temp % 10
            if digit % 2 == 0:
                has_even_digit = True
                break
            temp //= 10
        if not has_even_digit:
            result.append(num)
    return sorted(result)

def unique_digits_across_lists(list_of_lists):
    """
    Given a list of lists of positive integers, return a sorted list of all elements that do not have any even digit across all the lists. The returned list should be sorted in increasing order.
    """
    result = []
    for sublist in list_of_lists:
        result.extend(unique_digits(sublist))
    return sorted(result)


assert unique_digits_across_lists([[15, 33, 1422, 1], [152, 323, 1422, 10]]) == [1, 15, 33]
assert unique_digits_across_lists([[152, 323, 1422, 10], [15, 33, 1422, 1]]) == [1, 15, 33]
assert unique_digits_across_lists([[152, 323, 1422, 10], [152, 323, 1422, 10]]) == []