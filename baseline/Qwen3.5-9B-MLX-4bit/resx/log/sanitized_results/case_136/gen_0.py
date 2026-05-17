
def largest_smallest_integers(lst):
    '''
    Create a function that returns a tuple (a, b), where 'a' is
    the largest of negative integers, and 'b' is the smallest
    of positive integers in a list.
    If there is no negative or positive integers, return them as None.

    Examples:
    largest_smallest_integers([2, 4, 1, 3, 5, 7]) == (None, 1)
    largest_smallest_integers([]) == (None, None)
    largest_smallest_integers([0]) == (None, None)
    '''
    import re

    largest_negative = None
    smallest_positive = None

    for num in lst:
        if num < 0:
            if largest_negative is None or num > largest_negative:
                largest_negative = num
        elif num > 0:
            if smallest_positive is None or num < smallest_positive:
                smallest_positive = num

    return (largest_negative, smallest_positive)

def find_largest_smallest_across_lists(list_of_lists):
    """Given a list of lists, find the largest negative integer and the smallest positive integer across all sublists. Return a tuple (a, b), where 'a' is the largest negative integer and 'b' is the smallest positive integer. If there are no negative or positive integers, return them as None.
    """
    result = (None, None)

    for sublist in list_of_lists:
        # Get the largest negative and smallest positive from each sublist
        sub_result = largest_smallest_integers(sublist)

        # Update the overall result based on the sub_result
        a, b = sub_result

        if a is not None:
            if result[0] is None or a > result[0]:
                result = (a, b)
        else:
            if result[0] is None and a is not None:
                result = (a, b)

        if b is not None:
            if result[1] is None or b < result[1]:
                result = (a, b)
        else:
            if result[1] is None and b is not None:
                result = (a, b)

    return result


assert find_largest_smallest_across_lists([[2, 4, 1, 3, 5, 7], [-1, -3, -5, -7], [0]]) == (-1, 1)
assert find_largest_smallest_across_lists([[], [-1, -3, -5, -7], [0]]) == (-1, None)
assert find_largest_smallest_across_lists([[2, 4, 1, 3, 5, 7], [], [0]]) == (None, 1)
assert find_largest_smallest_across_lists([[0], [0], [0]]) == (None, None)