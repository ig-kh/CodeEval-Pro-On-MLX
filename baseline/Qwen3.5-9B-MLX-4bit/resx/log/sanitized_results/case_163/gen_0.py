
def generate_integers(a, b):
    """
    Given two positive integers a and b, return the even digits between a
    and b, in ascending order.

    For example:
    generate_integers(2, 8) => [2, 4, 6, 8]
    generate_integers(8, 2) => [2, 4, 6, 8]
    generate_integers(10, 14) => []
    """
    import re

    # Ensure a is less than or equal to b
    start, end = sorted([a, b])

    result = []
    for num in range(start, end + 1):
        # Check if the number is even
        if num % 2 == 0:
            result.append(num)

    return result

def generate_even_digits_lists(tuples_list):
    """
    Given a list of tuples, where each tuple contains two positive integers, return a list of lists where each sublist contains the even digits between the two integers in the corresponding tuple, in ascending order. If a tuple contains integers that are the same, the sublist should be an empty list.
    """
    even_digits_lists = []

    for a, b in tuples_list:
        if a == b:
            even_digits_lists.append([])
        else:
            even_digits_lists.append(generate_integers(a, b))

    return even_digits_lists


assert generate_even_digits_lists([(2, 8), (8, 2), (10, 14)]) == [[2, 4, 6, 8], [2, 4, 6, 8], []]
assert generate_even_digits_lists([(1, 5), (6, 10), (12, 12)]) == [[2, 4], [6, 8], []]
assert generate_even_digits_lists([(1, 1), (2, 2), (3, 3)]) == [[], [2], []]