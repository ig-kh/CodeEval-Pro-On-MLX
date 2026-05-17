
def solution(lst):
    """Given a non-empty list of integers, return the sum of all of the odd elements that are in even positions.
    

    Examples
    solution([5, 8, 7, 1]) ==> 12
    solution([3, 3, 3, 3, 3]) ==> 9
    solution([30, 13, 24, 321]) ==>0
    """
    import re

    total_sum = 0

    # Iterate through the list with index
    for i, num in enumerate(lst):
        # Check if the element is odd and its position (index) is even
        if num % 2 != 0 and i % 2 == 0:
            total_sum += num

    return total_sum

def new_solution(lst_of_lsts):
    """Given a list of lists of integers, return the sum of all odd elements that are in even positions across all sublists. If a sublist is empty, it should be ignored.
    """
    total_sum = 0

    for sublist in lst_of_lsts:
        # Skip empty sublists
        if not sublist:
            continue

        # Call the solution function for each non-empty sublist
        total_sum += solution(sublist)

    return total_sum


assert new_solution([[5, 8, 7, 1], [3, 3, 3, 3, 3], [30, 13, 24, 321]]) == 21
assert new_solution([[], [1, 2, 3], [4, 5, 6, 7]]) == 4
assert new_solution([[10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20, 21]]) == 85
assert new_solution([[22, 23, 24, 25], [], [26, 27, 28, 29, 30]]) == 0