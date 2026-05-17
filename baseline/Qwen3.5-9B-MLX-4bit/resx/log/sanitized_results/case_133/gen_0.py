

def sum_squares(lst):
    """You are given a list of numbers.
    You need to return the sum of squared numbers in the given list,
    round each element in the list to the upper int(Ceiling) first.
    Examples:
    For lst = [1,2,3] the output should be 14
    For lst = [1,4,9] the output should be 98
    For lst = [1,3,5,7] the output should be 84
    For lst = [1.4,4.2,0] the output should be 29
    For lst = [-2.4,1,1] the output should be 6
    

    """
    import math
    import re


    total = 0
    for num in lst:
        # Ceiling the number
        ceiling_num = math.ceil(num)
        total += ceiling_num ** 2

    return total

def sum_of_sum_squares(lst_of_lsts):
    """You are given a list of lists of numbers. You need to return the sum of the sums of squared numbers in each sublist, where each element in the sublist is rounded to the upper integer (Ceiling) first. For example, for the input [[1,2,3], [1,4,

assert sum_of_sum_squares([[1,2,3], [1,4,9]]) == 112
assert sum_of_sum_squares([[1,3,5,7], [1.4,4.2,0]]) == 113
assert sum_of_sum_squares([[-2.4,1,1], [1,2,3]]) == 20
assert sum_of_sum_squares([[1,4,9], [1,3,5,7]]) == 182
assert sum_of_sum_squares([[1.4,4.2,0], [-2.4,1,1]]) == 35