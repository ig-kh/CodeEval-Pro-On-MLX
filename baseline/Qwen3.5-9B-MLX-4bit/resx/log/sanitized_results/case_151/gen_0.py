
def double_the_difference(lst):
    '''
    Given a list of numbers, return the sum of squares of the numbers
    in the list that are odd. Ignore numbers that are negative or not integers.
    
    double_the_difference([1, 3, 2, 0]) == 1 + 9 + 0 + 0 = 10
    double_the_difference([-1, -2, 0]) == 0
    double_the_difference([9, -2]) == 81
    double_the_difference([0]) == 0  
   
    If the input list is empty, return 0.
    '''
    import re

    total_sum = 0
    for num in lst:
        if isinstance(num, int) and num >= 0 and num % 2 == 1:
            total_sum += num ** 2
    return total_sum

def sum_of_double_differences(lst_of_lsts):
    '''
    Given a list of lists of numbers, return the sum of squares of the numbers in each sublist that are odd. Ignore numbers that are `double_the_difference` function.
    '''
    total_sum = 0
    for sublist in lst_of_lsts:
        total_sum += double_the_difference(sublist)
    return total_sum


assert sum_of_double_differences([[1, 3, 2, 0], [-1, -2, 0], [9, -2], [0]]) == 91
assert sum_of_double_differences([[], [1, 3], [5, 7, 9], [-1, -3, -5]]) == 165
assert sum_of_double_differences([[0], [0], [0]]) == 0
assert sum_of_double_differences([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]) == 165