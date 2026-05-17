
def count_nums(arr):
    """
    Write a function count_nums which takes an array of integers and returns
    the number of elements which has a sum of digits > 0.
    If a number is negative, then its first signed digit will be negative:
    e.g. -123 has signed digits -1, 2, and 3.
    >>> count_nums([]) == 0
    >>> count_nums([-1, 11, -11]) == 1
    >>> count_nums([1, 1, 2]) == 3
    """
    count = 0
    for num in arr:
        if num == 0:
            continue
        
        temp = abs(num)
        digit_sum = 0
        while temp > 0:
            digit_sum += temp % 10
            temp //= 10
        
        if num < 0:
            digit_sum -= 2 * (digit_sum // 10) + 2 * (digit_sum % 10)
        
        if digit_sum > 0:
            count += 1
    
    return count

def count_nums_in_matrix(matrix):
    """
    Write a function `count_nums_in_matrix` which takes a 2D array (matrix) of integers and returns the number of elements in the matrix which have a sum of digits > 0. If a number is negative, then its first signed digit will be negative: e.g. -123 has signed digits -1, 

assert count_nums_in_matrix([]) == 0
assert count_nums_in_matrix([[-1, 11, -11], [1, 1, 2]]) == 4
assert count_nums_in_matrix([[1, 1, 2], [-1, -2, -3]]) == 3
assert count_nums_in_matrix([[0, 0, 0], [0, 0, 0]]) == 0
assert count_nums_in_matrix([[123, -456, 789], [10, -20, 30]]) == 5