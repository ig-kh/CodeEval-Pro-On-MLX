
def add(lst):
    """Given a non-empty list of integers lst. add the even elements that are at odd indices..


    Examples:
        add([4, 2, 6, 7]) ==> 2 
    """
    total = 0
    for i in range(1, len(lst), 2):
        if lst[i] % 2 == 0:
            total += lst[i]
    
    return total

def total_even_odd_sum(matrix):
    """Given a list of lists of integers, where each sublist represents a row in a matrix, calculate the sum of all even elements that are located at odd indices in each row. Return the total sum of these elements across all rows."""
    total_sum = 0
    
    for row in matrix:
        total_sum += add(row)
    
    return total_sum


assert total_even_odd_sum([[4, 2, 6, 7], [1, 3, 5, 8], [9, 10, 11, 12]]) == 32
assert total_even_odd_sum([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) == 10
assert total_even_odd_sum([[10, 11], [12, 13, 14], [15, 16]]) == 16