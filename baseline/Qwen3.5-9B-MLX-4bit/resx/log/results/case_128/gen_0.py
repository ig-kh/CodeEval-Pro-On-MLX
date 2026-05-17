
def prod_signs(arr):
    """
    You are given an array arr of integers and you need to return
    sum of magnitudes of integers multiplied by product of all signs
    of each number in the array, represented by 1, -1 or 0.
    Note: return None for empty arr.

    Example:
    >>> prod_signs([1, 2, 2, -4]) == -9
    >>> prod_signs([0, 1]) == 0
    >>> prod_signs([]) == None
    """
    if not arr:
        return None
    
    # Calculate the product of signs
    sign_product = 1
    for num in arr:
        if num > 0:
            sign_product *= 1
        elif num < 0:
            sign_product *= -1
        else:
            # If there's a 0, the product of signs is 0
            return 0
    
    # Calculate the sum of magnitudes
    magnitude_sum = sum(abs(num) for num in arr)
    
    # Return the product of signs multiplied by the sum of magnitudes
    return sign_product * magnitude_sum

def sum_prod_signs(list_of_lists):
    """
    Given a list of lists of integers, calculate the sum of the results of the 'prod_signs' function applied to each sublist. If any sublist is empty, it should be ignored in the sum. The 'prod_signs' function returns the sum of magnitudes of integers multiplied by the product of all signs of each number in the array, represented by 

assert sum_prod_signs([[1, 2, 2, -4], [0, 1], []]) == -9
assert sum_prod_signs([[1, 2, 3], [-1, -2, -3], [0, 0, 0]]) == 0
assert sum_prod_signs([[], [], []]) == 0
assert sum_prod_signs([[1, -1, 1, -1], [2, -2, 2, -2]]) == 12
assert sum_prod_signs([[10, -20, 30], [40, -50, 60], [70, -80, 90]]) == -450