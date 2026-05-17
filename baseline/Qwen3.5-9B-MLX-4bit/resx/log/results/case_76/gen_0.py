
def is_simple_power(x, n):
    """Your task is to write a function that returns true if a number x is a simple
    power of n and false in other cases.
    x is a simple power of n if n**int=x
    For example:
    is_simple_power(1, 4) => true
    is_simple_power(2, 2) => true
    is_simple_power(8, 2) => true
    is_simple_power(3, 2) => false
    is_simple_power(3, 1) => false
    is_simple_power(5, 3) => false
    """
    if x == 1:
        return True
    
    if n == 0 or n == 1:
        return False
    
    # Check if x is a power of n
    temp = x
    while temp % n == 0:
        temp //= n
    
    return temp == 1

def check_powers(numbers, base):
    """Given a list of numbers and a base number, determine if each number in the list is a simple power of the base number. Return a list of boolean values indicating whether each number is a simple power of the base number or not."""
    results = []
    for num in numbers:
        results.append(is_simple_power(num, base))
    
    return results


assert check_powers([1, 2, 8, 3, 5], 2) == [True, True, True, False, False]
assert check_powers([16, 81, 49, 100], 4) == [True, False, False, False]
assert check_powers([1, 1, 1, 1], 1) == [True, True, True, True]
assert check_powers([], 3) == []