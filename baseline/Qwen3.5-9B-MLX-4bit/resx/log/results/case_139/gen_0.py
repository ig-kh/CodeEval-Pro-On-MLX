
def special_factorial(n):
    """The Brazilian factorial is defined as:
    brazilian_factorial(n) = n! * (n-1)! * (n-2)! * ... * 1!
    where n > 0

    For example:
    >>> special_factorial(4)
    288

    The function will receive an integer as input and should return the special
    factorial of this integer.
    """
    import math
    
    def factorial(k):
        return math.factorial(k)
    
    result = 1
    for i in range(1, n + 1):
        result *= factorial(i)
    
    return result

def sum_of_special_factorials(lst):
    """Given a list of integers, calculate the sum of the Brazilian factorials for each integer in the list. The Brazilian factorial is defined as: brazilian_factorial(n) = n! * (n-1)! * (n-2)! * ... * 1! where n > 0. For example, if the list is [1, 2, 3], the function should return the sum of the Brazilian factorials of 1, 

assert sum_of_special_factorials([1, 2, 3]) == 15
assert sum_of_special_factorials([4, 5]) == 34848
assert sum_of_special_factorials([1]) == 1
assert sum_of_special_factorials([2, 3, 4]) == 302
assert sum_of_special_factorials([10]) == 6658606584104736522240000000