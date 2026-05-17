

def modp(n: int, p: int):
    """Return 2^n modulo p (be aware of numerics).
    >>> modp(3, 5)
    3
    >>> modp(1101, 101)
    2
    >>> modp(0, 101)
    1
    >>> modp(3, 11)
    8
    >>> modp(100, 101)
    1
    """
    if p == 1:
        return 0
    
    result = 1
    base = 2
    
    exponent = n
    while exponent > 0:
        if exponent % 2 == 1:
            result = (result * base) % p
        base = (base * base) % p
        exponent //= 2
    
    return result

def sum_modp_list(tuples_list):
    """
    Given a list of tuples where each tuple contains two integers (n, p), compute the sum of all 2^n modulo p 
    for each tuple in the list. Return the final sum modulo 123.
    
    >>> sum_modp_list([(3, 5), (0, 101), (2, 7)])
    123
    """
    total_sum = 0
    
    for n, p in tuples_list:
        total_sum = (total_sum + modp(n, p)) % 123
    
    return total_sum


assert sum_modp_list([(3, 5), (1101, 101), (0, 101), (3, 11), (100, 101)]) == 15
assert sum_modp_list([(1, 2), (2, 3), (3, 4), (4, 5)]) == 2
assert sum_modp_list([(10, 11), (100, 101), (1000, 1001)]) == 72
assert sum_modp_list([(0, 1), (0, 2), (0, 3), (0, 4)]) == 4