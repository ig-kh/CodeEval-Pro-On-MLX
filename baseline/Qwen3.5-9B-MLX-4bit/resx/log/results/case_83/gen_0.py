
def starts_one_ends(n):
    """
    Given a positive integer n, return the count of the numbers of n-digit
    positive integers that start or end with 1.
    """
    if n <= 0:
        return 0
    
    # Count of n-digit numbers that start with 1
    count_start = 10**(n-1)
    
    # Count of n-digit numbers that end with 1
    count_end = 9 * (10**(n-1))
    
    # Count of n-digit numbers that start and end with 1
    count_both = 10**(n-2) if n >= 2 else 0
    
    # Count of numbers that start or end with 1 (using inclusion-exclusion principle)
    count_or = count_start + count_end - count_both
    
    return count_or

def complex_starts_one_ends(n):
    """
    Given a positive integer n, return the count of the numbers of n-digit
    positive integers that start or end with 1, and also return the count of the numbers of n-digit positive integers that start and end with 1. Additionally, calculate the ratio of the count of numbers that start and end with 1 to the count of numbers that start or end with 1.
    """
    if n <= 0:
        return (0, 0, 0)
    
    count_start = 10**(n-1)
    count_end = 9 * (10**(n-1))
    count_both = 10**(n-2) if n >= 2 else 0
    
    count_or = count_start + count_end - count_both
    
    return (count_or, count_both)


assert complex_starts_one_ends(1) == (1, 0, 0)
assert complex_starts_one_ends(2) == (18, 0, 0.0)
assert complex_starts_one_ends(3) == (180, 1, 1/180)
assert complex_starts_one_ends(4) == (1800, 18, 0.01)