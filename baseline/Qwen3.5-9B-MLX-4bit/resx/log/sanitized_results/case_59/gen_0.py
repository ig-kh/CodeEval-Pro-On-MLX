

def largest_prime_factor(n: int):
    """Return the largest prime factor of n. Assume n > 1 and is not a prime.
    >>> largest_prime_factor(13195)
    29
    >>> largest_prime_factor(2048)
    2
    """
    import itertools

    d = 2
    while d * d <= n:
        while n % d == 0:
            n //= d
            d += 1
    return n


def product_of_largest_prime_factors(numbers: list):
    """
    Given a list of integers, find the product of the largest prime factors of each integer. If an integer is

assert product_of_largest_prime_factors([13195, 2048]) == 58
assert product_of_largest_prime_factors([1, 2, 3, 4, 5]) == 2
assert product_of_largest_prime_factors([10, 15, 21, 25, 30]) == 4375
assert product_of_largest_prime_factors([100, 101, 102, 103, 104]) == 1105
