

def is_prime(n):
    """Return true if a given number is prime, and false otherwise.
    >>> is_prime(6)
    False
    >>> is_prime(101)
    True
    >>> is_prime(11)
    True
    >>> is_prime(13441)
    True
    >>> is_prime(61)
    True
    >>> is_prime(4)
    False
    >>> is_prime(1)
    False
    """
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def prime_check_list(numbers):
    """Write a function that takes a list of numbers and returns a list of tuples. Each tuple should contain a number from the input list and a boolean indicating whether the number is prime. The function should use the is_prime function to determine if a prime.
    >>> prime_check_list([1, 2, 3, 4, 5])
    [(1, False), (2, True), (3, True), (4, False), (5, 0


assert prime_check_list([6, 101, 11, 13441, 61, 4, 1]) == [(6, False), (101, True), (11, True), (13441, True), (61, True), (4, False), (1, False)]