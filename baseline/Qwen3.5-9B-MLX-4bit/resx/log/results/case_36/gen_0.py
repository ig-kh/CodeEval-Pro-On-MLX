

def fizz_buzz(n: int):
    """Return the number of times the digit 7 appears in integers less than n which are divisible by 11 or 13.
    >>> fizz_buzz(50)
    0
    >>> fizz_buzz(78)
    2
    >>> fizz_buzz(79)
    3
    """
    count = 0
    for i in range(1, n):
        if i % 11 == 0 or i % 1

assert new_fizz_buzz([50, 78, 79]) == 5
assert new_fizz_buzz([100, 200, 300]) == 19
assert new_fizz_buzz([11, 13, 22, 26]) == 0
assert new_fizz_buzz([77, 91, 143, 154]) == 11
assert new_fizz_buzz([176, 234, 286, 319]) == 29