from typing import List


def factorize(n: int) -> List[int]:
    """ Return list of prime factors of given integer in the order from smallest to largest.
    Each of the factors should be listed number of times corresponding to how many times it appeares in factorization.
    Input number should be equal to the product of all factors
    >>> factorize(8)
    [2, 2, 2]
    >>> factorize(25)
    [5, 5]
    >>> factorize(70)
    [2, 5, 7]
    """
    import re
    from typing import List

    if n <= 1:
        raise ValueError("Number must be greater than 1.")

    factors = []
    divisor = 2

    while n > 1:
        while n % divisor == 0:
            factors.append(divisor)
            n //= divisor
        divisor += 1

    return factors


def factorize_list(numbers: List[int]) -> List[List[int]]:
    """ Given a list of integers, return a list of lists where each sublist contains the prime factors of the corresponding integer in the original list. The prime factors should be listed in ascending order, and each factor should appear as many times as it does in the factorization of the integer.
    >>> factorize_list([8, 25, 70])
    [[2, 2, 2], [5, 5], [2, 5, 7]]
    """
    return [factorize(num) for num in numbers]


assert factorize_list([8, 25, 70]) == [[2, 2, 2], [5, 5], [2, 5, 7]]
assert factorize_list([10, 12, 14]) == [[2, 5], [2, 2, 3], [2, 7]]
assert factorize_list([1, 2, 3]) == [[], [2], [3]]
assert factorize_list([49, 64, 81]) == [[7, 7], [2, 2, 2, 2, 2, 2], [3, 3, 3, 3]]