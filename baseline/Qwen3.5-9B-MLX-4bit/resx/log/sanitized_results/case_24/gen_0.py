

def largest_divisor(n: int) -> int:
    """ For a given number n, find the largest number that divides n evenly, smaller than n
    >>> largest_divisor(15)
    5
    """
    from typing import List

    if n <= 1:
        raise ValueError("Number must be greater than 1.")

    for i in range(n - 1, 0, -1):
        if n % i == 0:
            return i

    return 1


def largest_common_divisor(numbers: list) -> int:
    """ For a given list of integers, find the largest number that divides each of the numbers in the list evenly, smaller than the smallest number in the list
    >>> largest_common_divisor([15, 3
    """
    if not numbers:
        raise ValueError("List must contain at least one element.")

    min_num = min(numbers)
    result = largest_divisor(min_num)

    for num in numbers:
        if num % result != 0:
            return largest_divisor(num)

    return result


assert largest_common_divisor([15, 30, 45]) == 15
assert largest_common_divisor([10, 20, 30]) == 10
assert largest_common_divisor([7, 14, 21]) == 7
assert largest_common_divisor([16, 32, 64]) == 16
assert largest_common_divisor([9, 18, 27]) == 9