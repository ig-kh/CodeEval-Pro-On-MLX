def greatest_common_divisor(a: int, b: int) -> int:
    """ Return a greatest common divisor of two integers a and b
    >>> greatest_common_divisor(3, 5)
    1
    >>> greatest_common_divisor(25, 15)
    5
    """
    from typing import List

    if b == 0:
        return a
    return greatest_common_divisor(b, a % b)


def gcd_of_list(lst: List[int]) -> int:
    """ Given a list of integers, find the greatest common divisor (GCD) of all the integers in the list.
    """
    if not lst:
        raise ValueError("The list must contain at least one integer.")

    result = lst[0]
    for num in lst[1:]:
        result = greatest_common_divisor(result, num)

    return result


assert gcd_of_list([3, 5]) == 1
assert gcd_of_list([25, 15]) == 5
assert gcd_of_list([12, 18, 24]) == 6
assert gcd_of_list([7, 13, 19]) == 1
assert gcd_of_list([0, 10, 20]) == 10
assert gcd_of_list([]) == 0