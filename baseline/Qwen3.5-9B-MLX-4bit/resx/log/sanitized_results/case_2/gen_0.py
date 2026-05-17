

def truncate_number(number: float) -> float:
    """ Given a positive floating point number, it can be decomposed into
    and integer part (largest integer smaller than given number) and decimals
    (leftover part always smaller than 1).

    Return the decimal part of the number.
    >>> truncate_number(3.5)
    0.5
    """
    import itertools
    import math

    return number - math.floor(number)


def sum_of_parts(numbers: list) -> float:
    """ Given a list of positive floating point numbers, decompose each number into its integer part and decimal part. Then, calculate the sum of all the integer parts and the sum of all the decimal parts separately. Finally, return the product of these two sums.
    >>> sum_of_parts([1.5, 2.5, 3.5])
    105.0
    """
    int_sum = 0
    dec_sum = 0

    for num in numbers:
        int_part = math.floor(num)
        dec_part = truncate_number(num)
        int_sum += int_part
        dec_sum += dec_part

    return int_sum * dec_sum


assert math.isclose(sum_of_parts([3.5, 2.7, 1.1]),7.8)
assert math.isclose(sum_of_parts([0.5, 1.5, 2.5]), 4.5)
assert math.isclose(sum_of_parts([10.0, 20.0, 30.0]), 0.0)
assert math.isclose(sum_of_parts([1.23, 4.56, 7.89]), 20.16)
