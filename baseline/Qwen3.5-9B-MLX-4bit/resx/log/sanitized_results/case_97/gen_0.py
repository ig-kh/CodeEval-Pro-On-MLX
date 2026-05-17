
def multiply(a, b):
    """Complete the function that takes two integers and returns 
    the product of their unit digits.
    Assume the input is always valid.
    Examples:
    multiply(148, 412) should return 16.
    multiply(19, 28) should return 72.
    multiply(2020, 1851) should return 0.
    multiply(14,-15) should return 20.
    """
    import itertools

    unit_digit_a = abs(a) % 10
    unit_digit_b = abs(b) % 10

    return unit_digit_a * unit_digit_b

def sum_of_products_of_unit_digits(tuples_list):
    """
    Given a list of tuples, where each tuple contains two integers, return the sum of the products of the unit digits of each tuple. For example, if the list is [(148, 412), (19, 28), (2020, 1851)], the function should return the sum of the products of the unit digits of each tuple, which is 16 + 72 + 0 = 88.
    """
    total_sum = 0

    for a, b in tuples_list:
        total_sum += multiply(a, b)

    return total_sum


assert sum_of_products_of_unit_digits([(148, 412), (19, 28), (2020, 1851)]) == 88
assert sum_of_products_of_unit_digits([(14, -15), (10, 10), (9, 9)]) == 101
assert sum_of_products_of_unit_digits([(123, 456), (789, 123), (456, 789)]) == 99
assert sum_of_products_of_unit_digits([(0, 0), (1, 1), (2, 2)]) == 5