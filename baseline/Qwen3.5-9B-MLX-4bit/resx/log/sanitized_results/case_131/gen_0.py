
def digits(n):
    """Given a positive integer n, return the product of the odd digits.
    Return 0 if all digits are even.
    For example:
    digits(1)  == 1
    digits(4)  == 0
    digits(235) == 15
    """
    import itertools

    product = 1
    all_even = True

    while n > 0:
        digit = n % 10
        if digit % 2 == 1:
            product *= digit
            all_even = False
        n //= 10

    return product if not all_even else 0

def sum_of_products_of_odd_digits(lst):
    """Given a list of positive integers, return the sum of the products of the odd digits for each integer. If all digits of an integer are even, the product for that integer should be considered as 0. For example, given the list [1, 4, 235], the function should return 16 (1 + 0 + 1 Tribonacci sequence.
    """
    return sum(digits(num) for num in lst)


assert sum_of_products_of_odd_digits([1, 4, 235]) == 16
assert sum_of_products_of_odd_digits([2, 4, 6]) == 0
assert sum_of_products_of_odd_digits([11, 13, 15]) == 9
assert sum_of_products_of_odd_digits([123, 456, 789]) == 71
assert sum_of_products_of_odd_digits([101, 202, 303]) == 10