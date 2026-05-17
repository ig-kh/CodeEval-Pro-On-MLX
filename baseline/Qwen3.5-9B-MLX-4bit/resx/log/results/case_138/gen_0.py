
def is_equal_to_sum_even(n):
    """Evaluate whether the given number n can be written as the sum of exactly 4 positive even numbers
    Example
    is_equal_to_sum_even(4) == False
    is_equal_to_sum_even(6) == False
    is_equal_to_sum_even(8) == True
    """
    # The smallest sum of 4 positive even numbers is 2 + 2 + 2 + 2 = 8
    # Also, the sum must be even because it's a sum of even numbers
    return n >= 8 and n % 2 == 0

def can_each_be_sum_of_even(numbers):
    """Given a list of integers, determine if each integer can be expressed as the sum of exactly 4 positive even numbers. Return a list of boolean values where each value corresponds to whether the integer at the same index in the input list can be expressed as such.
    """
    result = []
    for num in numbers:
        result.append(is_equal_to_sum_even(num))
    
    return result


assert can_each_be_sum_of_even([4, 6, 8, 10, 12]) == [False, False, True, True, True]