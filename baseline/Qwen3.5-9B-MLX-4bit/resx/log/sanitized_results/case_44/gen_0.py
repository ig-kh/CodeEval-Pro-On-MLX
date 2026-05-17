

def change_base(x: int, base: int):
    """Change numerical base of input number x to base.
    return string representation after the conversion.
    base numbers are less than 10.
    >>> change_base(8, 3)
    '22'
    >>> change_base(8, 2)
    '1000'
    >>> change_base(7, 2)
    '111'
    """
    import math

    if base < 2:
        raise ValueError("Base must be at least 2.")

    if x == 0:
        return "0"

    is_negative = x < 0
    if is_negative:
        x = -x

    digits = []
    while x > 0:
        digits.append(str(x % base))
        x //= base

    result = "".join(digits[::-1])

    if is_negative:
        result = "-" + result

    return result

def convert_and_concatenate(numbers: list, base: int) -> str:
    """
    Given a list of integers and a target base, convert each integer in the list to the target base 
    and concatenate the results into a single string. If the list contains any number that cannot be 
    represented in the given base (i.e., the number is negative or the base is less than 2), 
    return 'Invalid input'. Finally, reverse the concatenated string and return it.

    Note: This solution assumes that all numbers in the list are non-negative and base >= 2.
    If negative numbers or invalid bases are present, it returns 'Invalid input'.
    """
    # Validate base
    if base < 2:
        return "Invalid input"

    # Validate numbers and convert
    converted_parts = []
    for num in numbers:
        if num < 0:
            return "Invalid input"
        converted_parts.append(change_base(num, base))

    # Concatenate and reverse
    result = "".join(converted_parts)
    return result[::-1]


assert convert_and_concatenate([8, 7], 2) == '1110001'  # 1000 (8) + 111 (7) -> reverse -> 1110001
assert convert_and_concatenate([10, 15, 3], 3) == '01021101'  # 101 (10) + 120 (15) + 10 (3) -> reverse -> 101120
assert convert_and_concatenate([5, -3, 2], 2) == 'Invalid input'  # Negative number
assert convert_and_concatenate([10, 20, 30], 1) == 'Invalid input'  # Base less than 2
assert convert_and_concatenate([1, 2, 3], 10) == '321'  # 1 + 2 + 3 -> reverse -> 321
