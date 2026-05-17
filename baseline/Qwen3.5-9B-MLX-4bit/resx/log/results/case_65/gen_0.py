
def circular_shift(x, shift):
    """Circular shift the digits of the integer x, shift the digits right by shift
    and return the result as a string.
    If shift > number of digits, return digits reversed.
    >>> circular_shift(12, 1)
    "21"
    >>> circular_shift(12, 2)
    "12"
    """
    if x < 0:
        raise ValueError("x must be non-negative")
    s = str(x)
    n = len(s)
    if shift >= n:
        return s[::-1]
    return s[-shift:] + s[:-shift]

def circular_shift_list(lst, shift):
    """circular_shift_list is a function that takes a list of integers and a shift value, and returns the list of transformed integers as strings.
    >>> circular_shift_list([12, 345], 1)
    ['21', '534']
    >>> circular_shift_list([12, 345], 2)
    ['12', '534']
    >>> circular_shift_list([12, 345], 3)
    ['21', '534']
    >>> circular_shift_list([12, 345], 4)
    ['21', '534']
    >>> circular_shift_list([12, 345], 5)
    ['21', '534']
    >>> circular_shift_list([12, 345], 6)
    ['21', '534']
    >>> circular_shift_list([12, 

assert circular_shift_list([12, 34, 56], 1) == ['21', '43', '65']
assert circular_shift_list([12, 34, 56], 2) == ['12', '34', '56']
assert circular_shift_list([123, 456, 789], 4) == ['321', '654', '987']